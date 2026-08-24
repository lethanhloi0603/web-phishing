import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import AutoModel


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, value, strength):
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


def gradient_reverse(value, strength=1.0):
    return _GradientReversal.apply(value, strength)


class DOMTransformer(nn.Module):
    """Transformer over a bounded preorder DOM tag sequence (not a DOM graph)."""

    def __init__(
        self,
        vocab_size=512,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_len=256,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position = nn.Embedding(max_len, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(self, dom_ids, dom_mask):
        batch_size, length = dom_ids.shape
        positions = torch.arange(length, device=dom_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, length)
        value = self.embedding(dom_ids) + self.position(positions)
        value = self.encoder(value, src_key_padding_mask=dom_mask.eq(0))
        mask = dom_mask.unsqueeze(-1).float()
        return (value * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class PhishGenModel(nn.Module):
    """Multi-branch generator-invariant phishing website detector."""

    def __init__(
        self,
        model_name="google/mobilebert-uncased",
        dom_vocab_size=512,
        dom_hidden_size=128,
        dom_layers=2,
        dom_heads=4,
        max_dom_length=256,
        static_feature_size=27,
        static_hidden_size=64,
        fusion_hidden_size=256,
        projection_size=128,
        num_generators=0,
        dropout=0.2,
        use_html=True,
        use_dom=True,
        use_static=True,
        freeze_html=False,
        normalize_branches=False,
        fusion_type="concat",
        modality_dropout=0.0,
    ):
        super().__init__()
        if not any((use_html, use_dom, use_static)):
            raise ValueError("At least one input branch must be enabled")

        self.use_html = use_html
        self.use_dom = use_dom
        self.use_static = use_static
        self.freeze_html = freeze_html
        self.normalize_branches = normalize_branches
        self.fusion_type = fusion_type
        self.modality_dropout = float(modality_dropout)
        self.num_generators = num_generators
        branch_sizes = []
        if fusion_type not in {"concat", "gated"}:
            raise ValueError("fusion_type must be 'concat' or 'gated'")
        if not 0.0 <= self.modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")

        if use_html:
            self.html_encoder = AutoModel.from_pretrained(model_name)
            html_hidden = self.html_encoder.config.hidden_size
            branch_sizes.append(html_hidden)
            if freeze_html:
                for parameter in self.html_encoder.parameters():
                    parameter.requires_grad = False
            self.html_norm = nn.LayerNorm(html_hidden) if normalize_branches else nn.Identity()
        else:
            self.html_encoder = None

        if use_dom:
            self.dom_encoder = DOMTransformer(
                vocab_size=dom_vocab_size,
                hidden_size=dom_hidden_size,
                num_layers=dom_layers,
                num_heads=dom_heads,
                max_len=max_dom_length,
                dropout=dropout,
            )
            branch_sizes.append(dom_hidden_size)
            self.dom_norm = nn.LayerNorm(dom_hidden_size) if normalize_branches else nn.Identity()
        else:
            self.dom_encoder = None

        if use_static:
            self.static_encoder = nn.Sequential(
                nn.Linear(static_feature_size, static_hidden_size),
                nn.LayerNorm(static_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            branch_sizes.append(static_hidden_size)
            self.static_norm = nn.LayerNorm(static_hidden_size) if normalize_branches else nn.Identity()
        else:
            self.static_encoder = None

        if fusion_type == "gated":
            # Project every modality to an equal-sized space before weighting.
            # This prevents the 512-d HTML vector from dominating DOM/static
            # merely because it has more coordinates.
            self.branch_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(size, fusion_hidden_size),
                    nn.LayerNorm(fusion_hidden_size),
                    nn.GELU(),
                )
                for size in branch_sizes
            ])
            self.branch_gate = nn.Linear(fusion_hidden_size, 1)
            self.fusion = nn.Sequential(
                nn.LayerNorm(fusion_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.branch_projections = None
            self.branch_gate = None
            self.fusion = nn.Sequential(
                nn.Linear(sum(branch_sizes), fusion_hidden_size),
                nn.LayerNorm(fusion_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.classifier = nn.Linear(fusion_hidden_size, 1)
        self.projector = nn.Sequential(
            nn.Linear(fusion_hidden_size, fusion_hidden_size),
            nn.GELU(),
            nn.Linear(fusion_hidden_size, projection_size),
        )
        self.generator_discriminator = None
        if num_generators > 1:
            self.generator_discriminator = nn.Sequential(
                nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_size // 2, num_generators),
            )

    def forward(
        self,
        input_ids,
        attention_mask,
        dom_ids,
        dom_mask,
        static_features,
        grl_strength=1.0,
    ):
        branches = []
        if self.use_html:
            if self.freeze_html:
                # A frozen encoder must not inject dropout noise into the trainable
                # fusion head.  model.train() recursively enables dropout, so force
                # this branch back to inference mode for every forward pass.
                self.html_encoder.eval()
                with torch.no_grad():
                    html_output = self.html_encoder(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).last_hidden_state
            else:
                html_output = self.html_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
            html_mask = attention_mask.unsqueeze(-1).float()
            html_vector = (html_output * html_mask).sum(dim=1)
            html_vector = html_vector / html_mask.sum(dim=1).clamp_min(1.0)
            branches.append(self.html_norm(html_vector))

        if self.use_dom:
            branches.append(self.dom_norm(self.dom_encoder(dom_ids, dom_mask)))

        if self.use_static:
            branches.append(self.static_norm(self.static_encoder(static_features)))

        branch_weights = None
        if self.fusion_type == "gated":
            projected = torch.stack([
                projection(value)
                for projection, value in zip(self.branch_projections, branches)
            ], dim=1)
            gate_logits = self.branch_gate(projected).squeeze(-1)
            if self.training and self.modality_dropout > 0.0 and len(branches) > 1:
                keep = torch.rand_like(gate_logits).ge(self.modality_dropout)
                # Always retain at least one branch per sample.
                all_dropped = ~keep.any(dim=1)
                if all_dropped.any():
                    best = gate_logits[all_dropped].argmax(dim=1)
                    keep[all_dropped] = False
                    keep[all_dropped, best] = True
                gate_logits = gate_logits.masked_fill(~keep, torch.finfo(gate_logits.dtype).min)
            branch_weights = torch.softmax(gate_logits, dim=1)
            embedding = self.fusion(
                (projected * branch_weights.unsqueeze(-1)).sum(dim=1)
            )
        else:
            embedding = self.fusion(torch.cat(branches, dim=1))
        generator_logits = None
        if self.generator_discriminator is not None:
            generator_logits = self.generator_discriminator(
                gradient_reverse(embedding, grl_strength)
            )

        return {
            "logits": self.classifier(embedding).squeeze(-1),
            "projection": self.projector(embedding),
            "embedding": embedding,
            "generator_logits": generator_logits,
            "branch_weights": branch_weights,
        }
