import argparse
import json
import math
import random
import shutil
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import WebPageDataset, build_generator_mapping, read_frame
from src.losses import generator_aware_supcon_loss, phishing_domain_alignment_loss
from src.metrics import binary_metrics
from src.model import PhishGenModel
from src.preprocessing import STATIC_FEATURE_NAMES
from src.research import sha256_file, source_snapshot


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_forward(model, batch, device, grl_strength=1.0):
    return model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["dom_ids"].to(device),
        batch["dom_mask"].to(device),
        batch["static_features"].to(device),
        grl_strength=grl_strength,
    )


def clean_view_forward(model, batch, device, grl_strength=1.0):
    """Run the unmodified view paired with an artifact-augmented training page."""
    return model(
        batch["clean_input_ids"].to(device),
        batch["clean_attention_mask"].to(device),
        batch["clean_dom_ids"].to(device),
        batch["clean_dom_mask"].to(device),
        batch["clean_static_features"].to(device),
        grl_strength=grl_strength,
    )


def capture_torch_rng_state():
    state = {"cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        state["cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return state


def restore_torch_rng_state(state):
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def paired_stochastic_forwards(model, batch, device, grl_strength=1.0):
    """Use identical dropout/modality masks for augmented and clean views.

    The global RNG advances as if one forward pass occurred. This prevents the
    consistency objective from learning random regularization noise.
    """
    before = capture_torch_rng_state()
    augmented = model_forward(model, batch, device, grl_strength=grl_strength)
    after_augmented = capture_torch_rng_state()
    restore_torch_rng_state(before)
    clean = clean_view_forward(model, batch, device, grl_strength=grl_strength)
    restore_torch_rng_state(after_augmented)
    return augmented, clean


def artifact_consistency_loss(augmented, clean, applied_mask):
    """Keep predictions and fused representations stable under artifact removal."""
    mask = applied_mask.view(-1).bool()
    if not mask.any():
        return augmented["logits"].sum() * 0.0
    probability_loss = F.mse_loss(
        torch.sigmoid(augmented["logits"][mask]),
        torch.sigmoid(clean["logits"][mask]),
    )
    representation_loss = 1.0 - F.cosine_similarity(
        augmented["embedding"][mask], clean["embedding"][mask], dim=1
    ).mean()
    return probability_loss + 0.1 * representation_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        output = model_forward(model, batch, device)
        probabilities.extend(torch.sigmoid(output["logits"]).cpu().numpy().tolist())
        labels.extend(batch["label"].cpu().numpy().tolist())
    return binary_metrics(labels, probabilities)


def balanced_sampler(frame):
    strata = frame["label"].astype(str) + "|" + frame["source"].astype(str)
    counts = strata.value_counts()
    weights = strata.map(lambda value: 1.0 / counts[value]).to_numpy(dtype=np.float64).copy()
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--model-name", default="google/mobilebert-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5, help="HTML encoder learning rate")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-html-length", type=int, default=256)
    parser.add_argument("--max-dom-length", type=int, default=256)
    parser.add_argument("--dom-hidden-size", type=int, default=128)
    parser.add_argument("--dom-layers", type=int, default=2)
    parser.add_argument("--dom-heads", type=int, default=4)
    parser.add_argument("--fusion-hidden-size", type=int, default=256)
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lambda-supcon", type=float, default=0.2)
    parser.add_argument("--lambda-adversarial", type=float, default=0.1)
    parser.add_argument("--lambda-alignment", type=float, default=0.05)
    parser.add_argument(
        "--lambda-artifact-consistency", type=float, default=0.0,
        help=(
            "Consistency weight between original and artifact-transformed views. "
            "Values around 0.1 are intended for the robustness ablation."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--cross-generator-weight", type=float, default=2.0)
    parser.add_argument(
        "--aux-warmup-epochs", type=float, default=1.0,
        help="Train with BCE only, then linearly ramp auxiliary losses",
    )
    parser.add_argument("--disable-html", action="store_true")
    parser.add_argument("--disable-dom", action="store_true")
    parser.add_argument("--disable-static", action="store_true")
    parser.add_argument("--freeze-html", action="store_true")
    parser.add_argument(
        "--normalize-branches", action=argparse.BooleanOptionalAction, default=True,
        help="Layer-normalize each enabled branch before fusion",
    )
    parser.add_argument("--fusion-type", choices=["concat", "gated"], default="gated")
    parser.add_argument(
        "--modality-dropout", type=float, default=0.1,
        help="Randomly suppress modalities during gated-fusion training",
    )
    parser.add_argument(
        "--artifact-augmentation-probability", type=float, default=0.3,
        help="Probability of applying one artifact transform to a training page only",
    )
    parser.add_argument(
        "--artifact-augmentation-modes",
        default="comments_removed,identifiers_normalized,visible_text_removed",
        help="Comma-separated train-only artifact transforms",
    )
    parser.add_argument(
        "--balanced-sampling", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--selection-metric", choices=["f1", "auprc", "mcc"], default="auprc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True,
        help="Use deterministic PyTorch algorithms where supported",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output", default="outputs/best_model.pt")
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping")
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use a new --output or pass --overwrite."
        )
    free_gb = shutil.disk_usage(output_path.parent).free / (1024 ** 3)
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.2f} GB free; refusing to initialize data/model below "
            f"--min-free-gb {args.min_free_gb:.2f}."
        )
    print(f"Free disk before model initialization: {free_gb:.2f} GB")

    train_frame = read_frame(args.train)
    val_frame = read_frame(args.val)
    training_data_sha256 = {
        "train": sha256_file(args.train),
        "val": sha256_file(args.val),
    }
    training_source_snapshot = source_snapshot(ROOT)
    generator_to_id = build_generator_mapping(train_frame)
    use_html = not args.disable_html
    use_dom = not args.disable_dom
    use_static = not args.disable_static
    artifact_augmentation_modes = tuple(
        value.strip() for value in args.artifact_augmentation_modes.split(",")
        if value.strip()
    )
    if args.lambda_artifact_consistency < 0:
        raise ValueError("--lambda-artifact-consistency must be non-negative")
    if args.lambda_artifact_consistency > 0 and args.artifact_augmentation_probability <= 0:
        raise ValueError(
            "Artifact consistency requires --artifact-augmentation-probability > 0"
        )

    train_dataset = WebPageDataset(
        train_frame,
        args.model_name,
        args.max_html_length,
        args.max_dom_length,
        generator_to_id=generator_to_id,
        use_html=use_html,
        use_dom=use_dom,
        use_static=use_static,
        artifact_augmentation_probability=args.artifact_augmentation_probability,
        artifact_augmentation_modes=artifact_augmentation_modes,
        augmentation_seed=args.seed,
        return_clean_view=args.lambda_artifact_consistency > 0,
    )
    val_dataset = WebPageDataset(
        val_frame,
        args.model_name,
        args.max_html_length,
        args.max_dom_length,
        generator_to_id=generator_to_id,
        static_mean=train_dataset.static_mean,
        static_std=train_dataset.static_std,
        use_html=use_html,
        use_dom=use_dom,
        use_static=use_static,
    )
    sampler = balanced_sampler(train_frame) if args.balanced_sampling else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_config = {
        "model_name": args.model_name,
        "max_dom_length": args.max_dom_length,
        "dom_hidden_size": args.dom_hidden_size,
        "dom_layers": args.dom_layers,
        "dom_heads": args.dom_heads,
        "static_feature_size": len(STATIC_FEATURE_NAMES),
        "fusion_hidden_size": args.fusion_hidden_size,
        "projection_size": args.projection_size,
        "num_generators": len(generator_to_id),
        "dropout": args.dropout,
        "use_html": use_html,
        "use_dom": use_dom,
        "use_static": use_static,
        "freeze_html": args.freeze_html,
        "normalize_branches": args.normalize_branches,
        "fusion_type": args.fusion_type,
        "modality_dropout": args.modality_dropout,
    }
    model = PhishGenModel(**model_config).to(device)

    html_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (html_parameters if name.startswith("html_encoder.") else head_parameters).append(parameter)
    parameter_groups = []
    if html_parameters:
        parameter_groups.append({"params": html_parameters, "lr": args.lr})
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": args.head_lr})
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=args.weight_decay
    )

    best_score = -float("inf")
    history = []
    epochs_without_improvement = 0
    total_steps = max(1, args.epochs * len(train_loader))
    aux_warmup_steps = max(0, int(args.aux_warmup_epochs * len(train_loader)))
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch - 1)
        model.train()
        running = {
            "loss": 0.0, "bce": 0.0, "supcon": 0.0,
            "adversarial": 0.0, "alignment": 0.0,
            "artifact_consistency": 0.0,
        }
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            progress = global_step / max(1, total_steps - 1)
            if global_step < aux_warmup_steps:
                aux_scale = 0.0
            else:
                aux_scale = min(
                    1.0,
                    (global_step - aux_warmup_steps + 1)
                    / max(1, total_steps - aux_warmup_steps),
                )
            grl_strength = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
            clean_output = None
            if args.lambda_artifact_consistency > 0:
                output, clean_output = paired_stochastic_forwards(
                    model, batch, device, grl_strength=grl_strength
                )
            else:
                output = model_forward(
                    model, batch, device, grl_strength=grl_strength
                )

            labels = batch["label"].to(device)
            generator_ids = batch["generator_id"].to(device)
            domain_ids = batch["domain_id"].to(device)
            template_ids = batch["template_id"].to(device)

            bce = F.binary_cross_entropy_with_logits(output["logits"], labels)
            supcon = generator_aware_supcon_loss(
                output["projection"], labels, generator_ids,
                template_ids=template_ids,
                temperature=args.temperature,
                cross_generator_weight=args.cross_generator_weight,
            )
            adversarial = output["embedding"].sum() * 0.0
            phishing_known = labels.eq(1) & generator_ids.ge(0)
            if output["generator_logits"] is not None and phishing_known.any():
                adversarial = F.cross_entropy(
                    output["generator_logits"][phishing_known], generator_ids[phishing_known]
                )
            alignment = phishing_domain_alignment_loss(
                output["embedding"], labels, domain_ids
            )
            artifact_consistency = output["embedding"].sum() * 0.0
            if args.lambda_artifact_consistency > 0:
                artifact_consistency = artifact_consistency_loss(
                    output,
                    clean_output,
                    batch["artifact_applied"].to(device),
                )
            loss = (
                bce
                + aux_scale * args.lambda_supcon * supcon
                + aux_scale * args.lambda_adversarial * adversarial
                + aux_scale * args.lambda_alignment * alignment
                + aux_scale * args.lambda_artifact_consistency * artifact_consistency
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

            values = {
                "loss": loss.item(), "bce": bce.item(), "supcon": supcon.item(),
                "adversarial": adversarial.item(), "alignment": alignment.item(),
                "artifact_consistency": artifact_consistency.item(),
                "aux_scale": aux_scale,
            }
            for key, value in values.items():
                if key == "aux_scale":
                    continue
                running[key] += value
            if step % args.log_every == 0 or step == len(train_loader):
                averages = " ".join(
                    f"{key}={value / step:.4f}" for key, value in running.items()
                )
                print(f"epoch={epoch} step={step}/{len(train_loader)} {averages}")

        metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, **metrics})
        print("VAL", json.dumps(metrics, indent=2))
        score = metrics[args.selection_metric]
        if np.isfinite(score) and score > best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "schema_version": 3,
                    "state_dict": model.state_dict(),
                    "model_config": model_config,
                    "model_name": args.model_name,
                    "max_html_length": args.max_html_length,
                    "max_dom_length": args.max_dom_length,
                    "generator_to_id": generator_to_id,
                    "static_feature_names": STATIC_FEATURE_NAMES,
                    "static_mean": train_dataset.static_mean.tolist(),
                    "static_std": train_dataset.static_std.tolist(),
                    "training_args": vars(args),
                    "training_data_sha256": training_data_sha256,
                    "source_snapshot": training_source_snapshot,
                    "best_validation_metrics": metrics,
                },
                output_path,
            )
            print(f"Saved best checkpoint: {output_path}")
        else:
            epochs_without_improvement += 1
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs (patience={args.patience})")
            break

    history_path = output_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if not output_path.exists():
        raise RuntimeError("No checkpoint was saved; check validation labels and metric values")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
