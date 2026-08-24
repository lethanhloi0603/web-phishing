import torch
import torch.nn.functional as F


def generator_aware_supcon_loss(
    embeddings,
    labels,
    generator_ids,
    template_ids=None,
    temperature=0.1,
    cross_generator_weight=2.0,
    exclude_same_template=True,
):
    """Supervised contrastive loss emphasizing cross-generator phishing pairs."""
    if embeddings.size(0) < 3:
        return embeddings.sum() * 0.0

    values = F.normalize(embeddings, dim=1)
    logits = (values @ values.T) / temperature
    size = values.size(0)
    eye = torch.eye(size, dtype=torch.bool, device=values.device)
    logits = logits.masked_fill(eye, -1e9)

    labels = labels.view(-1).long()
    generators = generator_ids.view(-1).long()
    same_class = labels[:, None].eq(labels[None, :]) & ~eye
    different_generator = ~generators[:, None].eq(generators[None, :])
    phishing_pair = labels[:, None].eq(1) & labels[None, :].eq(1)

    positive_mask = same_class
    if template_ids is not None and exclude_same_template:
        templates = template_ids.view(-1).long()
        same_template = templates[:, None].eq(templates[None, :])
        positive_mask = positive_mask & ~(phishing_pair & same_template)

    weights = torch.ones_like(logits)
    cross_generator_phishing = phishing_pair & different_generator
    weights = torch.where(
        cross_generator_phishing,
        torch.full_like(weights, cross_generator_weight),
        weights,
    )
    positive_weights = weights * positive_mask.float()
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    denominator = positive_weights.sum(dim=1)
    valid = denominator > 0
    if not valid.any():
        return embeddings.sum() * 0.0

    mean_positive = (
        (positive_weights * log_probability).sum(dim=1)
        / denominator.clamp_min(1e-8)
    )
    return -mean_positive[valid].mean()


def coral_loss(source_embeddings, target_embeddings):
    """CORAL covariance alignment between two representation domains."""
    if source_embeddings.size(0) < 2 or target_embeddings.size(0) < 2:
        return (source_embeddings.sum() + target_embeddings.sum()) * 0.0

    def covariance(values):
        centered = values - values.mean(dim=0, keepdim=True)
        return centered.T @ centered / max(1, values.size(0) - 1)

    source_cov = covariance(source_embeddings)
    target_cov = covariance(target_embeddings)
    dimension = source_embeddings.size(1)
    return (source_cov - target_cov).pow(2).sum() / (4.0 * dimension * dimension)


def phishing_domain_alignment_loss(embeddings, labels, domain_ids):
    """Align conventional and LLM phishing without mixing benign/class effects."""
    phishing = labels.view(-1).long().eq(1)
    conventional = embeddings[phishing & domain_ids.view(-1).long().eq(0)]
    llm = embeddings[phishing & domain_ids.view(-1).long().eq(1)]
    return coral_loss(conventional, llm)
