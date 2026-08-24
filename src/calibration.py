import numpy as np

from .metrics import binary_metrics, clopper_pearson_upper


def select_threshold(
    y_true,
    probability,
    mode="f1",
    target_fpr=0.01,
    fpr_confidence=0.95,
):
    """Select a threshold using validation predictions only."""
    if mode not in {"f1", "target_fpr", "target_fpr_confident"}:
        raise ValueError(
            "mode must be 'f1', 'target_fpr', or 'target_fpr_confident'"
        )
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        raise ValueError("Threshold calibration requires both classes in validation data")

    candidates = np.unique(np.concatenate([
        np.asarray([0.0, 0.5, 1.0, 1.000001]), probability,
    ]))
    scored = [
        (float(threshold), binary_metrics(y_true, probability, threshold=float(threshold)))
        for threshold in candidates
    ]
    if mode == "f1":
        return max(
            scored,
            key=lambda item: (
                item[1]["f1"], -item[1]["fpr"], item[1]["precision"], item[0]
            ),
        )

    if mode == "target_fpr_confident":
        for _threshold, metrics in scored:
            negative_count = metrics["tn"] + metrics["fp"]
            metrics["fpr_upper_bound"] = clopper_pearson_upper(
                metrics["fp"], negative_count, confidence=fpr_confidence
            )
            metrics["fpr_confidence"] = float(fpr_confidence)
        eligible = [
            item for item in scored
            if item[1]["fpr_upper_bound"] <= target_fpr
        ]
        certified = bool(eligible)
        if not eligible:
            eligible = scored
        threshold, metrics = max(
            eligible,
            key=lambda item: (
                item[1]["recall"], item[1]["precision"],
                -item[1]["fpr_upper_bound"], item[0],
            ),
        )
        metrics["target_fpr_certified"] = certified
        return threshold, metrics

    eligible = [item for item in scored if item[1]["fpr"] <= target_fpr]
    if not eligible:
        eligible = scored
    return max(
        eligible,
        key=lambda item: (
            item[1]["recall"], item[1]["precision"], -item[1]["fpr"], item[0]
        ),
    )


def calibrated_evaluation(
    y_val,
    val_probability,
    y_test,
    test_probability,
    target_fpr=0.01,
    fpr_confidence=0.95,
):
    report = {}
    for mode in ("f1", "target_fpr", "target_fpr_confident"):
        threshold, validation_metrics = select_threshold(
            y_val,
            val_probability,
            mode=mode,
            target_fpr=target_fpr,
            fpr_confidence=fpr_confidence,
        )
        report[mode] = {
            "threshold": threshold,
            "validation": validation_metrics,
            "test": binary_metrics(y_test, test_probability, threshold=threshold),
        }
    return report
