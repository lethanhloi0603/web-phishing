import numpy as np
from scipy.stats import beta
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix,
)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(1, len(y_true))
    error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        include = (y_prob >= lower) & (y_prob < upper)
        if upper == 1.0:
            include = (y_prob >= lower) & (y_prob <= upper)
        if include.any():
            accuracy = y_true[include].mean()
            confidence = y_prob[include].mean()
            error += include.sum() / total * abs(accuracy - confidence)
    return float(error)


def clopper_pearson_upper(successes, trials, confidence=0.95):
    """Exact one-sided upper confidence bound for a binomial proportion."""
    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        return float("nan")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if successes == trials:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))

def binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    auprc = float("nan")
    auroc = float("nan")
    if len(np.unique(y_true)) > 1:
        auprc = average_precision_score(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auprc": float(auprc),
        "auroc": float(auroc),
        "fpr": float(fpr),
        "fpr_upper_95": clopper_pearson_upper(fp, fp + tn, confidence=0.95),
        "fnr": float(fnr),
        "fnr_upper_95": clopper_pearson_upper(fn, fn + tp, confidence=0.95),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
