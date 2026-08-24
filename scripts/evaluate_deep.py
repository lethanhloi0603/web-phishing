import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import WebPageDataset, precompute_page_features, read_frame
from src.calibration import calibrated_evaluation
from src.metrics import binary_metrics
from src.model import PhishGenModel
from src.site_evaluation import site_level_evaluation
from src.preprocessing import (
    STATIC_FEATURE_NAMES,
    normalize_html_identifiers,
    remove_visible_text,
    strip_html_comments,
)
from src.research import evaluation_data_evidence, sha256_file


def evaluate_frame(
    model, frame, checkpoint, device, batch_size, precomputed_features=None
):
    config = checkpoint["model_config"]
    raw_static, dom_ids = precomputed_features or (None, None)
    dataset = WebPageDataset(
        frame,
        checkpoint["model_name"],
        checkpoint["max_html_length"],
        checkpoint["max_dom_length"],
        generator_to_id=checkpoint["generator_to_id"],
        static_mean=checkpoint["static_mean"],
        static_std=checkpoint["static_std"],
        use_html=config["use_html"],
        use_dom=config["use_dom"],
        use_static=config["use_static"],
        precomputed_static_features=raw_static,
        precomputed_dom_ids=dom_ids,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    labels, probabilities, branch_weight_rows = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["dom_ids"].to(device),
                batch["dom_mask"].to(device),
                batch["static_features"].to(device),
            )
            labels.extend(batch["label"].numpy().tolist())
            probabilities.extend(torch.sigmoid(output["logits"]).cpu().numpy().tolist())
            if output.get("branch_weights") is not None:
                branch_weight_rows.extend(
                    output["branch_weights"].cpu().numpy().tolist()
                )
    branch_names = [
        name for name, enabled in (
            ("html", config["use_html"]),
            ("dom", config["use_dom"]),
            ("static", config["use_static"]),
        ) if enabled
    ]
    weights = {
        name: [float(row[index]) for row in branch_weight_rows]
        for index, name in enumerate(branch_names)
    } if branch_weight_rows else {}
    return binary_metrics(labels, probabilities), probabilities, weights


def feature_cache_path(cache_dir, split_path, checkpoint, cache_variant="original"):
    config = checkpoint["model_config"]
    payload = (
        f"v2:{cache_variant}:{sha256_file(split_path)}:{checkpoint['max_dom_length']}:"
        f"{int(config['use_dom'])}:{int(config['use_static'])}:"
        f"{','.join(STATIC_FEATURE_NAMES)}"
    )
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{key}.npz"


def load_or_build_feature_cache(
    frame, split_path, checkpoint, cache_dir, cache_variant="original"
):
    config = checkpoint["model_config"]
    if not config["use_dom"] and not config["use_static"]:
        return None
    path = feature_cache_path(
        cache_dir, split_path, checkpoint, cache_variant=cache_variant
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            raw_static = cached["static"] if config["use_static"] else None
            dom_ids = cached["dom"] if config["use_dom"] else None
        expected_static = (len(frame), len(STATIC_FEATURE_NAMES))
        expected_dom = (len(frame), checkpoint["max_dom_length"])
        if (
            (raw_static is None or raw_static.shape == expected_static)
            and (dom_ids is None or dom_ids.shape == expected_dom)
        ):
            return raw_static, dom_ids
        path.unlink()
    raw_static, dom_ids = precompute_page_features(
        frame,
        checkpoint["max_dom_length"],
        use_dom=config["use_dom"],
        use_static=config["use_static"],
    )
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            static=(
                raw_static
                if raw_static is not None
                else np.empty((len(frame), 0), dtype=np.float32)
            ),
            dom=(
                dom_ids
                if dom_ids is not None
                else np.empty((len(frame), 0), dtype=np.int64)
            ),
        )
    temporary.replace(path)
    return raw_static, dom_ids


def grouped_report(frame, threshold=0.5):
    reports = {}
    for column in ("source", "generator_family", "generator", "brand", "quality_level"):
        if column not in frame:
            continue
        reports[column] = {}
        for value, group in frame.groupby(column, dropna=False):
            y_true = group["label"].astype(int).to_numpy()
            y_prob = group["probability"].astype(float).to_numpy()
            if len(set(y_true.tolist())) > 1:
                reports[column][str(value)] = binary_metrics(
                    y_true, y_prob, threshold=threshold
                )
            else:
                predictions = y_prob >= threshold
                reports[column][str(value)] = {
                    "n": int(len(group)),
                    "label": int(y_true[0]),
                    "class_recall": float((predictions == y_true).mean()),
                    "mean_probability": float(y_prob.mean()),
                }
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val", default="")
    parser.add_argument("--test", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--artifact-tests", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--predictions", default="")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--fpr-confidence", type=float, default=0.95)
    parser.add_argument(
        "--feature-cache-dir", default="outputs/cache/page_features",
        help="Reuse deterministic raw DOM/static features across checkpoint seeds",
    )
    parser.add_argument(
        "--evaluation-role", choices=["primary", "external", "ablation"],
        default="primary",
        help="Declare how this result may be used by the evidence gate",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") not in {2, 3}:
        raise ValueError("Unsupported checkpoint schema; retrain with the updated train_deep.py")
    model = PhishGenModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    frame = read_frame(args.test)
    test_features = load_or_build_feature_cache(
        frame, args.test, checkpoint, args.feature_cache_dir
    )
    overall, probabilities, branch_weights = evaluate_frame(
        model, frame, checkpoint, device, args.batch_size, test_features
    )
    predictions = frame.copy()
    predictions["probability"] = probabilities
    report = {
        "overall": overall,
        "fixed_threshold": 0.5,
        "groups": grouped_report(predictions),
        "evaluation_role": args.evaluation_role,
        "evaluation_data": evaluation_data_evidence(frame, args.test),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_schema_version": checkpoint.get("schema_version"),
        "training_data_sha256": checkpoint.get("training_data_sha256", {}),
        "source_snapshot_sha256": checkpoint.get("source_snapshot", {}).get("sha256"),
        "training_args": checkpoint.get("training_args", {}),
    }
    if branch_weights:
        for name, values in branch_weights.items():
            predictions[f"gate_weight_{name}"] = values
        report["mean_gate_weights"] = {
            name: float(sum(values) / len(values))
            for name, values in branch_weights.items()
        }
    calibrated_thresholds = {}
    if args.val:
        val_frame = read_frame(args.val)
        val_features = load_or_build_feature_cache(
            val_frame, args.val, checkpoint, args.feature_cache_dir
        )
        report["validation_data"] = evaluation_data_evidence(val_frame, args.val)
        _, val_probabilities, _ = evaluate_frame(
            model, val_frame, checkpoint, device, args.batch_size, val_features
        )
        report["calibration"] = calibrated_evaluation(
            val_frame["label"].astype(int).to_numpy(),
            val_probabilities,
            frame["label"].astype(int).to_numpy(),
            probabilities,
            target_fpr=args.target_fpr,
            fpr_confidence=args.fpr_confidence,
        )
        report["target_fpr"] = args.target_fpr
        val_site_input = val_frame.copy()
        val_site_input["probability"] = val_probabilities
        site_report, _, test_sites = site_level_evaluation(
            val_site_input,
            predictions,
            target_fpr=args.target_fpr,
            aggregation="max",
            fpr_confidence=args.fpr_confidence,
        )
        for mode, values in site_report.get("calibration", {}).items():
            site_report[f"groups_{mode}"] = grouped_report(
                test_sites, threshold=float(values["threshold"])
            )
        report["site_level"] = site_report
        calibrated_thresholds = {
            mode: values["threshold"] for mode, values in report["calibration"].items()
        }
        target_threshold = calibrated_thresholds["target_fpr"]
        report["groups_target_fpr"] = grouped_report(
            predictions, threshold=target_threshold
        )
        predictions["predicted_target_fpr"] = (
            predictions["probability"] >= target_threshold
        ).astype(int)
        confident_threshold = calibrated_thresholds["target_fpr_confident"]
        report["groups_target_fpr_confident"] = grouped_report(
            predictions, threshold=confident_threshold
        )
        predictions["predicted_target_fpr_confident"] = (
            predictions["probability"] >= confident_threshold
        ).astype(int)
    predictions["predicted_0_5"] = (
        predictions["probability"] >= 0.5
    ).astype(int)

    if args.artifact_tests:
        transforms = {
            "comments_removed": strip_html_comments,
            "identifiers_normalized": normalize_html_identifiers,
            "visible_text_removed": remove_visible_text,
        }
        report["artifact_tests"] = {}
        for name, transform in transforms.items():
            altered = frame.copy()
            altered["html"] = altered["html"].astype(str).map(transform)
            artifact_features = load_or_build_feature_cache(
                altered,
                args.test,
                checkpoint,
                args.feature_cache_dir,
                cache_variant=f"artifact:{name}",
            )
            metrics, altered_probabilities, _ = evaluate_frame(
                model, altered, checkpoint, device, args.batch_size,
                artifact_features,
            )
            artifact_report = {"fixed_0_5": metrics}
            altered_predictions = altered.copy()
            altered_predictions["probability"] = altered_probabilities
            artifact_report["groups_fixed_0_5"] = grouped_report(
                altered_predictions, threshold=0.5
            )
            for mode, threshold in calibrated_thresholds.items():
                artifact_report[mode] = binary_metrics(
                    altered["label"].astype(int).to_numpy(),
                    altered_probabilities,
                    threshold=threshold,
                )
                artifact_report[f"groups_{mode}"] = grouped_report(
                    altered_predictions, threshold=threshold
                )
            report["artifact_tests"][name] = artifact_report

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.predictions:
        path = Path(args.predictions)
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction_export = predictions.drop(columns=["html"], errors="ignore")
        prediction_export.to_csv(path, index=False)


if __name__ == "__main__":
    main()
