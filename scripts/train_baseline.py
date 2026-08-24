import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import binary_metrics
from src.calibration import calibrated_evaluation
from src.preprocessing import extract_static_features, html_to_dom_tags, html_to_semantic_text
from src.site_evaluation import site_level_evaluation
from src.research import evaluation_data_evidence, sha256_file


def text_features(raw_html):
    semantic = html_to_semantic_text(str(raw_html), max_chars=12000)
    dom = " ".join("TAG_" + tag for tag in html_to_dom_tags(str(raw_html), max_nodes=300))
    return semantic + " " + dom


def static_matrix(frame):
    urls = frame["url"] if "url" in frame else pd.Series([""] * len(frame))
    return np.vstack([
        extract_static_features(str(html), str(url))
        for html, url in zip(frame["html"], urls)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", default="")
    parser.add_argument("--test", required=True)
    parser.add_argument("--model", choices=["tfidf", "char_tfidf", "static"], default="tfidf")
    parser.add_argument("--max-features", type=int, default=40000)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--predictions", default="")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument(
        "--evaluation-role",
        choices=["baseline", "temporal", "external", "ablation"],
        default="baseline",
    )
    args = parser.parse_args()

    train = pd.read_csv(args.train).dropna(subset=["html", "label"])
    val = pd.read_csv(args.val).dropna(subset=["html", "label"]) if args.val else None
    test = pd.read_csv(args.test).dropna(subset=["html", "label"])
    y_train = train["label"].astype(int).to_numpy()
    y_test = test["label"].astype(int).to_numpy()

    if args.model in {"tfidf", "char_tfidf"}:
        vectorizer = {
            "tfidf": TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2,
                max_features=args.max_features, sublinear_tf=True,
            ),
            "char_tfidf": TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), min_df=2,
                max_features=args.max_features, sublinear_tf=True,
            ),
        }[args.model]
        classifier = Pipeline([
            ("tfidf", vectorizer),
            ("classifier", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42,
            )),
        ])
        x_train = [text_features(value) for value in train["html"]]
        x_val = [text_features(value) for value in val["html"]] if val is not None else None
        x_test = [text_features(value) for value in test["html"]]
    else:
        classifier = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, l2_regularization=1.0, random_state=42
        )
        x_train = static_matrix(train)
        x_val = static_matrix(val) if val is not None else None
        x_test = static_matrix(test)

    classifier.fit(x_train, y_train)
    probability = classifier.predict_proba(x_test)[:, 1]
    report = {
        "model": args.model,
        "overall": binary_metrics(y_test, probability),
        "fixed_threshold": 0.5,
        "evaluation_role": args.evaluation_role,
        "evaluation_data": evaluation_data_evidence(test, args.test),
        "input_sha256": {
            "train": sha256_file(args.train),
            "val": sha256_file(args.val) if args.val else None,
            "test": sha256_file(args.test),
        },
        "training_args": vars(args),
    }
    if val is not None:
        y_val = val["label"].astype(int).to_numpy()
        report["validation_data"] = evaluation_data_evidence(val, args.val)
        val_probability = classifier.predict_proba(x_val)[:, 1]
        report["calibration"] = calibrated_evaluation(
            y_val, val_probability, y_test, probability, target_fpr=args.target_fpr
        )
        report["target_fpr"] = args.target_fpr
        target_threshold = report["calibration"]["target_fpr"]["threshold"]
        val_site_input = val.copy()
        val_site_input["probability"] = val_probability
        test_site_input = test.copy()
        test_site_input["probability"] = probability
        site_report, _, _ = site_level_evaluation(
            val_site_input,
            test_site_input,
            target_fpr=args.target_fpr,
            aggregation="max",
        )
        report["site_level"] = site_report
    else:
        target_threshold = 0.5

    if "generator" in test:
        grouped = test.copy()
        grouped["probability"] = probability
        report["phishing_recall_by_generator"] = {
            str(generator): {
                "n": int(len(group)),
                "recall": float(group["probability"].ge(0.5).mean()),
            }
            for generator, group in grouped[grouped["label"].astype(int).eq(1)].groupby("generator")
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.predictions:
        path = Path(args.predictions)
        path.parent.mkdir(parents=True, exist_ok=True)
        exported = test.drop(columns=["html"], errors="ignore").copy()
        exported["probability"] = probability
        exported["predicted_0_5"] = (probability >= 0.5).astype(int)
        exported["predicted_target_fpr"] = (probability >= target_threshold).astype(int)
        exported.to_csv(path, index=False)


if __name__ == "__main__":
    main()
