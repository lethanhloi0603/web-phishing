# Copyright (c) 2026 Le Thanh Loi. All rights reserved.
# Author: Le Thanh Loi

"""Train the branch-aligned CompPhish model zoo and OOF stackers.

Train contains every model/feature decision. StratifiedGroupKFold,
grouped by registered domain, produces out-of-fold (OOF) predictions. Test is
scored once by models refitted on all Train rows and is never used to select a
feature set or learner.

Feature selection is inside each sklearn Pipeline, so it is refitted in every
OOF fold. No resampling or class weighting is applied.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from compphish_common import (
    DIRECTLY_VISIBLE_URL_COLUMNS,
    FEATURE_SETS,
    FEATURE_SETS_PUBLISHED,
    FoldSafePhishingMeanFeatures,
    LABEL_DERIVED_COLUMNS,
    PROJECT_AUTHOR,
    PROJECT_REPOSITORY,
    RANDOM_STATE,
    evaluate_binary,
    project_path,
    read_processed,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    view: str
    family: str
    estimator: object
    columns: tuple[str, ...]


STRUCTURED_VIEWS = ("brand", "url", "brand_url", "html", "full")
PRIMARY_VIEWS = ("brand_url", "html", "full")

# Both recipes are evaluated independently inside every feature branch.  A
# Brand-URL stack therefore consumes only Brand-URL base predictions, an HTML
# stack only HTML predictions, and a Full stack only Full predictions.
STACKING_RECIPES = {
    "stacking_boosting": (
        "hist_gradient_boosting", "xgboost", "lightgbm", "catboost",
    ),
    "stacking_heterogeneous": (
        "logistic_regression", "rbf_svm", "random_forest", "catboost", "mlp",
    ),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/compphish/compphish_integrity_split.csv.gz",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/compphish/branch_stacking_full"
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-tfidf-features", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--train-limit", type=int, default=3000)
    parser.add_argument(
        "--model-set", choices=("framework", "extended"), default="framework",
        help=(
            "framework runs the 12 prespecified learners; extended adds five "
            "secondary historical baselines."
        ),
    )
    parser.add_argument(
        "--extended-all-views", action="store_true",
        help=(
            "Run secondary extended baselines on every numeric view. By default "
            "they run on Full only."
        ),
    )
    parser.add_argument(
        "--views", nargs="+", choices=(*STRUCTURED_VIEWS, "url_text"),
        default=list(PRIMARY_VIEWS),
    )
    parser.add_argument(
        "--feature-selector", choices=("none", "mutual_info", "l1", "tree"),
        default="none",
        help="Selector is fitted inside every OOF fold; numeric views only.",
    )
    parser.add_argument(
        "--selector-k", type=int, default=0,
        help="Features kept by mutual_info; 0 means approximately 60 percent.",
    )
    parser.add_argument(
        "--optional-boosters", choices=("auto", "none", "required"),
        default="required",
        help="Use installed XGBoost/LightGBM/CatBoost, skip, or require all three.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Fixed decision threshold used for every model (default: 0.5).",
    )
    parser.add_argument(
        "--reference-model", default="full__catboost",
        help=(
            "Pre-specified model used by downstream explanation/report scripts. "
            "It is not selected from Test."
        ),
    )
    parser.add_argument(
        "--drop-features", nargs="*", default=[],
        help="Published feature names to remove, for example count_www.",
    )
    parser.add_argument(
        "--drop-visible-url-cues", action="store_true",
        help="Ablation: remove documented directly visible URL cues.",
    )
    parser.add_argument(
        "--include-fold-safe-mean-features", action="store_true",
        default=True,
        help=(
            "Use all 70 published features, but recompute url_more_than_mean "
            "and domain_more_than_mean inside every OOF fit from that fold's "
            "training phishing rows. Published workbook values are overwritten."
        ),
    )
    return parser.parse_args()


def selector_for(name: str, feature_count: int, args: argparse.Namespace) -> object:
    if name == "none":
        return "passthrough"
    if name == "mutual_info":
        k = args.selector_k or max(1, int(np.ceil(feature_count * 0.60)))
        scorer = partial(mutual_info_classif, random_state=args.seed)
        return SelectKBest(score_func=scorer, k=min(k, feature_count))
    if name == "l1":
        return SelectFromModel(
            LogisticRegression(
                penalty="l1", solver="liblinear", C=0.1, max_iter=3000,
                random_state=args.seed,
            ),
            threshold="median",
        )
    if name == "tree":
        return SelectFromModel(
            ExtraTreesClassifier(
                n_estimators=150 if args.quick else 400, min_samples_leaf=2,
                n_jobs=-1, random_state=args.seed,
            ),
            threshold="median",
        )
    raise ValueError(f"Unsupported feature selector: {name}")


def numeric_pipeline(estimator: object, columns: tuple[str, ...], args: argparse.Namespace) -> Pipeline:
    # Scaling is necessary for margin/distance models and the L1 selector; it
    # does not change tree split ordering, so one safe ordering serves all.
    steps = []
    if args.include_fold_safe_mean_features and any(
        column in columns for column in LABEL_DERIVED_COLUMNS
    ):
        steps.append(("fold_safe_mean_features", FoldSafePhishingMeanFeatures()))
    steps.extend([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", selector_for(args.feature_selector, len(columns), args)),
        ("model", estimator),
    ])
    return Pipeline(steps)


def optional_boosters(args: argparse.Namespace) -> list[tuple[str, str, object]]:
    if args.optional_boosters == "none":
        return []
    result: list[tuple[str, str, object]] = []
    missing: list[str] = []
    trees = 120 if args.quick else 500
    try:
        from xgboost import XGBClassifier
        result.append(("xgboost", "xgboost", XGBClassifier(
            n_estimators=trees, max_depth=6, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1,
            random_state=args.seed,
        )))
    except ImportError:
        missing.append("xgboost")
    try:
        from lightgbm import LGBMClassifier
        result.append(("lightgbm", "lightgbm", LGBMClassifier(
            n_estimators=trees, learning_rate=0.05, num_leaves=31, subsample=0.9,
            colsample_bytree=0.9, verbosity=-1, n_jobs=-1, random_state=args.seed,
        )))
    except ImportError:
        missing.append("lightgbm")
    try:
        from catboost import CatBoostClassifier
        result.append(("catboost", "catboost", CatBoostClassifier(
            iterations=trees, depth=7, learning_rate=0.05, verbose=False,
            allow_writing_files=False, thread_count=-1, random_seed=args.seed,
        )))
    except ImportError:
        missing.append("catboost")
    if missing and args.optional_boosters == "required":
        raise SystemExit(
            "Missing optional packages: " + ", ".join(missing)
            + ". Install with: python -m pip install " + " ".join(missing)
        )
    if missing:
        print("Optional models skipped: " + ", ".join(missing))
    return result


def structured_estimators(args: argparse.Namespace) -> list[tuple[str, str, object]]:
    trees = 100 if args.quick else 400
    iterations = 100 if args.quick else 300
    result: list[tuple[str, str, object]] = [
        ("logistic_regression", "linear", LogisticRegression(
            max_iter=4000, random_state=args.seed,
        )),
        ("gaussian_naive_bayes", "probabilistic", GaussianNB()),
        ("knn", "distance", KNeighborsClassifier(
            n_neighbors=7, weights="distance", n_jobs=-1,
        )),
        ("decision_tree", "single_tree", DecisionTreeClassifier(
            min_samples_leaf=3, random_state=args.seed,
        )),
        ("rbf_svm", "kernel_margin", SVC(
            C=2.0, kernel="rbf", gamma="scale", probability=True,
            random_state=args.seed,
        )),
        ("mlp", "neural_network", MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-4, early_stopping=True,
            max_iter=120 if args.quick else 400, random_state=args.seed,
        )),
        ("random_forest", "bagging_tree", RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=args.seed,
        )),
        ("extra_trees", "randomized_tree", ExtraTreesClassifier(
            n_estimators=trees, min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=args.seed,
        )),
        ("hist_gradient_boosting", "gradient_boosting", HistGradientBoostingClassifier(
            max_iter=iterations, learning_rate=0.08, l2_regularization=0.1,
            random_state=args.seed,
        )),
    ]
    result.extend(optional_boosters(args))
    if args.model_set == "extended":
        result.extend([
            ("linear_svm", "margin", LinearSVC(
                C=1.0, max_iter=10000, random_state=args.seed,
            )),
            ("bagging", "bagging_tree", BaggingClassifier(
                estimator=DecisionTreeClassifier(min_samples_leaf=2),
                n_estimators=50 if args.quick else 200, n_jobs=-1,
                random_state=args.seed,
            )),
            ("adaboost", "boosting", AdaBoostClassifier(
                n_estimators=100 if args.quick else 300, random_state=args.seed,
            )),
            ("gradient_boosting", "gradient_boosting", GradientBoostingClassifier(
                n_estimators=100 if args.quick else 300, learning_rate=0.05,
                min_samples_leaf=2, random_state=args.seed,
            )),
        ])
    return result


def model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    feature_sets = (
        FEATURE_SETS_PUBLISHED
        if args.include_fold_safe_mean_features
        else FEATURE_SETS
    )
    dropped = set(args.drop_features)
    if args.drop_visible_url_cues:
        dropped.update(DIRECTLY_VISIBLE_URL_COLUMNS)
        # The two fold-safe indicators are thresholded forms of URL/domain
        # length.  Retaining them would leak the very visible length cues this
        # ablation is designed to remove, and their source columns are absent.
        if args.include_fold_safe_mean_features:
            dropped.update(LABEL_DERIVED_COLUMNS)
    known = set().union(*[set(value) for value in FEATURE_SETS_PUBLISHED.values()])
    unknown = dropped - known
    if unknown:
        raise ValueError(f"Unknown --drop-features values: {sorted(unknown)}")

    specs: list[ModelSpec] = []
    estimators = structured_estimators(args)
    framework_algorithms = {
        "logistic_regression", "gaussian_naive_bayes", "knn",
        "decision_tree", "rbf_svm", "mlp", "random_forest", "extra_trees",
        "hist_gradient_boosting", "xgboost", "lightgbm", "catboost",
    }
    suffix = "" if args.feature_selector == "none" else f"__fs_{args.feature_selector}"
    for view in args.views:
        if view == "url_text":
            continue
        columns = tuple(column for column in feature_sets[view] if column not in dropped)
        if args.include_fold_safe_mean_features:
            for derived, source in (
                ("url_more_than_mean", "url_length"),
                ("domain_more_than_mean", "domain_length"),
            ):
                if derived in columns and source not in columns:
                    raise ValueError(
                        f"Cannot compute {derived} after dropping its source {source}."
                    )
        if not columns:
            raise ValueError(f"Ablation removed every feature from view {view}")
        view_estimators = estimators
        if args.model_set == "extended" and view != "full" and not args.extended_all_views:
            view_estimators = [
                item for item in estimators if item[0] in framework_algorithms
            ]
        for algorithm, family, estimator in view_estimators:
            specs.append(ModelSpec(
                f"{view}__{algorithm}{suffix}", view, family,
                numeric_pipeline(estimator, columns, args), columns,
            ))

    if "url_text" in args.views:
        vectorizer = ColumnTransformer([("url", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2,
            max_features=args.max_tfidf_features, sublinear_tf=True,
        ), "url")], remainder="drop")
        specs.extend([
            ModelSpec(
                "url_text__tfidf_logreg", "url_text", "text_linear",
                Pipeline([
                    ("vectorizer", clone(vectorizer)),
                    ("model", LogisticRegression(
                        max_iter=4000, solver="liblinear", random_state=args.seed,
                    )),
                ]),
                ("url",),
            ),
            ModelSpec(
                "url_text__tfidf_linear_svm", "url_text", "text_margin",
                Pipeline([
                    ("vectorizer", clone(vectorizer)),
                    ("model", LinearSVC(
                        C=1.0, max_iter=10000, random_state=args.seed,
                    )),
                ]),
                ("url",),
            ),
        ])
    return specs


def stratified_sample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or len(frame) <= limit:
        return frame.reset_index(drop=True)
    pieces = []
    proportions = frame["label_binary"].value_counts(normalize=True)
    for label, proportion in proportions.items():
        group = frame[frame["label_binary"].eq(label)]
        size = min(len(group), max(1, round(limit * proportion)))
        pieces.append(group.sample(n=size, random_state=seed))
    return pd.concat(pieces, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)


def scores(model: object, values: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(values), dtype=float)[:, 1]
    raw = np.asarray(model.decision_function(values), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))


def selected_feature_names(model: Pipeline, columns: tuple[str, ...]) -> list[str]:
    selector = model.named_steps.get("selector")
    if selector is None or selector == "passthrough":
        return list(columns)
    if hasattr(selector, "get_support"):
        return [
            column for column, keep in zip(columns, selector.get_support()) if keep
        ]
    return list(columns)


def recipe_members(
    view: str, recipe: str, available: set[str], selector_suffix: str,
) -> tuple[list[str], list[str]]:
    """Resolve a prespecified stacking recipe inside one feature branch."""
    requested = [
        f"{view}__{algorithm}{selector_suffix}"
        for algorithm in STACKING_RECIPES[recipe]
    ]
    members = [name for name in requested if name in available]
    missing = [name for name in requested if name not in available]
    if len(members) < 2:
        raise RuntimeError(
            f"{view} / {recipe} has fewer than two available base learners. "
            f"Missing: {missing}"
        )
    return members, missing


def prediction_table(
    subset: pd.DataFrame, score_matrix: pd.DataFrame,
    selected_name: str, selected_threshold: float,
) -> pd.DataFrame:
    output = subset[[
        "sample_id", "serial_number", "url", "registered_domain",
        "label_binary", "split",
    ]].reset_index(drop=True).copy()
    for column in score_matrix.columns:
        output[f"score__{column}"] = score_matrix[column].to_numpy()
    output["selected_model"] = selected_name
    output["selected_score"] = score_matrix[selected_name].to_numpy()
    output["selected_prediction"] = (
        output["selected_score"] >= selected_threshold
    ).astype(int)
    return output


def main() -> None:
    args = arguments()
    if args.cv_folds < 3:
        raise ValueError("--cv-folds must be at least 3")
    output = project_path(args.output_dir)
    (output / "models").mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(parents=True, exist_ok=True)

    frame = read_processed(args.data)
    # Old files contain train/val/test. Merging train+val gives the requested
    # two-set protocol without touching Test or forcing an immediate rebuild.
    train = frame[~frame["split"].eq("test")].reset_index(drop=True)
    test = frame[frame["split"].eq("test")].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Processed data must contain non-empty Train and Test sets")
    if args.quick:
        train = stratified_sample(train, args.train_limit, args.seed)

    y_train = train["label_binary"].to_numpy(dtype=int)
    y_test = test["label_binary"].to_numpy(dtype=int)
    groups = train["registered_domain"].to_numpy()
    folds = list(StratifiedGroupKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.seed,
    ).split(train, y_train, groups=groups))

    specs = model_specs(args)
    print(f"Model zoo: {len(specs)} base configurations")
    oof_base = pd.DataFrame(index=np.arange(len(train)))
    test_base = pd.DataFrame(index=np.arange(len(test)))
    member_rows: list[dict[str, object]] = []

    for member_number, spec in enumerate(specs, start=1):
        print(f"[{member_number}/{len(specs)}] OOF {spec.name}")
        member_oof = np.zeros(len(train), dtype=float)
        for fold_number, (fit_index, holdout_index) in enumerate(folds, start=1):
            print(f"  fold {fold_number}/{args.cv_folds}")
            model = clone(spec.estimator)
            model.fit(train.iloc[fit_index][list(spec.columns)], y_train[fit_index])
            member_oof[holdout_index] = scores(
                model, train.iloc[holdout_index][list(spec.columns)]
            )
        oof_base[spec.name] = member_oof

        print("  refit on all Train rows")
        model = clone(spec.estimator)
        model.fit(train[list(spec.columns)], y_train)
        test_base[spec.name] = scores(model, test[list(spec.columns)])
        model_path = output / "models" / f"{spec.name}.joblib"
        joblib.dump(model, model_path, compress=3)

        oof_metrics = evaluate_binary(y_train, member_oof, args.threshold)
        chosen_features = (
            selected_feature_names(model, spec.columns)
            if spec.view != "url_text" else ["character_ngrams_from_url"]
        )
        member_rows.append({
            "model": spec.name, "view": spec.view, "family": spec.family,
            "input_feature_count": len(spec.columns),
            "selected_feature_count": len(chosen_features),
            "train_oof_pr_auc": float(oof_metrics["pr_auc"]),
            "train_oof_mcc": float(oof_metrics["mcc"]),
            "train_oof_f1": float(oof_metrics["f1"]),
            "evaluation_threshold": args.threshold, "model_path": str(model_path),
            "input_features": "|".join(spec.columns),
            "selected_features": "|".join(chosen_features),
        })

    base_table = pd.DataFrame(member_rows)
    oof_all = oof_base.copy()
    test_all = test_base.copy()
    selector_suffix = (
        "" if args.feature_selector == "none" else f"__fs_{args.feature_selector}"
    )
    available = set(oof_base.columns)
    stacking_rows: list[dict[str, object]] = []

    # Build the same two stacking designs independently in Brand-URL, HTML,
    # and Full. No stack is allowed to consume predictions from another view.
    stacking_views = [view for view in PRIMARY_VIEWS if view in args.views]
    for view in stacking_views:
        for recipe in STACKING_RECIPES:
            members, missing = recipe_members(
                view, recipe, available, selector_suffix
            )
            stack_name = f"{view}__{recipe}"
            print(f"OOF {stack_name}: {members}")

            # Cross-fit the meta layer so its Train score is not an in-sample
            # fit of Logistic Regression on the complete OOF matrix.
            meta_oof = np.zeros(len(train), dtype=float)
            for fit_index, holdout_index in folds:
                meta_fold = LogisticRegression(
                    max_iter=4000, random_state=args.seed
                )
                meta_fold.fit(
                    oof_base.iloc[fit_index][members], y_train[fit_index]
                )
                meta_oof[holdout_index] = meta_fold.predict_proba(
                    oof_base.iloc[holdout_index][members]
                )[:, 1]

            meta = LogisticRegression(max_iter=4000, random_state=args.seed)
            meta.fit(oof_base[members], y_train)
            oof_all[stack_name] = meta_oof
            test_all[stack_name] = meta.predict_proba(test_base[members])[:, 1]
            joblib.dump(meta, output / "models" / f"{stack_name}.joblib", compress=3)

            for member, coefficient in zip(members, meta.coef_[0]):
                stacking_rows.append({
                    "stacking_model": stack_name,
                    "feature_view": view,
                    "recipe": recipe,
                    "meta_classifier": "logistic_regression",
                    "base_member": member,
                    "meta_coefficient": float(coefficient),
                    "missing_prespecified_members": "|".join(missing),
                })

    metric_rows: list[dict[str, object]] = []
    for model_name in oof_all.columns:
        metric_rows.append({
            "model": model_name, "split": "train_oof",
            "decision_threshold": args.threshold,
            **evaluate_binary(
                y_train, oof_all[model_name].to_numpy(), args.threshold
            ),
        })
        metric_rows.append({
            "model": model_name, "split": "test",
            "decision_threshold": args.threshold,
            **evaluate_binary(
                y_test, test_all[model_name].to_numpy(), args.threshold
            ),
        })

    metrics = pd.DataFrame(metric_rows)
    train_ranking = metrics[metrics["split"].eq("train_oof")].sort_values(
        ["pr_auc", "mcc", "f1"], ascending=False
    )
    reference_name = args.reference_model
    if args.feature_selector != "none" and reference_name not in oof_all:
        reference_name = f"{reference_name}{selector_suffix}"
    if reference_name not in oof_all:
        raise ValueError(
            f"Pre-specified --reference-model is unavailable: {reference_name}"
        )
    metrics["prespecified_reference_model"] = metrics["model"].eq(reference_name)

    metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    base_table.to_csv(output / "base_members.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stacking_rows).to_csv(
        output / "stacking_members.csv", index=False, encoding="utf-8-sig"
    )

    prediction_table(train, oof_all, reference_name, args.threshold).to_csv(
        output / "predictions" / "train_oof.csv", index=False,
        encoding="utf-8-sig",
    )
    prediction_table(test, test_all, reference_name, args.threshold).to_csv(
        output / "predictions" / "test.csv", index=False, encoding="utf-8-sig",
    )

    selected = {
        "model": reference_name, "threshold": args.threshold,
        "project_author": PROJECT_AUTHOR,
        "selection_source": "pre-specified before Test",
        "selection_order": None,
        "threshold_source": "fixed by protocol",
        "test_used_for_selection": False,
    }
    (output / "selected_model.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    configuration = {
        "dataset": "CompPhish v4",
        "project_author": PROJECT_AUTHOR,
        "project_repository": PROJECT_REPOSITORY,
        "processed_data": str(project_path(args.data, must_exist=True)),
        "split": "registered-domain-disjoint Train/Test",
        "train_rows": len(train), "test_rows": len(test),
        "cv": "StratifiedGroupKFold inside Train, grouped by registered_domain",
        "cv_folds": args.cv_folds, "model_set": args.model_set,
        "extended_all_views": args.extended_all_views,
        "views": args.views, "feature_selector": args.feature_selector,
        "dropped_features": sorted(set(args.drop_features)),
        "drop_visible_url_cues": args.drop_visible_url_cues,
        "include_fold_safe_mean_features": args.include_fold_safe_mean_features,
        "fold_safe_mean_feature_rule": (
            "thresholds fitted from phishing rows inside each OOF training fold; "
            "final thresholds fitted from all Train rows; Test never used"
            if args.include_fold_safe_mean_features else None
        ),
        "resampling": "none", "class_weight": "none",
        "published_global_mean_values_used": False,
        "fold_safe_label_conditioned_features_used": (
            args.include_fold_safe_mean_features
        ),
        "full_view_feature_count": (
            len(FEATURE_SETS_PUBLISHED["full"])
            if args.include_fold_safe_mean_features
            else len(FEATURE_SETS["full"])
        ),
        "decision_threshold": args.threshold,
        "reference_model": reference_name,
        "ensemble_training": (
            "branch-aligned two-level OOF stacking; no cross-view mixing; "
            "Test unused for training"
        ),
        "stacking_recipes": {
            name: list(members) for name, members in STACKING_RECIPES.items()
        },
        "quick": args.quick,
        "seed": args.seed,
    }
    (output / "experiment_config.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    display = [
        "model", "accuracy", "precision", "recall", "specificity", "f1",
        "pr_auc", "roc_auc", "mcc", "fpr", "fnr",
    ]
    print("\nTrain OOF ranking:")
    print(train_ranking[display].to_string(index=False))
    print(
        f"\nPre-specified reporting reference: {reference_name}; "
        f"threshold={args.threshold:.6f}"
    )
    print("\nTest results (not used for selection):")
    test_ranking = metrics[metrics["split"].eq("test")].sort_values(
        ["pr_auc", "mcc", "f1"], ascending=False
    )
    print(test_ranking[display].to_string(index=False))
    print(f"\nSaved experiment: {output}")


if __name__ == "__main__":
    main()
