import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import dom_fingerprint, html_sha256, normalize_generator
from src.research import sha256_file, split_integrity


REQUIRED = {"html", "label", "source", "generator"}


def load_and_enrich(path):
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = df.dropna(subset=["html", "label"]).copy()
    df["html"] = df["html"].astype(str)
    df["label"] = df["label"].astype(int)
    df["source"] = df["source"].fillna("unknown").astype(str).str.lower()
    df["generator"] = df["generator"].fillna("unknown").astype(str).str.lower()

    if "generator_family" not in df:
        df["generator_family"] = df["generator"].map(normalize_generator)
    else:
        df["generator_family"] = (
            df["generator_family"].fillna("unknown").astype(str).map(normalize_generator)
        )
    if "html_hash" not in df:
        if "sha256" in df:
            df["html_hash"] = df["sha256"].astype(str)
        else:
            df["html_hash"] = df["html"].map(html_sha256)
    if "dom_hash" not in df:
        df["dom_hash"] = df["html"].map(dom_fingerprint)

    if "template_id" not in df:
        df["template_id"] = ""
    supplied_template = df["template_id"].fillna("").astype(str).str.strip()
    df["template_id_source"] = supplied_template.ne("").map(
        {True: "provided", False: "dom_fingerprint"}
    )
    df["template_id"] = supplied_template.where(supplied_template.ne(""), df["dom_hash"])

    if "dataset_provenance" not in df:
        df["dataset_provenance"] = df["source"]
    defaults = {
        "brand": "unknown", "prompt_id": "unknown", "quality_level": "unknown",
        "manual_validated": 0, "url": "", "collection_date": "",
        "label_provenance": "unknown",
    }
    for column, default in defaults.items():
        if column not in df:
            if column == "collection_date" and "date" in df:
                df[column] = df["date"]
            else:
                df[column] = default
        df[column] = df[column].fillna(default)
    if "dataset_domain" not in df:
        df["dataset_domain"] = df["source"].map(
            lambda value: "llm_generated" if value == "llm" else "real_world"
        )
    df["dataset_domain"] = df["dataset_domain"].fillna("unknown").astype(str).str.lower()
    if "sample_id" not in df:
        df["sample_id"] = df["html_hash"].str[:16]
    if "page_label" not in df:
        df["page_label"] = df["label"]
    if "site_label" not in df:
        df["site_label"] = df["label"]
    if "codebase_id" not in df:
        df["codebase_id"] = df["sample_id"]
    df["codebase_id"] = df["codebase_id"].fillna(df["sample_id"]).astype(str)
    if "label_unit" not in df:
        df["label_unit"] = "page"
    if "page_role" not in df:
        df["page_role"] = "standalone"

    before = len(df)
    df = df.drop_duplicates(subset=["html_hash"], keep="first").reset_index(drop=True)
    print(f"{path}: loaded={before}, unique_html={len(df)}")
    return df


def filter_validated_llm(
    df,
    require_validated=False,
    allow_ai_validated=False,
    allow_published_labels=False,
):
    manual = pd.to_numeric(df["manual_validated"], errors="coerce").fillna(0).astype(int).eq(1)
    accepted = manual.copy()
    if allow_ai_validated and "ai_validated" in df:
        ai_accepted = pd.to_numeric(df["ai_validated"], errors="coerce").fillna(0).astype(int).eq(1)
        accepted |= ai_accepted
        print(f"AI-assisted accepted rows enabled: {int(ai_accepted.sum())}")
    if allow_published_labels and "published_label" in df:
        published = pd.to_numeric(df["published_label"], errors="coerce").fillna(0).astype(int).eq(1)
        accepted |= published
        print(f"Published-study labels enabled: {int(published.sum())}")
    if accepted.any():
        result = df[accepted].copy()
        print(f"LLM validation filter: {len(df)} -> {len(result)}")
        return result
    if require_validated:
        raise ValueError(
            "No manual_validated=1 LLM rows. Complete the audit before final experiments."
        )
    print("WARNING: no manual_validated=1 rows; using all LLM rows for pipeline testing")
    return df


def apply_audit(df, audit_path):
    audit = pd.read_csv(audit_path)
    if "sample_id" not in audit:
        raise ValueError(f"{audit_path}: missing sample_id")
    audit = audit.drop_duplicates("sample_id", keep="last").set_index("sample_id")
    result = df.copy()
    for column in (
        "manual_validated", "generator", "generator_family", "brand", "template_id",
        "prompt_id", "quality_level", "ai_validated", "validation_status", "validated_by",
        "validation_reason", "review_status", "page_label", "site_label",
        "label_unit", "page_role", "codebase_id",
    ):
        if column in audit:
            mapped = result["sample_id"].map(audit[column])
            if column in result:
                result[column] = mapped.where(mapped.notna(), result[column])
            else:
                result[column] = mapped
    result["generator_family"] = result["generator_family"].map(normalize_generator)
    print(f"Applied audit metadata from {audit_path}")
    return result


def group_split(df, validation_size=0.15, seed=42):
    if len(df) < 2:
        raise ValueError("Need at least two rows for a train/validation split")
    groups = df["template_id"].fillna(df["html_hash"]).astype(str)
    if groups.nunique() < 2:
        raise ValueError("Need at least two template groups; annotate template_id first")

    splitter = GroupShuffleSplit(n_splits=80, test_size=validation_size, random_state=seed)
    candidates = []
    overall_rate = df["label"].mean()
    family_group_counts = df.groupby("generator_family")["template_id"].nunique()
    required_families = set(family_group_counts[family_group_counts.ge(2)].index)
    overall_family_rates = df["generator_family"].value_counts(normalize=True)
    for train_index, val_index in splitter.split(df, df["label"], groups):
        train = df.iloc[train_index]
        val = df.iloc[val_index]
        if df["label"].nunique() > 1:
            if train["label"].nunique() < 2 or val["label"].nunique() < 2:
                continue
        if len(required_families) > 1:
            if not required_families.issubset(set(train["generator_family"])):
                continue
            if not required_families.issubset(set(val["generator_family"])):
                continue
        size_error = abs(len(val) / len(df) - validation_size)
        balance_error = abs(val["label"].mean() - overall_rate)
        val_family_rates = val["generator_family"].value_counts(normalize=True)
        family_error = sum(
            abs(val_family_rates.get(family, 0.0) - rate)
            for family, rate in overall_family_rates.items()
        )
        candidates.append((size_error + balance_error + family_error, train_index, val_index))
    if not candidates:
        raise ValueError(
            "Could not create a class-valid group split. Add more independent templates."
        )
    _, train_index, val_index = min(candidates, key=lambda item: item[0])
    return (
        df.iloc[train_index].reset_index(drop=True),
        df.iloc[val_index].reset_index(drop=True),
    )


def group_split_three(df, validation_size=0.15, test_size=0.15, seed=42):
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to < 1")
    train_val, test = group_split(df, validation_size=test_size, seed=seed)
    relative_validation_size = validation_size / (1.0 - test_size)
    train, val = group_split(
        train_val, validation_size=relative_validation_size, seed=seed + 1
    )
    return train, val, test


def combine_unique(*frames):
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["html_hash"], keep="first")
        .reset_index(drop=True)
    )


def compose_holdout_test(held_out, reference_test, include_human_phishing=False):
    """Build a held-out-family test with optional human controls from both labels."""
    human_controls = (
        reference_test.copy()
        if include_human_phishing
        else reference_test[reference_test["label"].eq(0)].copy()
    )
    return combine_unique(held_out, human_controls)


def compose_temporal_pool(phresh_train, phresh_test, llm, include_llm=False):
    """Keep the default temporal task within the real-world collection domain."""
    pool = combine_unique(phresh_train, phresh_test)
    return combine_unique(pool, llm) if include_llm else pool


def remove_test_leakage(test, train, val, strict_template_isolation=False):
    reference = pd.concat([train, val], ignore_index=True)
    exact_overlap = test["html_hash"].isin(set(reference["html_hash"]))
    template_overlap = test["template_id"].isin(set(reference["template_id"]))
    remove = exact_overlap.copy()
    if strict_template_isolation:
        remove |= template_overlap
    report = {
        "test_rows_before": int(len(test)),
        "exact_html_overlap": int(exact_overlap.sum()),
        "template_overlap_observed": int(template_overlap.sum()),
        "strict_template_isolation": bool(strict_template_isolation),
        "test_rows_removed": int(remove.sum()),
    }
    return test.loc[~remove].reset_index(drop=True), report


def ensure_usable(name, df, require_both_classes=True):
    if df.empty:
        raise ValueError(f"{name} is empty after leakage filtering")
    if require_both_classes and df["label"].nunique() < 2:
        raise ValueError(f"{name} must contain both benign and phishing rows")


def save_split(train, val, test, out_dir, metadata, compression="none"):
    ensure_usable("train", train)
    ensure_usable("validation", val)
    ensure_usable("test", test)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    csv_compression = "gzip" if compression == "gzip" else None
    split_paths = {
        "train": out / f"train{suffix}",
        "val": out / f"val{suffix}",
        "test": out / f"test{suffix}",
    }
    train.to_csv(split_paths["train"], index=False, compression=csv_compression)
    val.to_csv(split_paths["val"], index=False, compression=csv_compression)
    test.to_csv(split_paths["test"], index=False, compression=csv_compression)

    integrity = split_integrity(train, val, test)
    if any(
        pair.get("html_hash", 0) > 0
        for pair in integrity["overlap"].values()
    ):
        raise RuntimeError("Exact HTML leakage remains between saved splits")

    summary = {
        **metadata,
        "compression": compression,
        "files": {name: path.name for name, path in split_paths.items()},
        "file_sha256": {name: sha256_file(path) for name, path in split_paths.items()},
        "integrity": integrity,
        "splits": {
            name: {
                "rows": int(len(frame)),
                "benign": int(frame["label"].eq(0).sum()),
                "phishing": int(frame["label"].eq(1).sum()),
                "templates": int(frame["template_id"].nunique()),
                "generators": sorted(frame["generator_family"].unique().tolist()),
                "dataset_domains": sorted(frame["dataset_domain"].unique().tolist()),
                "dataset_provenance": sorted(frame["dataset_provenance"].unique().tolist()),
            }
            for name, frame in (("train", train), ("val", val), ("test", test))
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved scenario to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phresh-train", required=True)
    parser.add_argument("--phresh-test", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--llm-audit", default="")
    parser.add_argument("--require-validated", action="store_true")
    parser.add_argument("--allow-ai-validated", action="store_true")
    parser.add_argument(
        "--allow-published-labels",
        action="store_true",
        help="Use labels released by the source study; keeps them distinct from manual validation",
    )
    parser.add_argument(
        "--scenario",
        choices=["id", "human_to_llm", "mixed", "logo", "lofo", "temporal", "external"],
        required=True,
    )
    parser.add_argument("--holdout-generator", default="")
    parser.add_argument("--holdout-family", default="")
    parser.add_argument("--cutoff-date", default="")
    parser.add_argument(
        "--include-llm-temporal", action="store_true",
        help=(
            "Explicitly include dated LLM pages in scenario=temporal. By default "
            "the temporal benchmark is real-world-only to avoid mixing domains."
        ),
    )
    parser.add_argument("--external-data", default="", help="Normalized public dataset for scenario=external")
    parser.add_argument("--strict-template-isolation", action="store_true")
    parser.add_argument(
        "--include-human-phishing-test",
        action="store_true",
        help=(
            "Include human phishing as well as human benign controls in held-out tests. "
            "This removes the source=human=>benign shortcut while held-out-family metrics "
            "remain available as a subgroup."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-size", type=float, default=0.15,
        help=(
            "Fraction reserved for validation/calibration. Use about 0.5 when "
            "the 1% FPR confidence gate requires >=1000 benign validation sites."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--compression", choices=["none", "gzip"], default="none")
    args = parser.parse_args()
    if not 0.0 < args.validation_size < 0.8:
        raise ValueError("--validation-size must be between 0 and 0.8")

    phresh_train = load_and_enrich(args.phresh_train)
    phresh_test = load_and_enrich(args.phresh_test)
    llm = load_and_enrich(args.llm)
    if args.llm_audit:
        llm = apply_audit(llm, args.llm_audit)
    validation_summary = {
        "total_llm_rows": int(len(llm)),
        "manual_validated": int(
            pd.to_numeric(llm["manual_validated"], errors="coerce").fillna(0).eq(1).sum()
        ),
        "ai_validated": int(
            pd.to_numeric(llm.get("ai_validated", 0), errors="coerce").fillna(0).eq(1).sum()
        ) if "ai_validated" in llm else 0,
        "allow_ai_validated": bool(args.allow_ai_validated),
        "allow_published_labels": bool(args.allow_published_labels),
        "published_labeled": int(
            pd.to_numeric(llm.get("published_label", 0), errors="coerce").fillna(0).eq(1).sum()
        ) if "published_label" in llm else 0,
    }
    uses_llm = args.scenario in {
        "human_to_llm", "mixed", "logo", "lofo", "external"
    } or (args.scenario == "temporal" and args.include_llm_temporal)
    validation_summary["used_in_scenario"] = bool(uses_llm)
    if uses_llm:
        llm = filter_validated_llm(
            llm,
            require_validated=args.require_validated,
            allow_ai_validated=args.allow_ai_validated,
            allow_published_labels=args.allow_published_labels,
        )
    else:
        llm = llm.iloc[0:0].copy()
    validation_summary["rows_used"] = int(len(llm))
    benign_test = phresh_test[phresh_test["label"].eq(0)].copy()
    leakage = {}

    if args.scenario == "id":
        train, val = group_split(
            phresh_train, validation_size=args.validation_size, seed=args.seed
        )
        test = phresh_test.copy()

    elif args.scenario == "human_to_llm":
        train, val = group_split(
            phresh_train, validation_size=args.validation_size, seed=args.seed
        )
        test = compose_holdout_test(
            llm[llm["label"].eq(1)], phresh_test, args.include_human_phishing_test
        )

    elif args.scenario == "mixed":
        mixed_pool = combine_unique(phresh_train, llm)
        train, val, mixed_test = group_split_three(
            mixed_pool, validation_size=args.validation_size, seed=args.seed
        )
        test = combine_unique(phresh_test, mixed_test)

    elif args.scenario == "logo":
        holdout = args.holdout_generator.strip().lower()
        if not holdout:
            raise ValueError("--holdout-generator is required for scenario=logo")
        mask = llm["generator"].str.contains(holdout, regex=False)
        held_out, known = llm[mask].copy(), llm[~mask].copy()
        if held_out.empty:
            raise ValueError(f"No generator matched '{holdout}'")
        train, val = group_split(
            combine_unique(phresh_train, known),
            validation_size=args.validation_size,
            seed=args.seed,
        )
        test = compose_holdout_test(
            held_out, phresh_test, args.include_human_phishing_test
        )

    elif args.scenario == "lofo":
        family = normalize_generator(args.holdout_family)
        if not args.holdout_family.strip():
            raise ValueError("--holdout-family is required for scenario=lofo")
        mask = llm["generator_family"].eq(family)
        held_out, known = llm[mask].copy(), llm[~mask].copy()
        if held_out.empty:
            raise ValueError(f"No generator family matched '{family}'")
        train, val = group_split(
            combine_unique(phresh_train, known),
            validation_size=args.validation_size,
            seed=args.seed,
        )
        test = compose_holdout_test(
            held_out, phresh_test, args.include_human_phishing_test
        )

    elif args.scenario == "temporal":
        if not args.cutoff_date:
            raise ValueError("--cutoff-date YYYY-MM-DD is required for scenario=temporal")
        combined = compose_temporal_pool(
            phresh_train, phresh_test, llm, include_llm=args.include_llm_temporal
        )
        dates = pd.to_datetime(combined["collection_date"], errors="coerce", utc=True)
        cutoff = pd.Timestamp(args.cutoff_date, tz="UTC")
        dated = combined[dates.notna()].copy()
        dates = dates[dates.notna()]
        old, test = dated[dates.le(cutoff)].copy(), dated[dates.gt(cutoff)].copy()
        train, val = group_split(
            old, validation_size=args.validation_size, seed=args.seed
        )

    else:
        if not args.external_data:
            raise ValueError("--external-data is required for scenario=external")
        external = load_and_enrich(args.external_data)
        train, val = group_split(
            combine_unique(phresh_train, llm),
            validation_size=args.validation_size,
            seed=args.seed,
        )
        test = external.copy()

    test, leakage = remove_test_leakage(
        test, train, val, strict_template_isolation=args.strict_template_isolation
    )
    save_split(
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        args.out_dir,
        {
            "scenario": args.scenario,
            "seed": args.seed,
            "validation_size": args.validation_size,
            "leakage": leakage,
            "llm_validation": validation_summary,
            "external_data": args.external_data if args.scenario == "external" else "",
            "include_human_phishing_test": bool(args.include_human_phishing_test),
            "include_llm_temporal": bool(args.include_llm_temporal),
        },
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
