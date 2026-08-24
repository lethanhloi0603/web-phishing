import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import dom_fingerprint, html_sha256, normalize_generator
from src.research import sha256_file


WEB_EXTENSIONS = {".html", ".htm", ".ejs", ".php", ".jsx", ".tsx", ".txt"}
POSITIVE_TOKENS = {
    "phishing", "phish", "malicious", "adversarial", "apw",
    "apw-lab", "apw_lab", "apw-wild", "apw_wild",
    "unperturbed phishing", "unperturbed_phishing", "unperturbed-phishing",
}
NEGATIVE_TOKENS = {"benign", "legitimate", "legit", "not-phishing", "not_phishing"}
GENERIC_PATH_TOKENS = {
    "data", "dataset", "datasets", "html", "pages", "webpages", "website", "websites",
    "samples", "sample", "files", "archive", "train", "training", "test", "testing",
    "validation", "val", "src", "public", "private",
}
GENERIC_BRANDS = {
    "", "unknown", "none", "nan", "null", "benign", "legit", "legitimate",
    "phish", "phishing", "malicious",
}


def normalized_label(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "phish", "phishing", "malicious", "adversarial"}:
        return 1
    if text in {"0", "false", "benign", "legitimate", "legit", "not-phishing"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def normalized_brand(value):
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return "unknown" if text.lower() in GENERIC_BRANDS else text


def infer_path_label(path):
    tokens = {part.lower() for part in path.parts}
    positive = bool(tokens & POSITIVE_TOKENS)
    negative = bool(tokens & NEGATIVE_TOKENS)
    if positive == negative:
        return None
    return 1 if positive else 0


def infer_path_brand(path):
    """Infer a brand directory while ignoring label and generic layout folders."""
    candidates = []
    for part in path.parts[:-1]:
        value = str(part).strip().lower()
        if (
            not value
            or value in POSITIVE_TOKENS
            or value in NEGATIVE_TOKENS
            or value in GENERIC_PATH_TOKENS
        ):
            continue
        candidates.append(str(part).strip())
    return candidates[-1] if candidates else "unknown"


def make_row(
    html, label, source, generator, relative, url="", brand="unknown", date="",
    dataset_domain="adversarial_external", dataset_version="unknown",
    label_provenance="published_dataset", codebase_id="",
    dataset_license="unknown", dataset_citation="",
):
    html = str(html)
    html_hash = html_sha256(html)
    dom_hash = dom_fingerprint(html)
    return {
        "sample_id": html_hash[:16],
        "html": html,
        "label": int(label),
        "source": source,
        "generator": generator,
        "generator_family": normalize_generator(generator),
        "brand": normalized_brand(brand),
        "template_id": dom_hash,
        "template_id_source": "dom_fingerprint",
        "codebase_id": str(codebase_id or html_hash[:24]),
        "page_label": int(label),
        "site_label": int(label),
        "label_unit": "page",
        "page_role": "standalone",
        "prompt_id": "unknown",
        "url": str(url or ""),
        "collection_date": str(date or ""),
        "date_source": "provided" if date else "unknown",
        "file_path": str(relative),
        "html_hash": html_hash,
        "dom_hash": dom_hash,
        "quality_level": "external_publication",
        "manual_validated": 0,
        "published_label": 1,
        "label_provenance": label_provenance,
        "dataset_domain": dataset_domain,
        "dataset_version": dataset_version,
        "dataset_provenance": source,
        "dataset_license": str(dataset_license or "unknown"),
        "dataset_citation": str(dataset_citation or ""),
    }


def url_site_identifier(source, url):
    """Return a stable site group so pages from one host are not pseudo-replicates."""
    value = str(url or "").strip()
    if not value:
        return ""
    parsed = urlsplit(value if "://" in value else f"//{value}")
    hostname = (parsed.hostname or "").strip(".").lower()
    if not hostname:
        return ""
    raw = f"{source}:host:{hostname}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def codebase_identifier(source, archive_path, member_path=None):
    archive_path = Path(archive_path)
    if member_path is not None:
        member_path = Path(member_path)
        site_component = member_path.parts[0] if len(member_path.parts) > 1 else archive_path.stem
    else:
        site_component = str(archive_path.parent) if str(archive_path.parent) != "." else archive_path.stem
    raw = f"{source}:{archive_path}:{site_component}".lower()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def archive_site_component(member_path, archive_stem):
    parts = Path(member_path).parts
    return parts[0] if len(parts) > 1 else archive_stem


def select_one_member_per_site(infos, archive_stem, preferred_names):
    preference = {
        name.strip().lower(): index
        for index, name in enumerate(preferred_names)
        if name.strip()
    }
    grouped = {}
    for info in infos:
        if info.is_dir() or Path(info.filename).suffix.lower() not in WEB_EXTENSIONS:
            continue
        member_path = Path(info.filename)
        site = archive_site_component(member_path, archive_stem)
        rank = (
            preference.get(member_path.name.lower(), len(preference) + 1),
            len(member_path.parts),
            -int(info.file_size),
            info.filename.lower(),
        )
        if site not in grouped or rank < grouped[site][0]:
            grouped[site] = (rank, info)
    return [grouped[site][1] for site in sorted(grouped)]


def import_directory(args):
    root = Path(args.input)
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    infos = archive.infolist()
                    if getattr(args, "one_page_per_site", False):
                        preferred = [
                            value.strip() for value in getattr(
                                args, "preferred_page_names",
                                "original.html,index.html,clean.html",
                            ).split(",") if value.strip()
                        ]
                        infos = select_one_member_per_site(
                            infos, path.stem, preferred
                        )
                    for info in infos:
                        if info.is_dir() or Path(info.filename).suffix.lower() not in WEB_EXTENSIONS:
                            continue
                        member_path = Path(info.filename)
                        label = infer_path_label(relative)
                        if label is None:
                            label = infer_path_label(member_path)
                        if label is None:
                            continue
                        # Bound decompression and never extract archive paths to disk.
                        if info.file_size > getattr(args, "max_member_bytes", 5_000_000):
                            continue
                        with archive.open(info) as member:
                            raw = member.read(args.max_chars + 1)
                        text = raw.decode("utf-8", errors="ignore")[: args.max_chars]
                        if len(text.strip()) < args.min_chars:
                            continue
                        virtual_path = Path(str(relative) + "!" + info.filename)
                        archive_brand_path = Path(
                            *relative.parts[:-1], path.stem, "archive-member.html"
                        )
                        member_brand_path = Path(
                            *relative.parts[:-1], *member_path.parts
                        )
                        inferred_brand = infer_path_brand(archive_brand_path)
                        if inferred_brand == "unknown":
                            inferred_brand = infer_path_brand(member_brand_path)
                        rows.append(make_row(
                            text,
                            label,
                            args.source,
                            args.generator,
                            virtual_path,
                            brand=(
                                inferred_brand
                                if getattr(args, "brand_from_path", False)
                                else "unknown"
                            ),
                            date=datetime.fromtimestamp(
                                path.stat().st_mtime, tz=timezone.utc
                            ).date().isoformat(),
                            dataset_domain=getattr(
                                args, "dataset_domain", "adversarial_external"
                            ),
                            dataset_version=getattr(args, "dataset_version", "unknown"),
                            label_provenance=getattr(
                                args, "label_provenance", "published_dataset"
                            ),
                            codebase_id=codebase_identifier(
                                args.source, relative, member_path
                            ),
                            dataset_license=getattr(args, "license", "unknown"),
                            dataset_citation=getattr(args, "citation", ""),
                        ))
            except (OSError, zipfile.BadZipFile):
                continue
            continue
        if path.suffix.lower() not in WEB_EXTENSIONS:
            continue
        label = infer_path_label(relative)
        if label is None:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text.strip()) < args.min_chars:
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()
        rows.append(make_row(
            text[:args.max_chars], label, args.source, args.generator, relative,
            brand=(
                infer_path_brand(relative)
                if getattr(args, "brand_from_path", False)
                else "unknown"
            ),
            date=modified, dataset_domain=getattr(args, "dataset_domain", "adversarial_external"),
            dataset_version=getattr(args, "dataset_version", "unknown"),
            label_provenance=getattr(args, "label_provenance", "published_dataset"),
            codebase_id=codebase_identifier(args.source, relative),
            dataset_license=getattr(args, "license", "unknown"),
            dataset_citation=getattr(args, "citation", ""),
        ))
    return pd.DataFrame(rows)


def read_parquet_bounded(args, input_path):
    """Read only selected rows in small batches; never materialize all raw HTML."""
    input_path = Path(input_path)
    parquet = pq.ParquetFile(input_path)
    available = set(parquet.schema_arrow.names)
    required = {args.html_column, args.label_column}
    missing = required - available
    if missing:
        raise ValueError(
            f"Missing PARQUET columns: {sorted(missing)}; available={sorted(available)}"
        )

    identity_column = next(
        (
            column for column in (
                "sha256", args.url_column, args.site_column
            )
            if column and column in available
        ),
        "",
    )
    metadata_columns = [args.label_column]
    if identity_column and identity_column not in metadata_columns:
        metadata_columns.append(identity_column)
    metadata = parquet.read(columns=metadata_columns).to_pandas()
    metadata["_normalized_label"] = metadata[args.label_column].map(normalized_label)
    identity = (
        metadata[identity_column].fillna("").astype(str)
        if identity_column
        else pd.Series(metadata.index.astype(str), index=metadata.index)
    )
    seed = int(getattr(args, "sample_seed", 42))
    metadata["_import_rank"] = identity.map(
        lambda value: hashlib.sha256(
            f"{seed}:{value}".encode("utf-8", errors="ignore")
        ).hexdigest()
    )
    max_per_class = int(getattr(args, "max_per_class", 0) or 0)
    if max_per_class > 0:
        selected_metadata = (
            metadata.sort_values(
                ["_normalized_label", "_import_rank"], kind="stable"
            )
            .groupby("_normalized_label", sort=True, group_keys=False)
            .head(max_per_class)
        )
    else:
        selected_metadata = metadata
    selected_indices = set(int(value) for value in selected_metadata.index)

    requested_columns = [
        args.html_column, args.label_column, args.url_column,
        getattr(args, "site_column", ""), args.brand_column, args.date_column,
    ]
    data_columns = []
    for column in requested_columns:
        if column and column in available and column not in data_columns:
            data_columns.append(column)
    batch_size = int(getattr(args, "parquet_batch_size", 4) or 4)
    if batch_size < 1:
        raise ValueError("--parquet-batch-size must be positive")
    rows = []
    offset = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=data_columns, use_threads=False
    ):
        local_indices = [
            local for local in range(batch.num_rows)
            if offset + local in selected_indices
        ]
        if local_indices:
            selected_batch = batch.take(pa.array(local_indices, type=pa.int64()))
            for local, record in zip(local_indices, selected_batch.to_pylist()):
                record[args.html_column] = str(
                    record.get(args.html_column, "") or ""
                )[: args.max_chars]
                record["_source_row_index"] = offset + local
                rows.append(record)
        offset += batch.num_rows
    return pd.DataFrame(rows)


def import_tabular(args, input_format="csv", input_path=None):
    input_path = Path(input_path or args.input)
    source = (
        read_parquet_bounded(args, input_path)
        if input_format == "parquet"
        else pd.read_csv(input_path)
    )
    required = {args.html_column, args.label_column}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(
            f"Missing {input_format.upper()} columns: {sorted(missing)}; "
            f"available={list(source.columns)}"
        )
    source = source.dropna(subset=[args.html_column, args.label_column]).copy()
    max_per_class = int(getattr(args, "max_per_class", 0) or 0)
    if max_per_class > 0 and input_format != "parquet":
        source["_normalized_label"] = source[args.label_column].map(normalized_label)
        source["_import_rank"] = source.apply(
            lambda row: hashlib.sha256(
                (
                    f"{getattr(args, 'sample_seed', 42)}:"
                    f"{row.get(args.url_column, '') if args.url_column else ''}:"
                    f"{row[args.html_column]}"
                ).encode("utf-8", errors="ignore")
            ).hexdigest(),
            axis=1,
        )
        source = (
            source.sort_values(["_normalized_label", "_import_rank"], kind="stable")
            .groupby("_normalized_label", sort=True, as_index=False, group_keys=False)
            .head(max_per_class)
            .drop(columns=["_normalized_label", "_import_rank"])
        )
    rows = []
    for index, row in source.iterrows():
        html = str(row[args.html_column])
        if len(html.strip()) < args.min_chars:
            continue
        url = row.get(args.url_column, "") if args.url_column else ""
        explicit_site = (
            row.get(args.site_column, "") if getattr(args, "site_column", "") else ""
        )
        site_value = (
            "" if explicit_site is None or pd.isna(explicit_site)
            else str(explicit_site).strip()
        )
        codebase_id = (
            hashlib.sha256(
                f"{args.source}:site:{site_value.lower()}".encode("utf-8")
            ).hexdigest()[:24]
            if site_value
            else url_site_identifier(args.source, url)
        )
        rows.append(make_row(
            html[:args.max_chars], normalized_label(row[args.label_column]), args.source,
            args.generator,
            f"{input_path.name}:row:{row.get('_source_row_index', index)}",
            url=url,
            brand=row.get(args.brand_column, "unknown") if args.brand_column else "unknown",
            date=row.get(args.date_column, "") if args.date_column else "",
            dataset_domain=getattr(args, "dataset_domain", "adversarial_external"),
            dataset_version=getattr(args, "dataset_version", "unknown"),
            label_provenance=getattr(args, "label_provenance", "published_dataset"),
            codebase_id=codebase_id,
            dataset_license=getattr(args, "license", "unknown"),
            dataset_citation=getattr(args, "citation", ""),
        ))
    return pd.DataFrame(rows)


def import_csv(args):
    return import_tabular(args, "csv")


def import_parquet(args):
    return import_tabular(args, "parquet")


def import_parquet_directory(args):
    paths = sorted(Path(args.input).glob("*.parquet"))
    if not paths:
        raise RuntimeError(f"No Parquet files found in {args.input}")
    frames = [import_tabular(args, "parquet", path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def deterministic_class_cap(frame, max_per_class, seed):
    """Storage-bounded, order-independent class sampling for large publications."""
    if not max_per_class or max_per_class < 1:
        return frame
    selected = []
    for _, group in frame.groupby("label", sort=True):
        ranked = group.assign(
            _rank=group["html_hash"].map(
                lambda value: hashlib.sha256(
                    f"{seed}:{value}".encode("utf-8")
                ).hexdigest()
            )
        ).sort_values(["_rank", "html_hash"], kind="stable")
        selected.append(ranked.head(max_per_class).drop(columns="_rank"))
    return pd.concat(selected, ignore_index=True) if selected else frame.iloc[0:0]


def main():
    parser = argparse.ArgumentParser(description="Normalize a published phishing dataset to the project schema")
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--format",
        choices=["directory", "csv", "parquet", "parquet-directory"],
        required=True,
    )
    parser.add_argument("--source", required=True, help="Stable dataset name, e.g. phishoracle")
    parser.add_argument(
        "--dataset-domain",
        choices=["real_world", "llm_generated", "adversarial_external"],
        default="adversarial_external",
    )
    parser.add_argument("--dataset-version", default="unknown")
    parser.add_argument("--label-provenance", default="published_dataset")
    parser.add_argument("--citation", default="", help="Paper DOI/arXiv/URL stored in the sidecar manifest")
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--generator", default="human_kit")
    parser.add_argument("--html-column", default="html")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--url-column", default="url")
    parser.add_argument(
        "--site-column", default="",
        help="Optional explicit site/domain column; otherwise derive the host from URL",
    )
    parser.add_argument("--brand-column", default="brand")
    parser.add_argument(
        "--brand-from-path", action="store_true",
        help="For directory imports, infer the nearest non-generic parent as brand",
    )
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=100000)
    parser.add_argument(
        "--max-per-class", type=int, default=0,
        help="Deterministically cap each class after normalization (0 keeps all)",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--parquet-batch-size", type=int, default=4,
        help="Bound raw-HTML decompression memory for Parquet inputs",
    )
    parser.add_argument(
        "--drop-site-label-conflicts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop every site/domain that has both benign and phishing labels",
    )
    parser.add_argument(
        "--max-member-bytes", type=int, default=5_000_000,
        help="Reject oversized archive members while still reading only --max-chars",
    )
    parser.add_argument(
        "--one-page-per-site", action="store_true",
        help=(
            "Within each ZIP, keep one canonical HTML-like member per top-level "
            "site directory to avoid pseudo-replication"
        ),
    )
    parser.add_argument(
        "--preferred-page-names",
        default="original.html,index.html,clean.html",
        help="Priority order used with --one-page-per-site",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    importers = {
        "directory": import_directory,
        "csv": import_csv,
        "parquet": import_parquet,
        "parquet-directory": import_parquet_directory,
    }
    frame = importers[args.format](args)
    if frame.empty:
        raise RuntimeError("No labeled webpage samples were imported")
    before = len(frame)
    frame = frame.drop_duplicates("html_hash", keep="first").reset_index(drop=True)
    frame = deterministic_class_cap(
        frame, args.max_per_class, args.sample_seed
    ).reset_index(drop=True)
    site_label_counts = frame.groupby("codebase_id")["label"].nunique()
    conflicting_sites = set(site_label_counts[site_label_counts.gt(1)].index)
    rows_removed_for_site_conflict = 0
    if args.drop_site_label_conflicts and conflicting_sites:
        rows_removed_for_site_conflict = int(
            frame["codebase_id"].isin(conflicting_sites).sum()
        )
        frame = frame[~frame["codebase_id"].isin(conflicting_sites)].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("No samples remain after duplicate/site-conflict filtering")
    known_brands = frame["brand"].fillna("").astype(str).str.strip().str.lower()
    known_brands = known_brands[~known_brands.isin({"", "unknown", "none", "nan"})]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    manifest = {
        "source": args.source,
        "dataset_domain": args.dataset_domain,
        "dataset_version": args.dataset_version,
        "label_provenance": args.label_provenance,
        "citation": args.citation,
        "license": args.license,
        "input": str(Path(args.input).resolve()),
        "input_format": args.format,
        "output": output.name,
        "output_sha256": sha256_file(output),
        "rows": int(len(frame)),
        "benign": int(frame["label"].eq(0).sum()),
        "phishing": int(frame["label"].eq(1).sum()),
        "unique_templates": int(frame["template_id"].nunique()),
        "known_brands": int(known_brands.nunique()),
        "max_chars": args.max_chars,
        "max_per_class": args.max_per_class,
        "sample_seed": args.sample_seed,
        "parquet_batch_size": args.parquet_batch_size,
        "drop_site_label_conflicts": args.drop_site_label_conflicts,
        "conflicting_sites_detected": len(conflicting_sites),
        "rows_removed_for_site_conflict": rows_removed_for_site_conflict,
        "max_member_bytes": args.max_member_bytes,
        "one_page_per_site": bool(args.one_page_per_site),
        "preferred_page_names": args.preferred_page_names,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Imported {len(frame)}/{before} unique samples to {output}")
    print(f"Provenance manifest: {manifest_path}")
    print(frame.groupby("label").size().rename(index={0: "benign", 1: "phishing"}).to_string())


if __name__ == "__main__":
    main()
