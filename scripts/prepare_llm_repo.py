import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import dom_fingerprint, html_sha256, normalize_generator
from src.research import sha256_file


GENERATOR_TOKENS = (
    "claude", "anthropic", "gemini", "bard", "google", "gpt", "openai", "chatgpt",
    "grok", "xai", "llama", "meta", "mistral", "mixtral", "qwen", "alibaba",
    "nova", "amazon",
)
DEFAULT_EXTENSIONS = (".html", ".htm", ".ejs", ".php", ".jsx", ".tsx")
PASSWORD_RE = re.compile(
    r"<input\b[^>]*(?:type\s*=\s*['\"]?password|name\s*=\s*['\"]?[^>]*(?:password|passwd|pwd))",
    re.IGNORECASE,
)
DISCLAIMER_RE = re.compile(
    r"educational|for learning|security demonstration|demo credentials|not for production",
    re.IGNORECASE,
)


def infer_generator(path: Path):
    prompt_index = next(
        (index for index, part in enumerate(path.parts) if re.fullmatch(r"prompt\d+", part.lower())),
        None,
    )
    if prompt_index is not None and prompt_index >= 2:
        family_raw = path.parts[prompt_index - 2]
        model_raw = path.parts[prompt_index - 1]
        raw = f"{family_raw}_{model_raw}"
        generator = re.sub(r"[^a-z0-9._-]+", "_", raw.lower()).strip("_") or "unknown"
        return generator, normalize_generator(family_raw)
    candidates = [part.lower() for part in path.parts if any(token in part.lower() for token in GENERATOR_TOKENS)]
    raw = candidates[-1] if candidates else "unknown"
    generator = re.sub(r"[^a-z0-9._-]+", "_", raw).strip("_") or "unknown"
    return generator, normalize_generator(raw)


def infer_codebase(relative: Path):
    parts = relative.parts
    prompt_index = next(
        (index for index, part in enumerate(parts) if re.fullmatch(r"prompt\d+", part.lower())),
        None,
    )
    boundary = prompt_index + 1 if prompt_index is not None else max(1, len(parts) - 1)
    codebase = "/".join(parts[:boundary])
    prompt_id = parts[prompt_index] if prompt_index is not None else "unknown"
    codebase_id = hashlib.sha256(codebase.lower().encode("utf-8")).hexdigest()[:24]
    return codebase, codebase_id, prompt_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-html-chars", type=int, default=100)
    parser.add_argument("--max-html-chars", type=int, default=100000)
    parser.add_argument("--brand", default="bluemind", help="Target brand used by the published generation task")
    parser.add_argument(
        "--label-unit", choices=["page", "site"], default="page",
        help="At page level only credential-entry pages are positive; site level labels all codebase pages positive",
    )
    parser.add_argument(
        "--extensions", default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated web template extensions",
    )
    parser.add_argument(
        "--group-by-codebase", action=argparse.BooleanOptionalAction, default=True,
        help="Use the parent model/prompt codebase as template group to prevent leakage",
    )
    args = parser.parse_args()

    root = Path(args.repo)
    if not root.exists():
        raise FileNotFoundError(root)

    rows = []
    seen_hashes = set()
    extensions = {
        value.strip().lower() if value.strip().startswith(".") else f".{value.strip().lower()}"
        for value in args.extensions.split(",") if value.strip()
    }
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(text.strip()) < args.min_html_chars:
            continue

        clipped = text[: args.max_html_chars]
        html_hash = html_sha256(clipped)
        if html_hash in seen_hashes:
            continue
        seen_hashes.add(html_hash)
        relative = path.relative_to(root)
        generator, family = infer_generator(relative)
        codebase, codebase_id, prompt_id = infer_codebase(relative)
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()
        dom_hash = dom_fingerprint(clipped)
        credential_collection = int(bool(PASSWORD_RE.search(clipped)))
        educational_disclaimer = int(bool(DISCLAIMER_RE.search(clipped)))
        page_label = credential_collection
        site_label = 1
        rows.append({
            "sample_id": html_hash[:16],
            "html": clipped,
            "label": page_label if args.label_unit == "page" else site_label,
            "page_label": page_label,
            "site_label": site_label,
            "label_unit": args.label_unit,
            "page_role": "credential_entry" if credential_collection else "auxiliary",
            "source": "llm",
            "generator": generator,
            "generator_family": family,
            "brand": args.brand,
            "template_id": codebase_id if args.group_by_codebase else dom_hash,
            "template_id_source": "codebase_id" if args.group_by_codebase else "dom_fingerprint",
            "codebase": codebase,
            "codebase_id": codebase_id,
            "prompt_id": prompt_id,
            "url": "",
            "collection_date": modified,
            "date_source": "file_mtime",
            "file_path": str(relative),
            "html_hash": html_hash,
            "dom_hash": dom_hash,
            "quality_level": "unknown",
            "manual_validated": 0,
            "published_label": 1,
            "label_provenance": "published_generation_task",
            "dataset_domain": "llm_generated",
            "dataset_version": "2026-paper-release",
            "dataset_provenance": "assessing_spearphishing",
            "generation_success": 1,
            "credential_collection": credential_collection,
            "educational_disclaimer": educational_disclaimer,
            "research_label": "candidate_requires_defensive_review",
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No HTML files found. Check repository path/layout.")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    codebase_summary = (
        frame.groupby(
            ["codebase_id", "codebase", "generator", "generator_family", "prompt_id"],
            dropna=False,
        )
        .agg(
            page_count=("sample_id", "size"),
            credential_page_count=("credential_collection", "sum"),
            page_positive_count=("page_label", "sum"),
            disclaimer_page_count=("educational_disclaimer", "sum"),
            unique_dom_count=("dom_hash", "nunique"),
        )
        .reset_index()
    )
    codebase_summary["generation_success"] = 1
    codebase_summary["dataset_provenance"] = "assessing_spearphishing"
    codebase_path = output.with_name(output.stem + "_codebases.csv")
    codebase_summary.to_csv(codebase_path, index=False)
    audit_columns = [
        "sample_id", "file_path", "generator", "generator_family", "brand",
        "codebase_id", "prompt_id", "quality_level", "manual_validated",
        "page_label", "site_label", "label_unit", "page_role",
        "credential_collection", "educational_disclaimer", "research_label",
        "html_hash", "dom_hash",
    ]
    audit_path = output.with_name(output.stem + "_audit.csv")
    frame[audit_columns].to_csv(audit_path, index=False)
    manifest = {
        "dataset": "assessing_spearphishing",
        "paper": "arXiv:2602.13363",
        "repository": str(root.resolve()),
        "pages": int(len(frame)),
        "codebases_observed": int(frame["codebase_id"].nunique()),
        "generator_families": frame["generator_family"].value_counts().to_dict(),
        "credential_pages": int(frame["credential_collection"].sum()),
        "page_positive": int(frame["page_label"].sum()),
        "label_unit": args.label_unit,
        "disclaimer_pages": int(frame["educational_disclaimer"].sum()),
        "output_sha256": sha256_file(output),
        "codebase_output_sha256": sha256_file(codebase_path),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(frame)} unique web templates from {frame['codebase_id'].nunique()} codebases to {output}")
    print(f"Codebase-level summary: {codebase_path}")
    print(f"Provenance manifest: {manifest_path}")
    print(f"Manual audit sheet: {audit_path}")
    print("\nGenerator-family counts:")
    print(frame["generator_family"].value_counts(dropna=False).to_string())
    print("\nIMPORTANT: set manual_validated=1 only after defensive data-quality review.")


if __name__ == "__main__":
    main()
