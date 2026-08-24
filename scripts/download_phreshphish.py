import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from urllib.parse import urlsplit
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import dom_fingerprint, html_sha256
from src.research import sha256_file


PINNED_REVISION = "dd3e82ce0bc80d30715e8bde350895e780a5416c"
LICENSE = "CC-BY-4.0"
CITATION = "https://arxiv.org/abs/2507.10854"


def clean_metadata(value, default="unknown"):
    if value is None:
        return default
    if isinstance(value, (list, tuple, set)):
        value = ";".join(str(item) for item in value if item is not None)
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "null"} else default


def site_identifier(url):
    value = str(url or "").strip()
    parsed = urlsplit(value if "://" in value else f"//{value}")
    hostname = (parsed.hostname or "").strip(".").lower()
    if not hostname:
        return ""
    return hashlib.sha256(f"phreshphish:host:{hostname}".encode("utf-8")).hexdigest()[:24]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--n-per-class", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-html-chars", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--revision", default=PINNED_REVISION)
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument(
        "--shuffle-buffer", type=int, default=10000,
        help=(
            "Deterministically shuffle this many streaming rows before sampling. "
            "Use 0 only to reproduce the old first-N behavior."
        ),
    )
    args = ap.parse_args()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output.parent).free / 2**30
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.2f} GB free; refusing download below "
            f"--min-free-gb {args.min_free_gb:.2f}"
        )

    ds = load_dataset(
        "phreshphish/phreshphish",
        split=args.split,
        revision=args.revision,
        streaming=True,
    )
    if args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    rows = []
    counts = Counter()
    target = args.n_per_class

    for r in ds:
        label_str = str(r.get("label", "")).lower()
        if label_str not in {"benign", "phish"}:
            continue
        if counts[label_str] >= target:
            continue

        raw_html = r.get("html")
        if not isinstance(raw_html, str) or len(raw_html.strip()) < 20:
            continue

        clipped = raw_html[: args.max_html_chars]
        html_hash = html_sha256(clipped)
        dom_hash = dom_fingerprint(clipped)
        brand = clean_metadata(r.get("target")) if label_str == "phish" else "unknown"
        url = r.get("url", "")
        rows.append({
            "sample_id": html_hash[:16],
            "html": clipped,
            "label": 1 if label_str == "phish" else 0,
            "source": "conventional",
            "generator": "human_kit",
            "generator_family": "human",
            "brand": brand,
            "template_id": dom_hash,
            "url": url,
            "collection_date": clean_metadata(r.get("date"), default=""),
            "origin_split": args.split,
            "html_hash": html_hash,
            "dom_hash": dom_hash,
            "dataset_domain": "real_world",
            "dataset_provenance": "phreshphish",
            "dataset_version": "huggingface_streaming",
            "label_provenance": "published_dataset",
            "dataset_license": LICENSE,
            "dataset_citation": CITATION,
            "published_label": 1,
            "page_label": 1 if label_str == "phish" else 0,
            "site_label": 1 if label_str == "phish" else 0,
            "label_unit": "page",
            "page_role": "standalone",
            "codebase_id": site_identifier(url) or html_hash[:24],
            "language": clean_metadata(r.get("lang", r.get("language"))),
            "target": clean_metadata(r.get("target")),
            "source_sha256": clean_metadata(r.get("sha256")),
        })
        counts[label_str] += 1

        if counts["benign"] >= target and counts["phish"] >= target:
            break

    if not rows:
        raise RuntimeError("No rows collected. Check internet access and dataset schema.")

    pd.DataFrame(rows).to_csv(output, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "phreshphish/phreshphish",
        "citation": "https://arxiv.org/abs/2507.10854",
        "license": LICENSE,
        "revision": args.revision,
        "split": args.split,
        "sampling": "streaming_buffer_shuffle" if args.shuffle_buffer > 0 else "streaming_first_n",
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "n_per_class_requested": args.n_per_class,
        "counts": dict(counts),
        "known_phishing_brands": len({
            row["brand"] for row in rows
            if row["label"] == 1 and row["brand"] != "unknown"
        }),
        "max_html_chars": args.max_html_chars,
        "min_free_gb": args.min_free_gb,
        "output": output.name,
        "output_sha256": sha256_file(output),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Saved:", args.out)
    print("Counts:", dict(counts))

if __name__ == "__main__":
    main()
