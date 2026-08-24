"""Extract Qwen HTML candidates from Innoc2Scam without executing generated code."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def extract_candidate(item, category, bucket):
    code = str(item.get("generated_code") or "")
    html_like = bool(re.search(r"<html\b|<form\b|<input\b", code, re.I))
    if not html_like:
        return None
    digest = hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()
    has_form = bool(re.search(r"<form\b", code, re.I))
    has_input = bool(re.search(r"<input\b", code, re.I))
    has_password = bool(
        re.search(r"type\s*=\s*['\"]?password|password", code, re.I)
    )
    return {
        "sample_id": f"innoc2scam_qwen_{digest[:16]}",
        "html_hash": digest,
        "generator": "qwen_qwen3-coder",
        "generator_family": "qwen",
        "prompt_id": str(item.get("prompt_index", "unknown")),
        "category": category,
        "source_bucket": bucket,
        "source_result_classification": str(item.get("result_classification", "")),
        "prompt": str(item.get("prompt") or ""),
        "html": code,
        "has_form": int(has_form),
        "has_input": int(has_input),
        "has_password": int(has_password),
        "credential_candidate": int(has_form and has_input and has_password),
        "manual_validated": 0,
        "reviewer_a_label": "",
        "reviewer_b_label": "",
        "review_reason": "",
        "label_provenance": "candidate_from_malicious_url_code_dataset",
        "dataset_provenance": "innoc2scam_bench_icml26",
        "dataset_version": "public_release_2026",
        "license": "MIT",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.repo) / "qwen_qwen3-coder"
    rows = []
    for path in sorted(root.rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        category = path.parent.name
        bucket = path.stem
        for item in document.get("prompts", []):
            candidate = extract_candidate(item, category, bucket)
            if candidate:
                rows.append(candidate)
    frame = pd.DataFrame(rows).drop_duplicates("html_hash")
    frame = frame.sort_values(
        ["credential_candidate", "has_password", "has_form", "has_input"],
        ascending=False,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(json.dumps({
        "rows": int(len(frame)),
        "credential_candidates": int(frame["credential_candidate"].sum()),
        "with_form": int(frame["has_form"].sum()),
        "with_input": int(frame["has_input"].sum()),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
