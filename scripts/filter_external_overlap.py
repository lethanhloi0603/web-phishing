import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import read_frame
from src.research import evaluation_data_evidence, sha256_file


IDENTITY_COLUMNS = ("html_hash", "template_id", "codebase_id")


def filter_overlap(external, references, drop_template_overlap=True):
    blocked = {column: set() for column in IDENTITY_COLUMNS}
    for reference in references:
        for column in IDENTITY_COLUMNS:
            if column in reference:
                blocked[column].update(
                    reference[column].fillna("").astype(str).str.strip()
                )
                blocked[column].discard("")
    masks = {}
    for column in IDENTITY_COLUMNS:
        if column not in external:
            masks[column] = pd.Series(False, index=external.index)
        else:
            masks[column] = external[column].fillna("").astype(str).isin(blocked[column])
    removal = masks["html_hash"] | masks["codebase_id"]
    if drop_template_overlap:
        removal |= masks["template_id"]
    report = {
        "input_rows": int(len(external)),
        "exact_html_overlap_rows": int(masks["html_hash"].sum()),
        "template_overlap_rows": int(masks["template_id"].sum()),
        "site_overlap_rows": int(masks["codebase_id"].sum()),
        "rows_removed": int(removal.sum()),
        "output_rows": int((~removal).sum()),
        "drop_template_overlap": bool(drop_template_overlap),
    }
    return external.loc[~removal].reset_index(drop=True), report


def main():
    parser = argparse.ArgumentParser(
        description="Remove primary-data identity/template overlaps from an external test."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--drop-template-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    external = read_frame(args.input)
    references = [read_frame(path) for path in args.reference]
    clean, overlap = filter_overlap(
        external, references, drop_template_overlap=args.drop_template_overlap
    )
    if clean.empty:
        raise RuntimeError("No external rows remain after overlap filtering")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(args.input).resolve()),
        "input_sha256": sha256_file(args.input),
        "references": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for path in args.reference
        ],
        "overlap": overlap,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "output_evidence": evaluation_data_evidence(clean, output),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(overlap, indent=2))
    print(f"Strict external test: {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
