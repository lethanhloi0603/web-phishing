import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import write_json
from src.shortcut_audit import shortcut_report


def resolve_split(root, name):
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = manifest.get("files", {}).get(name)
        if declared and (root / declared).exists():
            return root / declared
    for suffix in ("csv.gz", "csv"):
        path = root / f"{name}.{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing split: {name}")


def main():
    parser = argparse.ArgumentParser(description="Detect label shortcuts by source, brand and generator")
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--max-group-purity", type=float, default=0.95)
    parser.add_argument("--require-source-label-2x2", action="store_true")
    parser.add_argument("--fail-on-high-purity", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.scenario_dir)
    reports = {}
    failures = []
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        frame = pd.read_csv(resolve_split(root, split))
        report = shortcut_report(frame, args.min_group_size, args.max_group_purity)
        reports[split] = report
        if args.require_source_label_2x2 and report["missing_source_label_cells"]:
            failures.append(f"{split}: missing {report['missing_source_label_cells']}")
        if args.fail_on_high_purity and report["high_purity_groups"]:
            failures.append(f"{split}: high-purity label shortcuts in {sorted(report['high_purity_groups'])}")

    output = {
        "passed": not failures,
        "scenario_dir": str(root.resolve()),
        "splits": reports,
        "failures": failures,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if args.out:
        write_json(args.out, output)
    raise SystemExit(0 if output["passed"] else 2)


if __name__ == "__main__":
    main()
