import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import source_snapshot, write_json


def main():
    parser = argparse.ArgumentParser(
        description="Write a deterministic source-tree hash when Git metadata is unavailable."
    )
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = source_snapshot(args.root)
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    write_json(args.out, report)
    print(f"Source snapshot: {args.out}")
    print(f"Files: {report['file_count']}; SHA-256: {report['sha256']}")


if __name__ == "__main__":
    main()
