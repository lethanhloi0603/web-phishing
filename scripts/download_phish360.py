import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import gdown


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import sha256_file


FILES = {
    "phish": {
        "id": "1RVVwm0dwBvKzTenV4nnMbBEp60IYeMUO",
        "name": "Phish360_phish.parquet",
        "bytes": 137_932_113,
    },
    "legit": {
        "id": "1nsiCIAZoBOSs9bEEXq78Nveu6Q90SLzr",
        "name": "Phish360_legit.parquet",
        "bytes": 500_062_998,
    },
}
CITATION = "https://doi.org/10.3390/app16020751"
DATASET_PAGE = "https://web.cs.hacettepe.edu.tr/~selman/phish360-dataset/"


def selected_files(selection):
    return FILES if selection == "both" else {selection: FILES[selection]}


def check_storage(output_dir, required_bytes, min_free_bytes):
    free = shutil.disk_usage(output_dir).free
    if free - required_bytes < min_free_bytes:
        raise RuntimeError(
            f"Refusing download: {required_bytes / 2**20:.1f} MB remaining would "
            f"leave less than {min_free_bytes / 2**30:.2f} GB free"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download the compact public Phish360 HTML/URL Parquet files."
    )
    parser.add_argument("--class", dest="selection", choices=["both", "phish", "legit"], default="both")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-free-gb", type=float, default=6.0)
    args = parser.parse_args()
    if args.min_free_gb < 0:
        raise ValueError("--min-free-gb must be non-negative")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = selected_files(args.selection)
    remaining = sum(
        max(0, item["bytes"] - (output_dir / item["name"]).stat().st_size)
        if (output_dir / item["name"]).exists()
        else item["bytes"]
        for item in selection.values()
    )
    check_storage(output_dir, remaining, int(args.min_free_gb * 2**30))

    manifest_files = []
    for label, item in selection.items():
        destination = output_dir / item["name"]
        if not destination.exists() or destination.stat().st_size != item["bytes"]:
            gdown.download(
                id=item["id"], output=str(destination), quiet=False, resume=True
            )
        if not destination.exists() or destination.stat().st_size != item["bytes"]:
            actual = destination.stat().st_size if destination.exists() else 0
            raise RuntimeError(
                f"Size mismatch for {item['name']}: {actual} != {item['bytes']}"
            )
        manifest_files.append({
            "label": label,
            "google_drive_file_id": item["id"],
            "local_file": item["name"],
            "bytes": item["bytes"],
            "sha256": sha256_file(destination),
        })

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Phish360",
        "dataset_page": DATASET_PAGE,
        "citation": CITATION,
        "terms": "academic-use (as stated by the paper; do not redistribute)",
        "license_note": (
            "No standalone SPDX dataset license was found. The paper explicitly "
            "states that the dataset is available for academic use."
        ),
        "files": manifest_files,
    }
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
