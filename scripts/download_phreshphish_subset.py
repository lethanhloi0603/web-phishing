import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import sha256_file


REPOSITORY = "phreshphish/phreshphish"
LICENSE = "CC-BY-4.0"
CITATION = "https://arxiv.org/abs/2507.10854"


def merge_manifest_entries(existing, entry):
    files = list(existing.get("files", [])) if isinstance(existing, dict) else []
    if isinstance(existing, dict) and not files and existing.get("local_file"):
        files.append({
            key: existing[key]
            for key in (
                "repository_path", "download_url", "local_file", "bytes", "sha256"
            )
            if key in existing
        })
    files = [item for item in files if item.get("local_file") != entry["local_file"]]
    files.append(entry)
    return sorted(files, key=lambda item: item["local_file"])


def download_with_resume(url, destination, min_free_bytes, chunk_bytes=1024 * 1024):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "phishgen-detector-research"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            offset = 0
            temporary.unlink(missing_ok=True)
        content_length = int(response.headers.get("Content-Length", 0))
        expected_remaining = content_length
        free = shutil.disk_usage(destination.parent).free
        if free - expected_remaining < min_free_bytes:
            raise RuntimeError(
                "Refusing download: declared remaining bytes would leave less than "
                f"{min_free_bytes / 2**30:.2f} GB free"
            )
        mode = "ab" if offset and status == 206 else "wb"
        with temporary.open(mode) as handle:
            next_progress = handle.tell() + 64 * 2**20
            while True:
                chunk = response.read(chunk_bytes)
                if not chunk:
                    break
                handle.write(chunk)
                if handle.tell() >= next_progress:
                    print(
                        f"Downloaded {handle.tell() / 2**20:.1f} MB -> {destination.name}",
                        flush=True,
                    )
                    next_progress += 64 * 2**20
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(
        description="Download one small, licensed PhreshPhish Parquet shard."
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-free-gb", type=float, default=6.0)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    if args.shard < 0 or args.min_free_gb < 0 or args.retries < 1:
        raise ValueError("--shard/min-free must be non-negative and retries positive")

    filename = f"{args.split}-{args.shard:03d}.parquet"
    repository_path = f"data/{filename}"
    url = (
        f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
        f"{args.revision}/{repository_path}?download=true"
    )
    output_dir = Path(args.output_dir).resolve()
    destination = output_dir / filename
    if not destination.exists():
        for attempt in range(1, args.retries + 1):
            try:
                download_with_resume(
                    url, destination, int(args.min_free_gb * 2**30)
                )
                break
            except (HTTPError, URLError, TimeoutError) as error:
                if attempt == args.retries:
                    raise
                delay = min(30, 2 ** attempt)
                print(
                    f"Download attempt {attempt}/{args.retries} failed: {error}; "
                    f"retrying in {delay}s",
                    flush=True,
                )
                time.sleep(delay)

    entry = {
        "repository_path": repository_path,
        "download_url": url,
        "local_file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }
    manifest_path = output_dir / "download_manifest.json"
    existing = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY,
        "revision": args.revision,
        "license": LICENSE,
        "anti_phishing_research_only": True,
        "citation": CITATION,
        "files": merge_manifest_entries(existing, entry),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Downloaded/verified: {destination}")
    print(f"SHA-256: {entry['sha256']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except (HTTPError, URLError) as error:
        raise SystemExit(f"Network error: {error}") from error
