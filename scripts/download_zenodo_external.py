import argparse
from datetime import datetime, timezone
import hashlib
import http.client
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


DEFAULT_RECORD = "8041387"
DEFAULT_FILES = (
    "phishing_5001-5151.zip",
    "not-phishing_5001-5244.zip",
)


def json_request(url):
    request = Request(url, headers={"User-Agent": "phishgen-detector-research"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def label_directory(filename):
    lowered = filename.lower()
    if lowered.startswith("not-phishing_"):
        return "not-phishing"
    if lowered.startswith("phishing_"):
        return "phishing"
    raise ValueError(f"Cannot infer class from Zenodo filename: {filename}")


def file_download_url(file_info):
    links = file_info.get("links") or {}
    for key in ("content", "download", "self"):
        value = links.get(key)
        if value:
            return value
    raise ValueError(f"Zenodo file has no download link: {file_info.get('key', '?')}")


def checksum_file(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download(file_info, destination, min_free_bytes, overwrite=False, max_retries=6):
    expected_size = int(file_info["size"])
    checksum_spec = str(file_info["checksum"])
    algorithm, expected_checksum = checksum_spec.split(":", 1)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        if destination.stat().st_size != expected_size:
            raise RuntimeError(
                f"Existing file has wrong size: {destination}; use --overwrite"
            )
        if checksum_file(destination, algorithm) != expected_checksum:
            raise RuntimeError(
                f"Existing file has wrong {algorithm}: {destination}; use --overwrite"
            )
        return False

    free_bytes = shutil.disk_usage(destination.parent).free
    if free_bytes - expected_size < min_free_bytes:
        raise RuntimeError(
            f"Refusing {destination.name}: download would leave less than "
            f"{min_free_bytes / 2**30:.2f} GB free"
        )

    temporary = destination.with_suffix(destination.suffix + ".part")
    if overwrite and temporary.exists():
        temporary.unlink()
    for attempt in range(max_retries + 1):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > expected_size:
            raise RuntimeError(f"Partial file exceeds declared size: {temporary}")
        if offset == expected_size:
            break
        headers = {"User-Agent": "phishgen-detector-research"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            print(
                f"Resuming {destination.name} at {offset / 2**20:.0f} MB",
                flush=True,
            )
        request = Request(file_download_url(file_info), headers=headers)
        try:
            with urlopen(request, timeout=60) as response:
                status = getattr(response, "status", response.getcode())
                append = bool(offset and status == 206)
                if offset and not append:
                    print(
                        f"Server ignored Range for {destination.name}; restarting file",
                        flush=True,
                    )
                    offset = 0
                mode = "ab" if append else "wb"
                with temporary.open(mode) as handle:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        if handle.tell() > expected_size:
                            raise RuntimeError(
                                f"Remote file exceeded declared size: {destination.name}"
                            )
                        if handle.tell() % (128 * 1024 * 1024) < len(chunk):
                            print(
                                f"{destination.name}: {handle.tell() / 2**20:.0f}/"
                                f"{expected_size / 2**20:.0f} MB",
                                flush=True,
                            )
        except (
            HTTPError, URLError, TimeoutError, OSError,
            http.client.HTTPException,
        ) as error:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Download failed after {max_retries + 1} attempts; "
                    f"partial file retained at {temporary}"
                ) from error
            print(
                f"Retry {attempt + 1}/{max_retries} after network error: "
                f"{type(error).__name__}",
                flush=True,
            )
            time.sleep(min(5, attempt + 1))
            continue
    if temporary.stat().st_size != expected_size:
        raise RuntimeError(
            f"Incomplete {destination.name}: got {temporary.stat().st_size}, "
            f"expected {expected_size} bytes; partial retained"
        )
    if checksum_file(temporary, algorithm) != expected_checksum:
        raise RuntimeError(
            f"Checksum mismatch for {destination.name}; partial retained for diagnosis"
        )
    temporary.replace(destination)
    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download selected raw-webpage archives from a pinned Zenodo record "
            "with disk and checksum safeguards."
        )
    )
    parser.add_argument("--record", default=DEFAULT_RECORD)
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-free-gb", type=float, default=6.0)
    parser.add_argument("--max-total-gb", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--acknowledge-license-unknown",
        action="store_true",
        help="Required when Zenodo does not declare a license; do not redistribute",
    )
    args = parser.parse_args()

    if args.min_free_gb < 0 or args.max_total_gb <= 0:
        raise ValueError("Invalid disk limits")
    record = json_request(f"https://zenodo.org/api/records/{args.record}")
    by_name = {item["key"]: item for item in record["files"]}
    missing = sorted(set(args.files) - set(by_name))
    if missing:
        raise ValueError(f"Files absent from Zenodo record: {missing}")

    rights = record.get("metadata", {}).get("rights") or {}
    license_id = rights.get("id") or "NOASSERTION"
    if license_id == "NOASSERTION" and not args.acknowledge_license_unknown:
        raise RuntimeError(
            "Zenodo record declares no license. Re-run with "
            "--acknowledge-license-unknown for local research only."
        )

    selected = [by_name[name] for name in args.files]
    total_bytes = sum(int(item["size"]) for item in selected)
    if total_bytes > int(args.max_total_gb * 2**30):
        raise RuntimeError(
            f"Selected files require {total_bytes / 2**30:.2f} GB, above "
            f"--max-total-gb={args.max_total_gb}"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for item in selected:
        destination = output_dir / label_directory(item["key"]) / item["key"]
        changed = download(
            item,
            destination,
            int(args.min_free_gb * 2**30),
            overwrite=args.overwrite,
        )
        print(
            f"{'Downloaded' if changed else 'Verified existing'}: {destination}",
            flush=True,
        )
        completed.append({
            "name": item["key"],
            "class_directory": destination.parent.name,
            "bytes": int(item["size"]),
            "zenodo_checksum": item["checksum"],
            "sha256": sha256_file(destination),
            "local_path": str(destination.relative_to(output_dir)),
        })

    metadata = record.get("metadata", {})
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": f"https://zenodo.org/records/{args.record}",
        "record": args.record,
        "doi": metadata.get("doi", f"10.5281/zenodo.{args.record}"),
        "title": metadata.get("title", ""),
        "publication_date": metadata.get("publication_date", ""),
        "license": license_id,
        "license_warning": (
            "No license is declared by the record; use locally for research and "
            "do not redistribute without permission."
            if license_id == "NOASSERTION" else ""
        ),
        "selected_bytes": total_bytes,
        "min_free_gb": args.min_free_gb,
        "max_total_gb": args.max_total_gb,
        "files": completed,
    }
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except (HTTPError, URLError) as error:
        raise SystemExit(f"Network error: {error}") from error
