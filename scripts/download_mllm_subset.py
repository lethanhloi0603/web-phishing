import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import sha256_file


DEFAULT_REPOSITORY = "JehLeeKR/Multimodal_LLM_Phishing_Detection"
CLASS_PREFIXES = {
    "benign": "data/MMLLM_Benign/",
    "phishing": "data/MMLLM_Phishing/",
}


def api_json(url, token=""):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "phishgen-detector-research",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def stable_candidates(entries, prefix, seed, max_archive_bytes):
    candidates = [
        entry for entry in entries
        if entry.get("type") == "blob"
        and str(entry.get("path", "")).startswith(prefix)
        and str(entry.get("path", "")).lower().endswith(".zip")
        and 0 < int(entry.get("size", 0)) <= max_archive_bytes
    ]
    return sorted(
        candidates,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item['path']}".encode("utf-8")
        ).hexdigest(),
    )


def select_bounded(entries, prefix, seed, count, max_archive_bytes, budget_bytes):
    selected = []
    used = 0
    for entry in stable_candidates(entries, prefix, seed, max_archive_bytes):
        size = int(entry["size"])
        if used + size > budget_bytes:
            continue
        selected.append(entry)
        used += size
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(
            f"Only {len(selected)}/{count} archives fit for {prefix!r}; "
            "raise --max-archive-mb/--max-total-mb or lower --n-per-class"
        )
    return selected


def safe_destination(output_dir, label, repository_path):
    prefix = PurePosixPath(CLASS_PREFIXES[label])
    relative = PurePosixPath(repository_path).relative_to(prefix)
    destination = output_dir / label / Path(*relative.parts)
    root = output_dir.resolve()
    resolved = destination.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Unsafe repository path: {repository_path}")
    return destination


def download_file(url, destination, expected_size, min_free_bytes, overwrite=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if destination.stat().st_size != expected_size:
            raise RuntimeError(
                f"Existing file has unexpected size: {destination}; use --overwrite"
            )
        return False
    free_bytes = shutil.disk_usage(destination.parent).free
    if free_bytes - expected_size < min_free_bytes:
        raise RuntimeError(
            f"Refusing download: it would leave less than "
            f"{min_free_bytes / 2**30:.2f} GB free"
        )
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = Request(url, headers={"User-Agent": "phishgen-detector-research"})
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                if handle.tell() > expected_size:
                    raise RuntimeError(f"Remote file exceeded declared size: {url}")
        if temporary.stat().st_size != expected_size:
            raise RuntimeError(
                f"Incomplete download for {url}: got {temporary.stat().st_size}, "
                f"expected {expected_size}"
            )
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download a deterministic, storage-bounded subset of the eCrime 2024 "
            "MMLLM benign/phishing ZIP dataset directly from a pinned Git commit."
        )
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--ref", default="main")
    parser.add_argument("--n-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-archive-mb", type=float, default=8.0)
    parser.add_argument("--max-total-mb", type=float, default=1024.0)
    parser.add_argument("--min-free-gb", type=float, default=6.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--github-token", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--acknowledge-license-unknown",
        action="store_true",
        help="Required because the upstream repository declares no SPDX license",
    )
    args = parser.parse_args()

    if args.n_per_class < 1:
        raise ValueError("--n-per-class must be positive")
    if args.max_archive_mb <= 0 or args.max_total_mb <= 0 or args.min_free_gb < 0:
        raise ValueError("Storage limits must be positive (min-free may be zero)")

    api_root = f"https://api.github.com/repos/{args.repository}"
    repository = api_json(api_root, args.github_token)
    commit = api_json(f"{api_root}/commits/{quote(args.ref, safe='')}", args.github_token)
    commit_sha = commit["sha"]
    tree = api_json(f"{api_root}/git/trees/{commit_sha}?recursive=1", args.github_token)
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned a truncated tree; refusing an irreproducible sample")

    license_info = repository.get("license") or {}
    license_spdx = license_info.get("spdx_id") or "NOASSERTION"
    if license_spdx in {"NOASSERTION", "NOASSERTION", "NONE"}:
        if not args.acknowledge_license_unknown:
            raise RuntimeError(
                "Upstream declares no license. Re-run with "
                "--acknowledge-license-unknown for local research only; do not redistribute."
            )

    total_budget = int(args.max_total_mb * 2**20)
    class_budget = total_budget // len(CLASS_PREFIXES)
    max_archive_bytes = int(args.max_archive_mb * 2**20)
    selected = {
        label: select_bounded(
            tree["tree"], prefix, args.seed, args.n_per_class,
            max_archive_bytes, class_budget,
        )
        for label, prefix in CLASS_PREFIXES.items()
    }
    declared_total = sum(
        int(entry["size"])
        for entries in selected.values()
        for entry in entries
    )
    if declared_total > total_budget:
        raise RuntimeError("Internal selection error: declared total exceeds budget")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for label, entries in selected.items():
        for index, entry in enumerate(entries, start=1):
            repository_path = entry["path"]
            destination = safe_destination(output_dir, label, repository_path)
            raw_url = (
                f"https://raw.githubusercontent.com/{args.repository}/{commit_sha}/"
                f"{quote(repository_path, safe='/')}"
            )
            downloaded = download_file(
                raw_url,
                destination,
                int(entry["size"]),
                int(args.min_free_gb * 2**30),
                overwrite=args.overwrite,
            )
            manifest_rows.append({
                "label": label,
                "repository_path": repository_path,
                "git_blob_sha": entry["sha"],
                "declared_bytes": int(entry["size"]),
                "local_path": str(destination.relative_to(output_dir)),
                "sha256": sha256_file(destination),
            })
            print(
                f"[{label} {index}/{len(entries)}] "
                f"{'downloaded' if downloaded else 'verified existing'}: {repository_path}",
                flush=True,
            )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_repository": f"https://github.com/{args.repository}",
        "paper": "https://ieeexplore.ieee.org/document/10778904",
        "repository": args.repository,
        "requested_ref": args.ref,
        "commit_sha": commit_sha,
        "license_spdx": license_spdx,
        "license_warning": (
            "Upstream repository declares no license; local research use only and "
            "do not redistribute archives without author permission."
            if license_spdx in {"NOASSERTION", "NONE"} else ""
        ),
        "selection": "sha256(seed:path) ascending, class-balanced, size-bounded",
        "seed": args.seed,
        "n_per_class": args.n_per_class,
        "max_archive_mb": args.max_archive_mb,
        "max_total_mb": args.max_total_mb,
        "min_free_gb": args.min_free_gb,
        "selected_declared_bytes": declared_total,
        "files": manifest_rows,
    }
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Downloaded/verified {len(manifest_rows)} archives ({declared_total / 2**20:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except (HTTPError, URLError) as error:
        raise SystemExit(f"Network error: {error}") from error
