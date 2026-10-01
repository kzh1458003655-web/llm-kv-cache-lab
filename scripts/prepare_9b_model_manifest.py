#!/usr/bin/env python3
"""Create a pinned Qwen3.5-9B file manifest and optionally download files.

All downloaded files are fetched from Hugging Face's public HTTPS endpoints.
No authentication token is read or sent by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REPO_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
API_ROOT = "https://huggingface.co/api/models"
HUB_ROOT = "https://huggingface.co"
USER_AGENT = "llm-kv-cache-public-pilot/9b-manifest"
SMALL_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "model.safetensors.index.json",
}
WEIGHT_RE = re.compile(
    r"(?:^|/)(?:model\.safetensors|model\.safetensors-\d{5}-of-\d{5}\.safetensors|model-\d{5}-of-\d{5}\.safetensors)$"
)


class PreparationError(RuntimeError):
    """Raised when a remote artifact cannot be safely prepared."""


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _fetch_json(url: str) -> tuple[Any, str | None]:
    with urllib.request.urlopen(_request(url), timeout=60) as response:
        payload = json.load(response)
        return payload, response.headers.get("Link")


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for item in link_header.split(","):
        match = re.match(r'\s*<([^>]+)>\s*;\s*rel="?next"?', item)
        if match:
            return match.group(1)
    return None


def fetch_manifest() -> dict[str, Any]:
    """Read immutable revision metadata and the expanded repository tree."""
    rev_url = f"{API_ROOT}/{REPO_ID}/revision/{REVISION}"
    revision_info, _ = _fetch_json(rev_url)
    resolved_revision = revision_info.get("sha")
    if resolved_revision != REVISION:
        raise PreparationError(
            f"Hub resolved revision {resolved_revision!r}, expected {REVISION}"
        )

    tree_url = (
        f"{API_ROOT}/{REPO_ID}/tree/{REVISION}"
        "?recursive=true&expand=true"
    )
    files: list[dict[str, Any]] = []
    visited: set[str] = set()
    while tree_url:
        if tree_url in visited:
            raise PreparationError("Hugging Face API pagination loop detected")
        visited.add(tree_url)
        page, link_header = _fetch_json(tree_url)
        if not isinstance(page, list):
            raise PreparationError("Unexpected Hugging Face tree API response")
        for entry in page:
            if entry.get("type") != "file":
                continue
            path = entry.get("path")
            if not isinstance(path, str) or path.startswith("/") or ".." in Path(path).parts:
                raise PreparationError(f"Unsafe repository path in API response: {path!r}")
            lfs = entry.get("lfs") or {}
            lfs_oid = lfs.get("oid")
            files.append(
                {
                    "path": path,
                    "size_bytes": entry.get("size"),
                    "git_oid": entry.get("oid"),
                    "lfs_sha256": lfs_oid if isinstance(lfs_oid, str) and len(lfs_oid) == 64 else None,
                    "download_url": (
                        f"{HUB_ROOT}/{REPO_ID}/resolve/{REVISION}/"
                        + urllib.parse.quote(path, safe="/")
                        + "?download=true"
                    ),
                }
            )
        tree_url = _next_link(link_header)

    paths = [item["path"] for item in files]
    if len(paths) != len(set(paths)):
        raise PreparationError("Duplicate file paths in Hugging Face tree response")
    files.sort(key=lambda item: item["path"])
    total_bytes = sum(item["size_bytes"] or 0 for item in files)
    weight_files = [item for item in files if WEIGHT_RE.search(item["path"])]
    if not weight_files:
        raise PreparationError("No safetensors weight shards found at pinned revision")

    return {
        "schema_version": 1,
        "source": "Hugging Face Hub public API",
        "repo_id": REPO_ID,
        "requested_revision": REVISION,
        "resolved_revision": resolved_revision,
        "files": files,
        "file_count": len(files),
        "total_repo_file_bytes": total_bytes,
        "weights": {
            "file_count": len(weight_files),
            "total_bytes": sum(item["size_bytes"] or 0 for item in weight_files),
            "files": [item["path"] for item in weight_files],
        },
    }


def _git_blob_oid(path: Path, size: int) -> str:
    """Compute Git's SHA1 object ID for a regular blob with the given size."""
    digest = hashlib.sha1()
    digest.update(b"blob " + str(size).encode("ascii") + b"\0")
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(record: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Download without overwriting; verify LFS SHA256 or the Git blob OID."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    received = 0
    created = False
    try:
        with urllib.request.urlopen(_request(record["download_url"]), timeout=120) as response:
            with destination.open("xb") as output:
                created = True
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
        expected_size = record.get("size_bytes")
        if expected_size is not None and received != expected_size:
            raise PreparationError(
                f"Size mismatch for {record['path']}: expected {expected_size}, got {received}"
            )
        actual_sha256 = digest.hexdigest()
        expected_sha256 = record.get("lfs_sha256")
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise PreparationError(
                f"LFS SHA256 mismatch for {record['path']}: expected {expected_sha256}, got {actual_sha256}"
            )
        actual_git_oid = None
        expected_git_oid = record.get("git_oid")
        if not expected_sha256:
            if not expected_git_oid:
                raise PreparationError(f"Missing expected Git blob OID for {record['path']}")
            actual_git_oid = _git_blob_oid(destination, received)
            if actual_git_oid != expected_git_oid:
                raise PreparationError(
                    f"Git blob OID mismatch for {record['path']}: expected {expected_git_oid}, got {actual_git_oid}"
                )
        return {
            "path": record["path"],
            "size_bytes": received,
            "sha256": actual_sha256,
            "lfs_sha256_verified": bool(expected_sha256),
            "git_blob_oid": actual_git_oid,
            "git_oid_verified": bool(not expected_sha256 and expected_git_oid),
            "size_verified": expected_size is None or received == expected_size,
            "status": "downloaded_verified",
        }
    except FileExistsError as exc:
        raise PreparationError(f"Refusing to overwrite existing file: {destination}") from exc
    except Exception:
        if created:
            try:
                destination.unlink()
            except OSError:
                pass
        raise


def _write_json_new(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise PreparationError(f"Refusing to overwrite existing file: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Explicit directory for the generated manifest.json",
    )
    parser.add_argument(
        "--metadata-dir",
        required=True,
        type=Path,
        help="Explicit directory for small tokenizer/config files downloaded by default",
    )
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Also download every weight shard; requires --weights-dir",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        help="Explicit directory for optional large weight shard downloads",
    )
    args = parser.parse_args(argv)
    if args.download_weights and args.weights_dir is None:
        parser.error("--download-weights requires an explicit --weights-dir")
    if not args.download_weights and args.weights_dir is not None:
        parser.error("--weights-dir is only valid with --download-weights")

    try:
        manifest = fetch_manifest()
        manifest_path = args.output_dir / "manifest.json"
        _write_json_new(manifest_path, manifest)

        file_map = {item["path"]: item for item in manifest["files"]}
        missing_small = sorted(SMALL_FILES - file_map.keys())
        if missing_small:
            raise PreparationError(
                "Pinned revision is missing expected tokenizer/config files: "
                + ", ".join(missing_small)
            )

        downloaded: list[dict[str, Any]] = []
        for name in sorted(SMALL_FILES):
            downloaded.append(
                _download_file(file_map[name], args.metadata_dir / name)
            )

        if args.download_weights:
            for name in manifest["weights"]["files"]:
                downloaded.append(
                    _download_file(file_map[name], args.weights_dir / name)
                )

        result = {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "manifest_path": str(manifest_path),
            "metadata_download_dir": str(args.metadata_dir),
            "weights_downloaded": bool(args.download_weights),
            "downloads": downloaded,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (urllib.error.URLError, TimeoutError, OSError, PreparationError, json.JSONDecodeError) as exc:
        print(f"prepare_9b_model_manifest: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
