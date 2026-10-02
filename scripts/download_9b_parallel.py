"""Download and verify a pinned 9B model with bounded parallel transfers."""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nineb_artifacts import load_manifest, model_files, verify_file, verify_model
from prepare_9b_model_manifest import HUB_ROOT, REPO_ID, REVISION, _download_file

MODELSCOPE_ROOT = "https://modelscope.cn"
ALLOWED_ENDPOINTS = (HUB_ROOT, "https://hf-mirror.com", MODELSCOPE_ROOT)
OFFICIAL_PREFIX = f"{HUB_ROOT}/{REPO_ID}/resolve/{REVISION}/"
PRINT_LOCK = threading.Lock()


def actual_url(original_url: str, endpoint: str) -> str:
    if not original_url.startswith(OFFICIAL_PREFIX):
        raise ValueError("manifest URL is not the pinned official model URL")
    if endpoint == MODELSCOPE_ROOT:
        path = original_url[len(OFFICIAL_PREFIX):].split("?", 1)[0]
        return f"{MODELSCOPE_ROOT}/models/{REPO_ID}/resolve/master/{path}"
    return endpoint + original_url[len(HUB_ROOT):]


def download_one(record: dict, target: Path, original_url: str, url: str) -> dict:
    with PRINT_LOCK:
        print(f"START {record['path']}", flush=True)
    try:
        adjusted = {**record, "download_url": url}
        result = _download_file(adjusted, target)
        result.update({"original_download_url": original_url,
                       "actual_download_url": url,
                       "already_present": False,
                       "verified": True})
        with PRINT_LOCK:
            print(f"DONE  {record['path']}", flush=True)
        return result
    except Exception as exc:
        with PRINT_LOCK:
            print(f"FAIL  {record['path']}: {type(exc).__name__}", flush=True)
        return {"path": record["path"], "original_download_url": original_url,
                "actual_download_url": url, "already_present": False,
                "verified": False, "status": "failed",
                "error_type": type(exc).__name__, "error": str(exc)}


def write_receipt(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--endpoint", choices=ALLOWED_ENDPOINTS, default=HUB_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    args.model_dir = args.model_dir.resolve()
    args.receipt = args.receipt.resolve()
    integrity_receipt = (args.model_dir / ".nineb-integrity.json").resolve()
    if args.receipt == integrity_receipt:
        parser.error("--receipt must not overwrite the model integrity receipt")
    if args.receipt.exists():
        parser.error(f"receipt already exists; refusing to overwrite: {args.receipt}")

    receipt = {"status": "running", "manifest": str(args.manifest.resolve()),
               "model_dir": str(args.model_dir), "endpoint": args.endpoint,
               "workers": args.workers, "files": []}
    if args.endpoint == MODELSCOPE_ROOT:
        receipt["endpoint_note"] = (
            "ModelScope resolve/master is not pinned to the manifest revision. "
            "Downloaded files are accepted only after matching the original "
            "fixed-revision manifest size and SHA256/Git OID."
        )
    try:
        # load_manifest enforces the pinned repository, commit, safe paths, and
        # exact official resolve URLs. The source manifest is never rewritten.
        manifest = load_manifest(args.manifest)
        records = model_files(manifest)

        # Audit every existing file before starting any network transfer.
        missing = []
        for record in records:
            original = record["download_url"]
            mirror_url = actual_url(original, args.endpoint)
            target = args.model_dir / record["path"]
            if target.exists():
                try:
                    check = verify_file(record, target)
                    receipt["files"].append({**check,
                        "original_download_url": original,
                        "actual_download_url": mirror_url,
                        "already_present": True, "verified": True,
                        "status": "verified_existing"})
                    print(f"EXISTS {record['path']} verified", flush=True)
                except Exception as exc:
                    receipt["files"].append({"path": record["path"],
                        "original_download_url": original,
                        "actual_download_url": mirror_url,
                        "already_present": True, "verified": False,
                        "status": "failed_existing",
                        "error_type": type(exc).__name__, "error": str(exc)})
                    raise RuntimeError(f"existing model file failed verification: {record['path']}") from exc
            else:
                missing.append((record, target, original, mirror_url))

        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for record, target, original, url in missing:
                future = pool.submit(download_one, record, target, original, url)
                futures[future] = (record, original, url)
            stop_submitting = False
            for future in as_completed(futures):
                try:
                    result = future.result()
                except BaseException as exc:
                    record, original, url = futures[future]
                    result = {"path": record["path"],
                              "original_download_url": original,
                              "actual_download_url": url,
                              "verified": False, "already_present": False,
                              "status": "cancelled" if future.cancelled() else "failed",
                              "error_type": type(exc).__name__, "error": str(exc)}
                receipt["files"].append(result)
                if not result.get("verified") and not stop_submitting:
                    stop_submitting = True
                    for pending in futures:
                        if pending is not future:
                            pending.cancel()

        failures = [item for item in receipt["files"] if not item.get("verified")]
        if failures:
            raise RuntimeError(f"{len(failures)} model file(s) failed; see receipt")

        verification = verify_model(args.manifest, args.model_dir, cache_receipt=True)
        receipt.update({"status": "verified", "verification": verification,
                       "verified_file_count": len(receipt["files"])})
    except BaseException as exc:
        receipt.update({"status": "failed", "error_type": type(exc).__name__,
                       "error": str(exc)})
        try:
            write_receipt(args.receipt, receipt)
        except FileExistsError:
            print(f"Cannot save receipt because it already exists: {args.receipt}", file=sys.stderr)
        raise

    write_receipt(args.receipt, receipt)
    print(json.dumps({"status": receipt["status"],
                      "verified_file_count": receipt["verified_file_count"],
                      "receipt": str(args.receipt)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
