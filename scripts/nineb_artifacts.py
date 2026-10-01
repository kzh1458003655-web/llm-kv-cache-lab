"""Pinned public model verification/download and upload-bundle verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
from pathlib import Path

from prepare_9b_model_manifest import REPO_ID, REVISION, SMALL_FILES, _download_file


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value['repo_id'] != REPO_ID or value['resolved_revision'] != REVISION:
        raise ValueError('model manifest does not match the fixed 9B revision')
    for record in value['files']:
        name = Path(record['path'])
        if name.is_absolute() or '..' in name.parts or '\\' in record['path']:
            raise ValueError('unsafe model path')
        expected = f'https://huggingface.co/{REPO_ID}/resolve/{REVISION}/' + urllib.parse.quote(record['path'], safe='/') + '?download=true'
        if record['download_url'] != expected:
            raise ValueError('model URL is not pinned to the fixed public repository')
    return value


def verify_file(record, path):
    path = Path(path)
    size = path.stat().st_size
    if size != record['size_bytes']:
        raise ValueError(f"size mismatch: {record['path']}")
    digest = hashlib.sha256()
    blob = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            blob.update(chunk)
    expected = record.get('lfs_sha256')
    if expected:
        if digest.hexdigest() != expected:
            raise ValueError(f"LFS hash mismatch: {record['path']}")
    elif blob.hexdigest() != record['git_oid']:
        raise ValueError(f"Git blob hash mismatch: {record['path']}")
    return {'path': record['path'], 'size_bytes': size, 'sha256': digest.hexdigest()}


def model_files(manifest, metadata_only=False):
    names = SMALL_FILES if metadata_only else SMALL_FILES | set(manifest['weights']['files'])
    records = {record['path']: record for record in manifest['files']}
    return [records[name] for name in sorted(names)]


def verify_model(manifest_path, model_dir, metadata_only=False, cache_receipt=False):
    manifest = load_manifest(manifest_path)
    records = model_files(manifest, metadata_only)
    root = Path(model_dir)
    def fingerprint():
        values = {}
        for record in records:
            stat = (root / record['path']).stat()
            values[record['path']] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        return values
    receipt_path = root / '.nineb-integrity.json'
    current = fingerprint()
    manifest_hash = sha256(manifest_path)
    if cache_receipt and not metadata_only and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('manifest_sha256') == manifest_hash and receipt.get('fingerprints') == current:
            return {**receipt['verification'], 'verification_mode': 'reused full hash verification; file fingerprints unchanged'}
    checked = [verify_file(record, root / record['path']) for record in records]
    if fingerprint() != current:
        raise ValueError('model files changed while being verified')
    result = {'repo_id': REPO_ID, 'revision': REVISION,
            'metadata_only': metadata_only, 'verified_files': checked}
    if cache_receipt and not metadata_only:
        receipt_path.write_text(json.dumps({'manifest_sha256': manifest_hash,
                                           'fingerprints': current, 'verification': result}, indent=2) + '\n')
    return {**result, 'verification_mode': 'full Git blob/LFS hash verification'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for action in ('verify-model', 'download-model'):
        item = sub.add_parser(action)
        item.add_argument('--manifest', required=True, type=Path)
        item.add_argument('--model-dir', required=True, type=Path)
        item.add_argument('--metadata-only', action='store_true')
    item = sub.add_parser('verify-bundle')
    item.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'verify-bundle':
        root = args.root.resolve()
        records = json.loads((root / 'CHECKSUMS.json').read_text())
        for name, expected in records.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or sha256(path) != expected:
                raise ValueError(f'bundle verification failed: {name}')
        print(json.dumps({'verified_bundle_files': len(records)}))
        return
    manifest = load_manifest(args.manifest)
    if args.command == 'download-model':
        for record in model_files(manifest, args.metadata_only):
            target = args.model_dir / record['path']
            if target.exists():
                verify_file(record, target)
                print('verified existing:', record['path'], flush=True)
            else:
                _download_file(record, target)
                print('downloaded and verified:', record['path'], flush=True)
    print(json.dumps(verify_model(args.manifest, args.model_dir, args.metadata_only), indent=2))


if __name__ == '__main__':
    main()
