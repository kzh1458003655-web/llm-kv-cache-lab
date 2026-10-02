"""Create an allowlisted private upload kit; weights and credentials excluded."""
import argparse
import json
import re
import subprocess
import tarfile
from pathlib import Path

from nineb_artifacts import sha256, verify_model
from prepare_9b_model_manifest import SMALL_FILES

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'configs/thesis-9b-protocol.json',
    'configs/nineb-opening-plan.json',
    'kv_cache_lab/__init__.py', 'kv_cache_lab/events.py', 'kv_cache_lab/hybrid_retention_prototype.py',
    'kv_cache_lab/hybrid_observer.py',
    'scripts/prepare_9b_model_manifest.py', 'scripts/nineb_artifacts.py',
    'scripts/server_workloads.py', 'scripts/prepare_9b_workloads.py',
    'scripts/nineb_online_client.py', 'scripts/run_9b_online.py',
    'scripts/run_9b_round.py', 'scripts/freeze_9b_calibration.py',
    'scripts/analyze_9b_boundaries.py',
    'scripts/analyze_9b_traces.py',
    'scripts/export_9b_results.py', 'scripts/prepare_9b_bundle.py',
    'tests/test_nineb_opening.py',
    'server-kit/nineb-bootstrap/sitecustomize.py',
    'server-kit/nineb/setup.sh', 'server-kit/nineb/README.md',
    'server-kit/nineb/calibration-settings.template.json',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'manifest', 'prepared', 'audit', 'source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--base-commit', help='Verified Git SHA from the host when bundling a Windows worktree in WSL')
    args = parser.parse_args()
    if args.base_commit and not re.fullmatch(r'[0-9a-f]{40}', args.base_commit):
        raise ValueError('--base-commit must be an exact 40-character Git SHA')
    verify_model(args.manifest, args.model, metadata_only=True)
    index = json.loads((args.prepared / 'index.json').read_text())
    if index['protocol_sha256'] != sha256(ROOT / 'configs/thesis-9b-protocol.json'):
        raise ValueError('request preparation protocol differs from the bundled protocol')
    if len(index['files']) != 20:
        raise ValueError('expected the complete 20-file fixed workload preparation')
    protocol = json.loads((ROOT / 'configs/thesis-9b-protocol.json').read_text())
    if sha256(args.source) != protocol['source']['sha256']:
        raise ValueError('source differs from the complete pinned source')
    args.output.mkdir(parents=True, exist_ok=False)

    def copy(source, relative):
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Path(source).open('rb') as input_stream, destination.open('xb') as output_stream:
            for chunk in iter(lambda: input_stream.read(1024 * 1024), b''):
                output_stream.write(chunk)

    for name in FILES:
        copy(ROOT / name, name)
    copy(args.manifest, 'model-manifest.json')
    copy(args.source, 'source/shortdep_qa.jsonl')
    for name in sorted(SMALL_FILES):
        copy(args.model / name, 'model/' + name)
    for entry in index['files']:
        name = entry['name'] + '.jsonl'
        if sha256(args.prepared / name) != entry['sha256']:
            raise ValueError('private request file hash mismatch')
        copy(args.prepared / name, 'prepared/' + name)
        copy(args.audit / (entry['name'] + '.json'), 'audits/' + entry['name'] + '.json')
    copy(args.prepared / 'index.json', 'prepared/index.json')
    copy(args.audit / 'index.json', 'audits/index.json')
    base_commit = args.base_commit
    diff = None
    try:
        if base_commit is None:
            base_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pass
    version = {'base_commit': base_commit,
               'working_tree_diff_sha256': None,
               'note': 'exact bundled source is identified by CHECKSUMS.json; base commit alone is not the artifact identity'}
    import hashlib
    version['working_tree_diff_sha256'] = hashlib.sha256(diff).hexdigest() if diff is not None else None
    (args.output / 'CODE_VERSION.json').write_text(json.dumps(version, indent=2) + '\n')
    checksums = {str(path.relative_to(args.output)).replace('\\', '/'): sha256(path)
                 for path in sorted(args.output.rglob('*')) if path.is_file()}
    (args.output / 'CHECKSUMS.json').write_text(json.dumps(checksums, indent=2) + '\n')
    archive = args.output.with_suffix('.tar.gz')
    with archive.open('xb') as output_stream:
        with tarfile.open(fileobj=output_stream, mode='w:gz') as packed:
            packed.add(args.output, arcname=args.output.name)
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': sha256(archive), 'verified_file_count': len(checksums),
                      'weights_included': False, 'gpu_run': False}, indent=2))


if __name__ == '__main__':
    main()
