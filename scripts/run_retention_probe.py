"""Eight serial diagnostic runs: four interventions, two independent starts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from nineb_artifacts import sha256
from run_9b_online import profile
from run_9b_round import run_bounded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--minutes', type=float, default=100)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.output.mkdir(parents=True, exist_ok=False)
    lock = json.loads(Path('calibration-lock.json').read_text())
    protocol_path = Path('configs/thesis-9b-protocol.json')
    requests = Path('prepared/hot_scan-ctx4096-rep0.jsonl')
    assert sha256(protocol_path) == lock['protocol_sha256']
    assert sha256(requests) == lock['request_files'][requests.name]
    runtime = profile(json.loads(protocol_path.read_text()))
    # A restart may change the vGPU UUID, not the measured device properties.
    old = lock['runtime_profile']
    assert runtime['versions'] == old['versions']
    assert {k:v for k,v in runtime['gpu'].items() if k != 'uuid'} == {k:v for k,v in old['gpu'].items() if k != 'uuid'}
    assert lock['cache_budgets']['moderate_pressure'] == 3 * 1024**3
    assert not lock['enforce_eager']
    native = args.output / 'native-options.json'
    native.write_text(json.dumps(lock['native_cache_options']))
    plan = [{'rep': r, 'mode': m} for r, modes in enumerate(
        [('stock', 'all', 'attention', 'state'), ('state', 'attention', 'all', 'stock')]) for m in modes]
    (args.output / 'plan.json').write_text(json.dumps({'runs': plan, 'request_count': 64,
        'concurrency': 1, 'cache_bytes': 3*1024**3, 'source': 'segment002-scan2',
        'target': 'segment007-scan0', 'prefix_end': 3696,
        'interpretation': 'Future-informed diagnostic; no performance claim or online policy.'}, indent=2))
    deadline = time.monotonic() + args.minutes * 60
    progress = []
    for i, item in enumerate(plan):
        os.environ['NINEB_RETENTION_PROBE'] = item['mode']
        out = args.output / f"run-{i:02d}-{item['mode']}"
        cmd = [sys.executable, 'scripts/run_9b_online.py', '--mode', 'diagnostic',
            '--trace-diagnostic', '--policy', 'stock', '--model', 'model',
            '--manifest', 'model-manifest.json', '--protocol', str(protocol_path),
            '--requests', str(requests), '--output', str(out), '--count', '64',
            '--concurrency', '1', '--rate', str(lock['request_rate']),
            '--kv-cache-bytes', str(3*1024**3), '--native-options', str(native)]
        code = run_bounded(cmd, deadline)
        progress.append({'run': out.name, 'exit_code': code})
        (args.output / 'progress.json').write_text(json.dumps(progress, indent=2))
        if code:
            raise RuntimeError(f'diagnostic failed: {out.name}')
    print('All eight diagnostic runs completed.', flush=True)


if __name__ == '__main__':
    main()
