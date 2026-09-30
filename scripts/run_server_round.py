"""Bounded paired runs. Stops owned children; does not stop rental billing."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--prepared', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--minutes', type=float, default=45)
    p.add_argument('--repeats', type=int, default=5)
    a = p.parse_args()
    if not 1 <= a.repeats <= 5 or not 1 <= a.minutes <= 60:
        p.error('repeats 1..5 and minutes 1..60 required')
    root = Path(__file__).resolve().parents[1]
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(root))
    env.pop('HYBRID_PILOT_EVENTS', None)
    started = time.monotonic()
    deadline = started + a.minutes * 60
    plan = []
    for rep in range(a.repeats):
        for scene, mib in [('hot_scan',128), ('mixed_zipf',256), ('roomy_control',4096)]:
            order = ['stock','reuse2'] if rep % 2 == 0 else ['reuse2','stock']
            for policy in order:
                plan.append({'rep':rep, 'scene':scene, 'kv_mib':mib, 'policy':policy,
                             'manifest': f'{scene}-{rep}.jsonl'})
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    outcomes = []
    try:
        for i, item in enumerate(plan):
            remaining = deadline-time.monotonic()
            # Do not begin a pair unless enough time remains for two capped runs.
            if i % 2 == 0 and remaining < 360:
                break
            name = f"{item['rep']}-{item['scene']}-{item['policy']}"
            cmd = [sys.executable, str(root/'scripts/server_benchmark.py'),
                   '--model',a.model,'--manifest',str(Path(a.prepared)/item['manifest']),
                   '--output',str(out/name),'--policy',item['policy'],'--kv-mib',str(item['kv_mib'])]
            print(f'Start {name}; remaining {remaining/60:.1f} minutes', flush=True)
            with (out/f'{name}.log').open('x') as log:
                process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=min(180, max(1, remaining)))
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    outcomes.append({**item,'name':name,'status':'interrupted_or_timeout'})
                    break
            outcomes.append({**item,'name':name,'exit_code':code})
            (out/'progress.json').write_text(json.dumps(outcomes,indent=2))
            if code != 0:
                break
    finally:
        (out/'progress.json').write_text(json.dumps(outcomes,indent=2))
        print('Saved results. Rental instance billing must be stopped separately.', flush=True)
        subprocess.run([sys.executable,str(root/'scripts/analyze_server_round.py'),'--input',str(out)],check=False)


if __name__ == '__main__':
    main()
