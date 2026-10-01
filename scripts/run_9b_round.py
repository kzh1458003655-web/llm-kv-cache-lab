"""Selective online A/A and A/B plan; plan-only by default, bounded execution."""
import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

from nineb_artifacts import sha256
from run_9b_online import stop_owned
from nineb_online_client import percentile


def plan():
    items = []

    def pair(scene, context, rep, budget, concurrency, label):
        policies = ('stock', 'reuse2') if rep % 2 == 0 else ('reuse2', 'stock')
        for policy in policies:
            items.append({'label': label, 'scene': scene, 'context': context, 'rep': rep,
                          'budget': budget, 'concurrency': concurrency, 'policy': policy})

    # Cover the two workloads and the roomy control before adding repeats.
    pair('hot_scan', 4096, 0, 'moderate_pressure', 8, 'main-hot-0')
    pair('mixed_zipf', 4096, 0, 'moderate_pressure', 8, 'mixed')
    pair('mixed_zipf', 4096, 0, 'roomy', 8, 'mixed-roomy')
    for rep in range(3):
        items.append({'label': f'AA-stock-{rep}', 'scene': 'hot_scan', 'context': 4096,
                      'rep': 0, 'budget': 'moderate_pressure', 'concurrency': 8, 'policy': 'stock'})
    for rep in range(1, 5):
        pair('hot_scan', 4096, rep, 'moderate_pressure', 8, f'main-hot-{rep}')
    pair('hot_scan', 4096, 0, 'high_pressure', 8, 'hot-high-pressure')
    pair('hot_scan', 4096, 0, 'roomy', 8, 'hot-roomy')
    pair('hot_scan', 4096, 0, 'moderate_pressure', 16, 'hot-concurrency16')
    pair('hot_scan', 2048, 0, 'moderate_pressure', 8, 'hot-input2048')
    return items


def analyze(output, items):
    pairs = []
    aa = []
    for index, item in enumerate(items):
        path = output / f'run-{index:02d}/run.json'
        if item['label'].startswith('AA-') and path.exists():
            record = json.loads(path.read_text())
            if record.get('status') == 'completed':
                aa.append({'label': item['label'], 'summary': record['client_summary']})
    for label in dict.fromkeys(item['label'] for item in items if not item['label'].startswith('AA-')):
        indices = [index for index, item in enumerate(items) if item['label'] == label]
        runs = {}
        for index in indices:
            path = output / f'run-{index:02d}'
            if (path / 'run.json').exists():
                record = json.loads((path / 'run.json').read_text())
                if record.get('status') == 'completed':
                    raw = [json.loads(line) for line in (path / 'client/requests.jsonl').read_text().splitlines()]
                    rows = {row['request_id']: row for row in raw}
                    if len(rows) != len(raw) or len(rows) != record['request_count']:
                        raise ValueError('duplicate IDs or incomplete completed run')
                    runs[items[index]['policy']] = (record, rows)
        if set(runs) != {'stock', 'reuse2'}:
            pairs.append({'label': label, 'status': 'incomplete'})
            continue
        stock, stock_rows = runs['stock']
        candidate, candidate_rows = runs['reuse2']
        if set(stock_rows) != set(candidate_rows):
            raise ValueError('paired request IDs differ')
        if any(stock_rows[key]['input_token_ids_sha256'] != candidate_rows[key]['input_token_ids_sha256']
               for key in stock_rows):
            raise ValueError('paired token inputs differ')
        mismatches = [key for key in stock_rows
                      if stock_rows[key]['output_token_ids_sha256'] != candidate_rows[key]['output_token_ids_sha256']]
        summary = {'label': label, 'status': 'paired', 'count': len(stock_rows),
                   'output_token_hash_mismatches': len(mismatches),
                   'mismatch_request_ids': mismatches,
                   'cached_tokens': {key: value[0]['client_summary']['cached_tokens'] for key, value in runs.items()},
                   'latency_comparison_note': 'independent online batches may choose numerically different greedy tokens; investigate mismatches before correctness claims'}
        for name in ('ttft_ms', 'tpot_ms', 'arrival_to_first_token_ms'):
            summary[name] = {policy: record['client_summary'][name] for policy, (record, _) in runs.items()}
        groups = {}
        for group in ('hot', 'scan_cold'):
            groups[group] = {}
            for policy, (_, rows) in runs.items():
                selected = [row for key, row in rows.items()
                            if ('-scan' in key) == (group == 'scan_cold') and key.startswith('segment')]
                groups[group][policy] = {'count': len(selected),
                    'ttft_ms': {f'p{int(p*100)}': percentile([row['ttft_ms'] for row in selected], p)
                                for p in (.5, .95, .99)}}
        summary['hot_scan_groups'] = groups
        pairs.append(summary)
    aa_range = {}
    for name in ('ttft_ms', 'tpot_ms'):
        values = [run['summary'][name]['p95'] for run in aa]
        aa_range[name] = {'p95_values': values,
                         'relative_range': (max(values) - min(values)) / min(values)
                                           if values and min(values) > 0 else None}
    main = [pair for pair in pairs if pair['label'].startswith('main-hot-') and pair['status'] == 'paired']
    independent = {}
    for name in ('ttft_ms', 'tpot_ms'):
        values = [(pair[name]['reuse2']['p95'] / pair[name]['stock']['p95'] - 1) * 100
                  for pair in main if pair[name]['stock']['p95'] and pair[name]['reuse2']['p95'] is not None]
        rng = random.Random(20261002)
        means = [sum(rng.choices(values, k=len(values))) / len(values) for _ in range(10000)] if len(values) >= 5 else []
        independent[name] = {'paired_p95_percent_changes': values,
            'mean_percent_change': sum(values) / len(values) if values else None,
            'bootstrap_mean_95_percent_interval': [percentile(means, .025), percentile(means, .975)] if means else None}
    (output / 'analysis.json').write_text(json.dumps({'pairs': pairs, 'AA_runs': aa,
        'AA_range': aa_range, 'representative_independent_pairs': independent,
        'statistics_note': 'bootstrap resamples independent paired runs, not individual correlated requests; n=5 remains limited',
        'AA_note': 'three runs estimate observed variation, not a reliable population confidence interval'}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'manifest', 'protocol', 'prepared', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--minutes', type=float, default=45)
    args = parser.parse_args()
    if args.minutes <= 0:
        parser.error('--minutes must be positive')
    if args.execute and not args.lock:
        parser.error('--execute requires a frozen --lock')
    items = plan()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    saved = {'status': 'planned', 'execute': args.execute, 'items': items,
             'protocol_sha256': sha256(args.protocol), 'deadline_minutes': args.minutes,
             'total_runs': len(items), 'shutdown_instance': False,
             'interpretation': 'partial coverage is saved; 45 minutes does not guarantee all runs complete'}
    (output / 'plan.json').write_text(json.dumps(saved, indent=2) + '\n')
    if not args.execute:
        print(json.dumps({'total_runs': len(items), 'plan_only': True, 'gpu_started': False}))
        return
    if os.name != 'posix':
        raise ValueError('execution requires Linux')
    deadline = time.monotonic() + args.minutes * 60
    progress = []
    try:
        for index, item in enumerate(items):
            remaining = deadline - time.monotonic()
            if remaining < 60:
                break
            requests = args.prepared / f"{item['scene']}-ctx{item['context']}-rep{item['rep']}.jsonl"
            command = [sys.executable, str(Path(__file__).with_name('run_9b_online.py')),
                '--model', str(args.model), '--manifest', str(args.manifest),
                '--protocol', str(args.protocol), '--requests', str(requests),
                '--output', str(output / f'run-{index:02d}'), '--lock', str(args.lock),
                '--policy', item['policy'], '--budget', item['budget'],
                '--concurrency', str(item['concurrency'])]
            process = subprocess.Popen(command, start_new_session=True)
            try:
                code = process.wait(timeout=max(1, remaining - 30))
            except BaseException:
                # TERM goes to the runner; its finally block stops its server.
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    stop_owned(process)
                raise
            progress.append({'index': index, 'exit_code': code})
            (output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
            analyze(output, items)
            if code:
                break
    finally:
        analyze(output, items)
        (output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')


if __name__ == '__main__':
    main()
