"""Selective online A/A and A/B plan; plan-only by default, bounded execution."""
import argparse
import json
import os
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

    def pair(scene, context, rep, budget, concurrency, label, variant='shared'):
        policies = ('stock', 'reuse2') if rep % 2 == 0 else ('reuse2', 'stock')
        for policy in policies:
            items.append({'label': label, 'scene': scene, 'context': context, 'rep': rep,
                          'budget': budget, 'concurrency': concurrency, 'policy': policy,
                          'workload_variant': variant})

    # Opening evidence: cover both workloads, a no-reuse control, and AA first.
    pair('hot_scan', 4096, 0, 'moderate_pressure', 8, 'main-hot-0')
    pair('mixed_zipf', 4096, 0, 'moderate_pressure', 8, 'main-mixed-0')
    pair('hot_scan', 4096, 0, 'moderate_pressure', 8, 'no-reuse', 'no_reuse')
    for rep in range(3):
        items.append({'label': f'AA-stock-{rep}', 'scene': 'hot_scan', 'context': 4096,
                      'rep': 0, 'budget': 'moderate_pressure', 'concurrency': 8, 'policy': 'stock',
                      'workload_variant': 'shared'})
    for rep in range(1, 3):
        pair('hot_scan', 4096, rep, 'moderate_pressure', 8, f'main-hot-{rep}')
        pair('mixed_zipf', 4096, rep, 'moderate_pressure', 8, f'main-mixed-{rep}')
    pair('hot_scan', 4096, 0, 'high_pressure', 8, 'hot-high-pressure')
    pair('hot_scan', 4096, 0, 'roomy', 8, 'hot-roomy')
    pair('hot_scan', 4096, 0, 'moderate_pressure', 16, 'hot-concurrency16')
    pair('hot_scan', 2048, 0, 'moderate_pressure', 8, 'hot-input2048')
    return items


def diagnostics_plan():
    return [{'name': 'boundary-' + policy, 'policy': policy, 'budget': 'roomy', 'boundary': True}
            for policy in ('stock', 'reuse2')] + [
           {'name': 'pressure-' + budget, 'policy': 'stock', 'budget': budget, 'boundary': False}
            for budget in ('moderate_pressure', 'roomy')]


def run_bounded(command, deadline):
    remaining = deadline - time.monotonic()
    if remaining < 45:
        raise TimeoutError('insufficient remaining time for a new run and cleanup')
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=max(1, remaining - 30))
    except BaseException:
        try:
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            stop_owned(process)
        raise


def estimate_from_calibration(root, run_count):
    records = [json.loads(path.read_text()) for path in root.rglob('run.json')]
    complete = [row for row in records if row.get('mode') == 'calibration'
                and row.get('status') == 'completed' and row.get('request_count', 0) >= 32
                and row.get('startup_elapsed_s') is not None]
    if not complete:
        return {'status': 'awaiting_measured_GPU_calibration', 'guarantees_one_hour': False}
    startup = max(row['startup_elapsed_s'] for row in complete)
    per_request = max(row['client_summary']['duration_s'] / row['request_count'] for row in complete)
    lower = run_count * (startup + per_request * 512) + 4 * (startup + per_request * 64)
    return {'status': 'rough_extrapolation', 'baseline_minutes': lower / 60,
            'planning_minutes_with_50_percent_margin': lower * 1.5 / 60,
            'excludes': ['installation', 'weight transfer', 'result download'],
            'note': 'startup and worst observed per-request calibration cost; longer traces and lower concurrency may take more time'}


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
               or stock_rows[key].get('cache_salt_sha256') != candidate_rows[key].get('cache_salt_sha256')
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
        for group in ('hot', 'scan'):
            groups[group] = {}
            for policy, (_, rows) in runs.items():
                selected = [row for key, row in rows.items()
                            if ('-scan' in key) == (group == 'scan') and key.startswith('segment')]
                groups[group][policy] = {'count': len(selected),
                    'ttft_ms': {f'p{int(p*100)}': percentile([row['ttft_ms'] for row in selected], p)
                                for p in (.5, .95, .99)}}
        summary['hot_scan_groups'] = groups
        summary['scan_note'] = 'scan means its role within a segment; documents can recur across segments'
        pairs.append(summary)
    aa_range = {}
    for name in ('ttft_ms', 'tpot_ms'):
        values = [run['summary'][name]['p95'] for run in aa]
        aa_range[name] = {'p95_values': values,
                         'relative_range': (max(values) - min(values)) / min(values)
                                           if values and min(values) > 0 else None}
    independent = {}
    for scene in ('hot', 'mixed'):
        main = [pair for pair in pairs if pair['label'].startswith('main-' + scene + '-') and pair['status'] == 'paired']
        independent[scene] = {}
        for name in ('ttft_ms', 'tpot_ms'):
            values = [(pair[name]['reuse2']['p95'] / pair[name]['stock']['p95'] - 1) * 100
                      for pair in main if pair[name]['stock']['p95'] and pair[name]['reuse2']['p95'] is not None]
            independent[scene][name] = {'paired_p95_percent_changes': values,
                'median_percent_change': percentile(values, .5),
                'range_percent_change': [min(values), max(values)] if values else None}
    (output / 'analysis.json').write_text(json.dumps({'pairs': pairs, 'AA_runs': aa,
        'AA_range': aa_range, 'representative_independent_pairs': independent,
        'statistics_note': 'opening plan uses three paired seeds per workload; report all signs and ranges, not a population confidence claim',
        'AA_note': 'three runs estimate observed variation, not a reliable population confidence interval'}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'manifest', 'protocol', 'prepared', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--minutes', type=float)
    parser.add_argument('--estimate-from', type=Path)
    args = parser.parse_args()
    if args.minutes is not None and args.minutes <= 0:
        parser.error('--minutes must be positive')
    if args.execute and not args.lock:
        parser.error('--execute requires a frozen --lock')
    if args.execute and args.minutes is None:
        parser.error('--execute requires an explicit --minutes budget after calibration')
    items = plan()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    saved = {'status': 'planned', 'execute': args.execute, 'items': items,
             'protocol_sha256': sha256(args.protocol), 'deadline_minutes': args.minutes,
             'opening_plan_sha256': sha256(Path(__file__).resolve().parents[1] / 'configs/nineb-opening-plan.json'),
             'short_diagnostics': diagnostics_plan(),
             'total_runs': len(items), 'shutdown_instance': False,
             'interpretation': '25 formal runs plus four short diagnostic starts; partial coverage is saved; duration depends on GPU calibration'}
    if args.estimate_from:
        saved['runtime_estimate'] = estimate_from_calibration(args.estimate_from, len(items))
    (output / 'plan.json').write_text(json.dumps(saved, indent=2) + '\n')
    if not args.execute:
        print(json.dumps({'total_runs': len(items), 'short_diagnostics': 4, 'plan_only': True, 'gpu_started': False,
                          'runtime_estimate': saved.get('runtime_estimate')}))
        return
    if os.name != 'posix':
        raise ValueError('execution requires Linux')
    deadline = time.monotonic() + args.minutes * 60
    progress = []
    common = [sys.executable, str(Path(__file__).with_name('run_9b_online.py')),
              '--model', str(args.model), '--manifest', str(args.manifest),
              '--protocol', str(args.protocol), '--lock', str(args.lock)]
    try:
        for item in diagnostics_plan():
            destination = output / 'diagnostics' / item['name']
            command = common + ['--mode', 'diagnostic', '--policy', item['policy'],
                '--budget', item['budget'], '--concurrency', '1', '--count', '64',
                '--requests', str(args.prepared / 'hot_scan-ctx4096-rep0.jsonl'),
                '--output', str(destination), '--boundary-probe' if item['boundary'] else '--trace-diagnostic']
            code = run_bounded(command, deadline)
            progress.append({'diagnostic': item['name'], 'exit_code': code})
            (output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
            if code:
                raise RuntimeError('short diagnostic failed; inspect saved results before formal measurements')
        from analyze_9b_boundaries import main as analyze_boundaries
        boundary = analyze_boundaries(['--stock', str(output / 'diagnostics/boundary-stock'),
            '--reuse2', str(output / 'diagnostics/boundary-reuse2'),
            '--output', str(output / 'diagnostics/boundary-analysis.json')])
        if boundary['output_hash_mismatches']:
            raise RuntimeError('serial output hashes differ; diagnose before latency experiments')
        from analyze_9b_traces import main as analyze_traces
        analyze_traces(['--moderate', str(output / 'diagnostics/pressure-moderate_pressure'),
            '--roomy', str(output / 'diagnostics/pressure-roomy'),
            '--output', str(output / 'diagnostics/pressure-analysis.json')])
        for index, item in enumerate(items):
            remaining = deadline - time.monotonic()
            if remaining < 60:
                break
            requests = args.prepared / f"{item['scene']}-ctx{item['context']}-rep{item['rep']}.jsonl"
            command = common + ['--requests', str(requests),
                '--output', str(output / f'run-{index:02d}'),
                '--policy', item['policy'], '--budget', item['budget'],
                '--workload-variant', item['workload_variant'],
                '--concurrency', str(item['concurrency'])]
            code = run_bounded(command, deadline)
            progress.append({'index': index, 'exit_code': code})
            (output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
            analyze(output, items)
            if item['workload_variant'] == 'no_reuse':
                result_path = output / f'run-{index:02d}/run.json'
                if result_path.exists():
                    record = json.loads(result_path.read_text())
                    if record.get('status') == 'completed' and record['client_summary']['cached_tokens'] != 0:
                        raise RuntimeError('no-reuse control did not report zero cached tokens')
            if code:
                break
    finally:
        analyze(output, items)
        (output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')


if __name__ == '__main__':
    main()
