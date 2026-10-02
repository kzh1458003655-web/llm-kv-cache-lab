"""Compare output-token hashes across stock reruns and paired A/B runs."""
import argparse
import itertools
import json
from pathlib import Path


def read_run(root, index):
    directory = root / f'run-{index:02d}'
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    if run.get('status') != 'completed':
        raise ValueError(f'run-{index:02d} is not completed')
    request_path = directory / 'client/requests.jsonl'
    if not request_path.exists():
        request_path = directory / 'requests.jsonl'
    rows = [json.loads(line) for line in request_path.read_text(encoding='utf-8').splitlines() if line]
    mapped = {row['request_id']: row for row in rows}
    if not rows or len(mapped) != len(rows) or len(rows) != run.get('request_count'):
        raise ValueError(f'run-{index:02d} request records are incomplete or duplicated')
    if any(row.get('status') != 'ok' for row in rows):
        raise ValueError(f'run-{index:02d} contains failed requests')
    if any(not row.get('input_token_ids_sha256') or not row.get('output_token_ids_sha256')
           or row.get('output_tokens') is None for row in rows):
        raise ValueError(f'run-{index:02d} lacks required hashes or output token counts')
    return {'index': index, 'run': run, 'rows': mapped}


def same_configuration(left, right):
    for key in ('settings', 'model_revision', 'runtime_profile'):
        if left['run'].get(key) != right['run'].get(key):
            raise ValueError(f'configuration mismatch: {key}')


def same_requests(left, right):
    a, b = left['rows'], right['rows']
    if set(a) != set(b):
        raise ValueError('request IDs differ')
    for key in a:
        x, y = a[key], b[key]
        if (x.get('input_token_ids_sha256') != y.get('input_token_ids_sha256')
                or x.get('cache_salt_sha256') != y.get('cache_salt_sha256')
                or x.get('output_tokens') != y.get('output_tokens')):
            raise ValueError(f'input signature differs for request {key}')


def output_pair(left, right):
    mismatches = []
    for request_id in sorted(left['rows']):
        a, b = left['rows'][request_id], right['rows'][request_id]
        if a.get('output_token_ids_sha256') != b.get('output_token_ids_sha256'):
            mismatches.append({'request_id': request_id,
                'left_sha256': a.get('output_token_ids_sha256'),
                'right_sha256': b.get('output_token_ids_sha256')})
    return {'left_run': left['index'], 'right_run': right['index'],
            'request_count': len(left['rows']), 'mismatch_count': len(mismatches),
            'mismatch_rate': len(mismatches) / len(left['rows']) if left['rows'] else None,
            'mismatches': mismatches}


def cached_strata(stock, candidate):
    groups = {'same': [], 'different': [], 'unavailable': []}
    for request_id in sorted(stock['rows']):
        a = (stock['rows'][request_id].get('usage') or {}).get('prompt_tokens_details') or {}
        b = (candidate['rows'][request_id].get('usage') or {}).get('prompt_tokens_details') or {}
        ca, cb = a.get('cached_tokens'), b.get('cached_tokens')
        if ca is None or cb is None:
            group = 'unavailable'
        else:
            group = 'same' if ca == cb else 'different'
        mismatch = stock['rows'][request_id].get('output_token_ids_sha256') != candidate['rows'][request_id].get('output_token_ids_sha256')
        groups[group].append({'request_id': request_id, 'output_hash_mismatch': mismatch,
                              'stock_cached_tokens': ca, 'reuse2_cached_tokens': cb})
    return {name: {'request_count': len(rows),
                   'hash_mismatch_count': sum(row['output_hash_mismatch'] for row in rows),
                   'hash_mismatch_rate': (sum(row['output_hash_mismatch'] for row in rows) / len(rows)
                                          if rows else None),
                   'requests': rows} for name, rows in groups.items()}


def analyze(root):
    plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
    items = plan['items']
    indexed = {item['label']: [] for item in items}
    for index, item in enumerate(items):
        indexed[item['label']].append((index, item))
    baseline_entries = indexed.get('main-hot-0', [])
    baseline_indices = [i for i, item in baseline_entries if item['policy'] == 'stock']
    aa_labels = sorted((label for label in indexed if label.startswith('AA-stock-')),
                       key=lambda label: int(label.rsplit('-', 1)[1]))
    if len(baseline_indices) != 1 or len(aa_labels) != 3:
        raise ValueError('plan must contain one main-hot-0 stock run and three AA-stock runs')
    stock_runs = [read_run(root, baseline_indices[0])]
    for label in aa_labels:
        entries = indexed[label]
        if len(entries) != 1 or entries[0][1]['policy'] != 'stock':
            raise ValueError(f'invalid AA plan entry: {label}')
        stock_runs.append(read_run(root, entries[0][0]))
    for left, right in itertools.combinations(stock_runs, 2):
        same_configuration(left, right)
        same_requests(left, right)
    stock_pairs = [output_pair(left, right) for left, right in itertools.combinations(stock_runs, 2)]

    ab_pairs = []
    incomplete_ab_labels = []
    for label, entries in indexed.items():
        if label.startswith('AA-') or {item['policy'] for _, item in entries} != {'stock', 'reuse2'}:
            continue
        by_policy = {item['policy']: index for index, item in entries}
        paths = [root / f'run-{index:02d}/run.json' for index in by_policy.values()]
        if any(not path.exists() or json.loads(path.read_text(encoding='utf-8')).get('status') != 'completed'
               for path in paths):
            incomplete_ab_labels.append(label)
            continue
        stock = read_run(root, by_policy['stock'])
        candidate = read_run(root, by_policy['reuse2'])
        same_configuration(stock, candidate)
        same_requests(stock, candidate)
        ab_pairs.append({'label': label, 'stock_run': stock['index'],
                         'reuse2_run': candidate['index'],
                         'request_count': len(stock['rows']),
                         'cached_token_strata': cached_strata(stock, candidate)})
    return {'stock_reruns': {'run_indices': [run['index'] for run in stock_runs],
            'pairwise_output_hashes': stock_pairs,
            'interpretation': 'AA differences show rerun variation in stock; they do not rule out a candidate bug.'},
            'ab_cached_token_strata': ab_pairs,
            'incomplete_ab_labels': incomplete_ab_labels,
            'privacy_note': 'contains request IDs, hashes, and counts only; no prompt or token text'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.runs)
    target = args.runs / 'aa-output-hash-analysis.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(target), 'stock_pairs': len(result['stock_reruns']['pairwise_output_hashes']),
                      'ab_pairs': len(result['ab_cached_token_strata'])}))


if __name__ == '__main__':
    main()
