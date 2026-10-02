"""Summarize diagnostic cache removal/registration events without token overclaims."""
import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(root):
    root = Path(root)
    run = json.loads((root / 'run.json').read_text())
    if run['status'] != 'completed' or not run.get('trace_diagnostic'):
        raise ValueError('expected a completed serial event diagnostic')
    removed = set()
    repeated = Counter()
    allocations = Counter()
    measuring = False
    for line in (root / 'diagnostic-events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['type'] == 'lookup' and 'measure-' in event['request_id'] and 'warmup-' not in event['request_id']:
            measuring = True
        if not measuring:
            continue
        if event['type'] == 'cache_hash_removed' and event.get('during_allocation'):
            for entry in event['entries']:
                removed.add(entry['key'])
                allocations[str(entry['group_id'])] += 1
        if event['type'] == 'cache_registration_observed':
            for entry in event['entries']:
                if entry['key'] in removed:
                    repeated[str(entry['group_id'])] += 1
                    removed.remove(entry['key'])
    return {'settings': run['settings'], 'request_count': run['request_count'],
            'cached_tokens': run['client_summary']['cached_tokens'],
            'input_tokens': run['client_summary']['input_tokens'],
            'allocation_removed_hash_entries_by_group': dict(allocations),
            'removed_hash_registered_again_by_group': dict(repeated),
            'scope': 'observed cache-content churn; not exact recomputed token count or an offline optimality gap'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--moderate', required=True, type=Path)
    parser.add_argument('--roomy', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = {'moderate': summarize(args.moderate), 'roomy': summarize(args.roomy),
              'note': 'roomy is the largest tested feasible budget; inspect removals before calling it eviction-free'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    main()
