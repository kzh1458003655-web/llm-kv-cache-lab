"""Compare two serial boundary diagnostics, without making performance claims."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stock', required=True, type=Path)
    parser.add_argument('--reuse2', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    runs = {}
    for policy in ('stock', 'reuse2'):
        root = getattr(args, policy)
        value = json.loads((root / 'run.json').read_text())
        if value['status'] != 'completed' or not value['boundary_probe'] or value['policy'] != policy:
            raise ValueError('need completed paired serial boundary diagnostics')
        rows = {row['request_id']: row for row in
                [json.loads(line) for line in (root / 'client/requests.jsonl').read_text().splitlines()]}
        events = [json.loads(line) for line in (root / 'diagnostic-events.jsonl').read_text().splitlines()]
        runs[policy] = {'rows': rows, 'events': events, 'run': value}
    stock = runs['stock']['run']; candidate = runs['reuse2']['run']
    for key in ('settings', 'model_revision', 'runtime_profile'):
        if stock[key] != candidate[key]:
            raise ValueError(f'diagnostic configuration mismatch: {key}')
    if set(runs['stock']['rows']) != set(runs['reuse2']['rows']):
        raise ValueError('diagnostic IDs differ')
    comparisons = []
    for key, row in runs['stock']['rows'].items():
        other = runs['reuse2']['rows'][key]
        if row['input_token_ids_sha256'] != other['input_token_ids_sha256']:
            raise ValueError('paired diagnostic token inputs differ')
        comparisons.append({'request_id': key,
            'input_tokens': row['expected_prompt_tokens'],
            'output_hash_match': row['output_token_ids_sha256'] == other['output_token_ids_sha256'],
            'cached_tokens': {policy: (run['rows'][key]['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
                              for policy, run in runs.items()}})
    output = {'comparisons': comparisons,
        'native_groups': {policy: [event for event in run['events'] if event['type'] == 'cache_config']
                          for policy, run in runs.items()},
        'hash_removals_during_allocation': {policy: sum(event['type'] == 'cache_hash_removed'
                                                   and event.get('during_allocation', False) for event in run['events'])
                                  for policy, run in runs.items()},
        'interpretation': 'boundary behavior only; use diagnostic events to separate alignment from eviction; no performance claim'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
