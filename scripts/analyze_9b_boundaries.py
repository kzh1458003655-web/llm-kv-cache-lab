"""Compare two serial boundary diagnostics, without making performance claims."""
import argparse
import json
import math
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stock', required=True, type=Path)
    parser.add_argument('--reuse2', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
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
    shared_lengths = {row['request_id']: row['shared_prefix_tokens'] for row in stock['boundary_inputs']}
    config_events = [event for event in runs['stock']['events'] if event['type'] == 'cache_config']
    full_sizes = [group['block_size'] for event in config_events for group in event['groups']
                  if group['spec'] == 'FullAttentionSpec']
    full_alignment = math.lcm(*full_sizes) if full_sizes else None
    for key, row in runs['stock']['rows'].items():
        other = runs['reuse2']['rows'][key]
        if row['input_token_ids_sha256'] != other['input_token_ids_sha256']:
            raise ValueError('paired diagnostic token inputs differ')
        cached = {policy: (run['rows'][key]['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
                  for policy, run in runs.items()}
        comparisons.append({'request_id': key,
            'input_tokens': row['expected_prompt_tokens'],
            'shared_prefix_tokens': shared_lengths[key],
            'full_attention_alignment_reference': shared_lengths[key] // full_alignment * full_alignment if full_alignment else None,
            'unreused_shared_prefix_tokens': {policy: shared_lengths[key] - value if value is not None else None
                                             for policy, value in cached.items()} if key.endswith('suffix102') else None,
            'output_hash_match': row['output_token_ids_sha256'] == other['output_token_ids_sha256'],
            'cached_tokens': cached})
    output = {'comparisons': comparisons,
        'native_groups': {policy: [event for event in run['events'] if event['type'] == 'cache_config']
                          for policy, run in runs.items()},
        'hash_removals_during_allocation': {policy: sum(event['type'] == 'cache_hash_removed'
                                                   and event.get('during_allocation', False) for event in run['events'])
                                  for policy, run in runs.items()},
        'output_hash_mismatches': sum(not row['output_hash_match'] for row in comparisons),
        'interpretation': 'each length has an isolated cache salt; full-attention alignment reference is block geometry, not a measured full-attention model or proof of a bug; no performance claim'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    return output


if __name__ == '__main__':
    main()
