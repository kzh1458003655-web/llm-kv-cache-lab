"""Summarize interventions without treating misses as executed recomputation."""
import argparse
from collections import Counter
import json
from pathlib import Path


def analyze(root):
    results = []
    baseline = None
    for folder in sorted(root.glob('run-*')):
        record = json.loads((folder / 'run.json').read_text())
        if record['status'] != 'completed':
            raise ValueError(f'incomplete: {folder.name}')
        path = folder / 'client/requests.jsonl'
        if not path.exists():
            path = folder / 'requests.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        by_id = {r['request_id']: r for r in rows}
        assert len(rows) == len(by_id) == 64 and all(r['status'] == 'ok' for r in rows)
        if baseline is None:
            baseline = by_id
        assert by_id.keys() == baseline.keys()
        assert all(r['input_token_ids_sha256'] == baseline[k]['input_token_ids_sha256'] for k, r in by_id.items())
        events = [json.loads(line) for line in (folder / 'diagnostic-events.jsonl').read_text().splitlines()]
        geometry = [e for e in events if e['type'] == 'cache_config']
        assert geometry and all(e['num_pool_blocks'] == 186 for e in geometry)
        probes = [e for e in events if e['type'] == 'retention_probe']
        assert probes and any(e['action'] == 'installed' for e in probes)
        captures = {entry['key']: entry['group_id'] for e in probes if e['action'] == 'capture' for entry in e['entries']}
        mode = probes[0]['mode']
        expected = {'stock': 0, 'all': 10, 'attention': 7, 'state': 3}[mode]
        assert len(captures) == expected, (folder.name, len(captures), expected)
        def cached(r):
            return r['usage']['prompt_tokens_details']['cached_tokens']
        changes = [{'request_id': k, 'cached_delta_vs_first_stock': cached(r)-cached(baseline[k])}
                   for k, r in by_id.items() if cached(r) != cached(baseline[k])]
        results.append({'run': folder.name, 'mode': mode, 'requests': len(rows),
            'target_cached_tokens': cached(by_id['segment007-scan0']),
            'total_cached_tokens': sum(cached(r) for r in rows),
            'input_tokens': sum(r['usage']['prompt_tokens'] for r in rows),
            'captured_keys_by_group': dict(Counter(captures.values())),
            'forced_protected_selections': sum(e.get('forced_protected_selections', 0) for e in probes),
            'changed_requests': changes,
            'output_hash_mismatches_vs_first_stock': sum(r['output_token_ids_sha256'] != baseline[k]['output_token_ids_sha256'] for k,r in by_id.items()),
            'note': 'Cache reuse accounting only; no scheduler execution-token instrumentation or latency claims.'})
    return {'complete': len(results) == 8, 'results': results,
            'limits': 'Future-informed protection of one preselected prefix; partial-group protection may induce different victims. This is not a general policy comparison or proof of optimality.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    value = analyze(a.runs)
    with a.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
