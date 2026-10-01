"""Freeze hardware-dependent settings only from successful stock calibration."""
import argparse
import json
from pathlib import Path

from nineb_artifacts import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'prepared', 'calibration-root', 'settings', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text(encoding='utf-8'))
    budgets = settings['cache_budgets']
    if set(budgets) != {'high_pressure', 'moderate_pressure', 'roomy'}:
        raise ValueError('all three fixed cache budgets are required')
    if not 0 < budgets['high_pressure'] < budgets['moderate_pressure'] < budgets['roomy']:
        raise ValueError('cache budgets must be positive and strictly increasing')
    if not settings.get('native_options_review') or not settings.get('rate_choice_reason'):
        raise ValueError('record native-option applicability and stock rate calibration rationale')
    records = []
    for path in sorted(args.calibration_root.rglob('run.json')):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('mode') == 'calibration':
            records.append((path, value))
    candidates = [value for _, value in records
                  if value.get('status') == 'completed' and value.get('policy') == 'stock'
                  and value['settings']['request_rate'] == settings['request_rate']
                  and value['settings']['native_cache_options'] == settings['native_cache_options']
                  and value['settings']['enforce_eager'] == settings['enforce_eager']]
    if not candidates:
        raise ValueError('no successful stock calibration matches selected settings')
    runtime = candidates[0]['runtime_profile']
    protocol_hash = sha256(args.protocol)
    for budget in budgets.values():
        matching = [value for value in candidates
                    if value['settings']['kv_cache_memory_bytes'] == budget
                    and value['runtime_profile'] == runtime
                    and value['protocol_sha256'] == protocol_hash
                    and value['settings']['concurrency'] >= 16
                    and value['client_summary']['peak_client_active'] >= 8]
        if not matching:
            raise ValueError(f'cache budget {budget} lacks a successful stock calibration at concurrency cap 16')
    value = {'status': 'frozen', 'protocol_sha256': protocol_hash,
             'runtime_profile': runtime, 'model_revision': candidates[0]['model_revision'],
             **settings,
             'request_files': {path.name: sha256(path) for path in sorted(args.prepared.glob('*.jsonl'))},
             'calibration_records': [{'relative_path': str(path.relative_to(args.calibration_root)),
                                      'sha256': sha256(path), 'status': record['status']}
                                     for path, record in records]}
    if len(value['request_files']) != 20:
        raise ValueError('expected 20 locked request files')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
