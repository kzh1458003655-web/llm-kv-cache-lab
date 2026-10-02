"""Audit completed 9B opening runs without touching the serving system."""
import argparse
import hashlib
import json
from pathlib import Path

CORE_SOURCES = ('scripts/run_9b_online.py', 'server-kit/nineb-bootstrap/sitecustomize.py',
                'kv_cache_lab/hybrid_retention_prototype.py', 'kv_cache_lab/hybrid_observer.py',
                'kv_cache_lab/events.py')
COUNTERS = ('selection_calls', 'selection_cpu_ns', 'reordered_calls', 'max_scanned_blocks')


def valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(ch in '0123456789abcdef' for ch in value.lower())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finding(index, code):
    return {'run_index': index, 'code': code}


def run_evidence(directory, run):
    evidence_dir = directory / 'evidence'
    groups = sorted(evidence_dir.glob('cache-groups-*.json'))
    engines = run.get('engine_evidence') or [json.loads(path.read_text(encoding='utf-8'))
        for path in sorted(evidence_dir.glob('engine-active-*.json'))]
    return ([json.loads(path.read_text(encoding='utf-8')) for path in groups], engines)


def request_file(directory):
    candidates = (directory / 'client/requests.jsonl', directory / 'requests.jsonl')
    return next((path for path in candidates if path.is_file()), None)


def audit_run(index, item, directory, lock, protocol_sha, expected_source, expected_count=512):
    findings = []
    def bad(code):
        findings.append(finding(index, code))
    try:
        run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return [finding(index, 'missing_or_invalid_run_json')], None
    if run.get('status') != 'completed' or run.get('mode') != 'formal':
        bad('run_not_completed_formal')
    if run.get('policy') != item.get('policy'):
        bad('policy_mismatch')
    settings = run.get('settings') or {}
    if settings.get('concurrency') != item.get('concurrency'):
        bad('concurrency_mismatch')
    for setting, lock_key in (('request_rate', 'request_rate'),
                              ('native_cache_options', 'native_cache_options'),
                              ('enforce_eager', 'enforce_eager')):
        if settings.get(setting) != lock.get(lock_key):
            bad(f'{setting}_mismatch')
    if run.get('owned_server_stopped') is not True:
        bad('owned_server_not_stopped')
    expected_budget = lock.get('cache_budgets', {}).get(item.get('budget'))
    actual_budget = settings.get('kv_cache_memory_bytes')
    if expected_budget is None or actual_budget != expected_budget:
        bad('cache_budget_mismatch')
    if settings.get('workload_variant') != item.get('workload_variant'):
        bad('workload_variant_mismatch')
    if run.get('protocol_sha256') != protocol_sha:
        bad('protocol_hash_mismatch')
    if run.get('model_revision') != lock.get('model_revision'):
        bad('model_revision_mismatch')
    if run.get('runtime_profile') != lock.get('runtime_profile'):
        bad('runtime_profile_mismatch')
    request_name = f"{item['scene']}-ctx{item['context']}-rep{item['rep']}.jsonl"
    if run.get('request_file_sha256') != lock.get('request_files', {}).get(request_name):
        bad('request_file_hash_mismatch')
    summary = run.get('client_summary') or {}
    if (run.get('request_count') != expected_count or summary.get('request_count') != expected_count
            or summary.get('success_count') != expected_count or summary.get('error_count') != 0):
        bad('request_success_count_mismatch')
    if run.get('generated_tokens_per_request') != 128:
        bad('generated_token_budget_mismatch')

    rows_path = request_file(directory)
    rows = []
    if rows_path is None:
        bad('request_results_missing')
    else:
        try:
            rows = jsonl(rows_path)
            ids = [row.get('request_id') for row in rows]
            if (len(rows) != expected_count or len(set(ids)) != expected_count
                    or any(row.get('status') != 'ok' or row.get('output_tokens') != 128
                           or not valid_sha256(row.get('input_token_ids_sha256'))
                           or not valid_sha256(row.get('output_token_ids_sha256')) for row in rows)):
                bad('request_rows_invalid')
        except (OSError, ValueError, TypeError):
            bad('request_rows_invalid')

    code_version = run.get('code_version') or {}
    sources = code_version.get('source_sha256') or {}
    if any(not valid_sha256(sources.get(name)) for name in CORE_SOURCES):
        bad('core_source_hashes_missing')
    core_sources = {name: sources[name] for name in CORE_SOURCES
                    if valid_sha256(sources.get(name))}
    if len(core_sources) == len(CORE_SOURCES) and expected_source is not None and core_sources != expected_source:
        bad('core_source_hash_mismatch')

    try:
        groups, engines = run_evidence(directory, run)
        if not groups:
            bad('cache_group_evidence_missing')
        elif any(not isinstance(group.get('total_tensor_bytes'), int)
                 or group['total_tensor_bytes'] < 0
                 or group['total_tensor_bytes'] > actual_budget for group in groups):
            bad('physical_cache_bytes_exceed_budget_or_missing')
        if not engines:
            bad('engine_evidence_missing')
        elif any(engine.get('policy') != item.get('policy')
                 or engine.get('reuse_hook_installed') != (item.get('policy') == 'reuse2')
                 for engine in engines):
            bad('engine_hook_policy_mismatch')
    except (OSError, ValueError, TypeError):
        groups, engines = [], []
        bad('engine_evidence_invalid')

    measurement = run.get('policy_measurement') or {}
    measured = measurement.get('engines') or []
    counters_invalid = any(
        any(not isinstance((entry.get('diagnostics') or {}).get(name), int)
            or isinstance((entry.get('diagnostics') or {}).get(name), bool)
            or (entry.get('diagnostics') or {}).get(name) < 0 for name in COUNTERS)
        or (entry.get('diagnostics') or {}).get('max_scanned_blocks', 33) > 32
        for entry in measured)
    if counters_invalid:
        bad('policy_measurement_counters_invalid')
    if (measurement.get('status') != 'available'
            or not (run.get('measurement_reset') or {}).get('success')
            or not engines or len(measured) != len(engines)
            or any(entry.get('status') != 'available'
                   or entry.get('policy') != item.get('policy')
                   or any(name not in (entry.get('diagnostics') or {}) for name in COUNTERS)
                   for entry in measured)):
        bad('policy_measurement_unavailable')
    return findings, {'index': index, 'policy': item.get('policy'),
        'budget_bytes': actual_budget, 'concurrency': settings.get('concurrency'),
        'request_count': len(rows),
        'has_core_source_hashes': all(valid_sha256(sources.get(name)) for name in CORE_SOURCES),
        'source_sha256': core_sources if len(core_sources) == len(CORE_SOURCES) else None}


def audit(runs, lock_path, protocol_path):
    runs, lock_path, protocol_path = Path(runs), Path(lock_path), Path(protocol_path)
    findings = []
    try:
        lock = json.loads(lock_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        lock = {}
        findings.append(finding(None, 'lock_missing_or_invalid'))
    if not isinstance(lock, dict):
        lock = {}
        findings.append(finding(None, 'lock_schema_invalid'))
    try:
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        protocol_sha = file_sha256(protocol_path)
    except (OSError, ValueError):
        protocol, protocol_sha = {}, None
        findings.append(finding(None, 'protocol_missing_or_invalid'))
    if not isinstance(protocol, dict):
        protocol = {}
        findings.append(finding(None, 'protocol_schema_invalid'))
    manifest = []
    if lock.get('status') != 'frozen' or lock.get('protocol_sha256') != protocol_sha:
        findings.append(finding(None, 'lock_or_protocol_hash_invalid'))
    workload = protocol.get('online_workload') or {}
    if not isinstance(workload, dict) or workload.get('output_tokens') != 128:
        findings.append(finding(None, 'protocol_output_budget_not_128'))
    try:
        plan = json.loads((runs / 'plan.json').read_text(encoding='utf-8'))
        items = plan['items']
    except (OSError, ValueError, KeyError, TypeError):
        items = []
        findings.append(finding(None, 'plan_missing_or_invalid'))
    if not isinstance(items, list):
        items = []
    if len(items) != 25:
        findings.append(finding(None, 'formal_plan_count_not_25'))
    expected_source = None
    for index, item in enumerate(items):
        try:
            if not isinstance(item, dict):
                raise TypeError('invalid plan item')
            run_findings, summary = audit_run(index, item, runs / f'run-{index:02d}', lock,
                                               protocol_sha, expected_source)
        except Exception:
            run_findings, summary = [finding(index, 'run_audit_exception')], None
        if summary and summary.get('has_core_source_hashes'):
            if expected_source is None:
                expected_source = summary['source_sha256']
        findings.extend(run_findings)
        if summary:
            summary.pop('source_sha256', None)
            manifest.append(summary)
    return {'status': 'pass' if not findings and len(manifest) == 25 else 'findings',
            'planned_run_count': len(items), 'audited_run_count': len(manifest),
            'valid_run_count': sum(not any(f['run_index'] == row['index'] for f in findings)
                                   for row in manifest),
            'protocol_sha256': protocol_sha,
            'core_source_hashes_consistent': not any(f['code'] in
                ('core_source_hashes_missing', 'core_source_hash_mismatch') for f in findings),
            'findings': findings, 'runs': manifest,
            'privacy_note': 'summary contains run indices, counts, budget values, and source hashes; no local paths, prompts, token text, or GPU UUIDs'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--lock', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = audit(args.runs, args.lock, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'audited_run_count': result['audited_run_count'],
                      'valid_run_count': result['valid_run_count'], 'finding_count': len(result['findings'])}))


if __name__ == '__main__':
    main()
