"""Export an explicit public subset; never copy logs, prompts, or credentials."""
import argparse
import json
from pathlib import Path


def export_evidence(source: Path, out: Path, value: dict) -> None:
    evidence_out = out / 'evidence'
    cache_files = sorted((source / 'evidence').glob('cache-groups-*.json'))
    engine_files = sorted((source / 'evidence').glob('engine-active-*.json'))
    if cache_files or engine_files or value.get('engine_evidence'):
        evidence_out.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(cache_files):
        record = json.loads(path.read_text(encoding='utf-8'))
        public = {key: record[key] for key in ('policy', 'vllm', 'num_blocks',
            'resolved_prefix_cache_retention_interval', 'tensor_bytes',
            'tensor_descriptor_size_sum', 'total_tensor_bytes', 'allocation_note',
            ) if key in record}
        public['groups'] = [{key: group[key] for key in
            ('spec', 'block_size', 'page_size_bytes', 'layers') if key in group}
            for group in record.get('groups', [])]
        public['tensor_descriptors'] = [{key: descriptor[key] for key in
            ('size', 'layers', 'offset', 'layer_stride', 'block_stride') if key in descriptor}
            for descriptor in record.get('tensor_descriptors', [])]
        (evidence_out / f'cache-groups-{index:02d}.json').write_text(
            json.dumps(public, indent=2) + '\n', encoding='utf-8')
    engine = value.get('engine_evidence', [])
    if not engine:
        engine = [json.loads(path.read_text(encoding='utf-8')) for path in engine_files]
    for index, record in enumerate(engine):
        public = {key: record[key] for key in ('policy', 'vllm', 'num_gpu_blocks',
                    'hash_block_size', 'reuse_hook_installed') if key in record}
        (evidence_out / f'engine-active-{index:02d}.json').write_text(
            json.dumps(public, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    exported = []
    for source in sorted(args.runs.rglob('run.json')):
        value = json.loads(source.read_text())
        relative = source.parent.relative_to(args.runs)
        out = args.output / relative
        out.mkdir(parents=True, exist_ok=True)
        allowed = ('status', 'mode', 'policy', 'settings', 'model_revision', 'model_verification',
                   'protocol_sha256', 'request_file_sha256', 'request_count', 'environment_overrides',
                   'owned_server_stopped', 'client_summary', 'error_type', 'boundary_probe', 'boundary_inputs', 'code_version',
                   'trace_diagnostic', 'generated_tokens_per_request', 'startup_elapsed_s', 'total_elapsed_s')
        public = {key: value[key] for key in allowed if key in value}
        if 'measurement_reset' in value:
            reset = value['measurement_reset']
            public['measurement_reset'] = {key: reset[key] for key in
                ('status', 'success', 'attempts', 'error_type') if key in reset}
        if 'policy_measurement' in value:
            measurement = value['policy_measurement']
            public['policy_measurement'] = {'status': measurement.get('status'),
                'engines': [{key: engine[key] for key in
                    ('engine_index', 'status', 'policy', 'diagnostics', 'error_type') if key in engine}
                    for engine in measurement.get('engines', [])]}
            for engine in public['policy_measurement']['engines']:
                if 'diagnostics' in engine:
                    engine['diagnostics'] = {key: engine['diagnostics'][key] for key in
                        ('selection_calls', 'selection_cpu_ns', 'reordered_calls', 'max_scanned_blocks')
                        if key in engine['diagnostics']}
        runtime = value['runtime_profile']
        public['runtime_profile'] = {'versions': runtime['versions'],
            'gpu': {key: runtime['gpu'][key] for key in ('name', 'memory_mib', 'driver')}}
        (out / 'run.json').write_text(json.dumps(public, indent=2) + '\n', encoding='utf-8')
        export_evidence(source.parent, out, value)
        requests = source.parent / 'client/requests.jsonl'
        if requests.exists():
            with requests.open(encoding='utf-8') as stream, (out / 'requests.jsonl').open('x', encoding='utf-8') as output:
                for line in stream:
                    row = json.loads(line)
                    if any(key in row for key in ('prompt', 'output_text', 'prompt_token_ids', 'error')):
                        raise ValueError('request record contains text or an unreviewed error body')
                    output.write(json.dumps(row, separators=(',', ':')) + '\n')
        for name in ('diagnostic-events.jsonl',):
            path = source.parent / name
            if path.exists():
                with path.open(encoding='utf-8') as stream, (out / name).open('x', encoding='utf-8') as output:
                    for line in stream:
                        record = json.loads(line)
                        if any(key in record for key in ('prompt', 'output_text')):
                            raise ValueError('unexpected diagnostic text')
                        output.write(json.dumps(record, separators=(',', ':')) + '\n')
        exported.append(str(relative))
    for name in ('plan.json', 'progress.json', 'analysis.json', 'boundary-analysis.json',
                 'pressure-analysis.json', 'aa-output-hash-analysis.json'):
        for source in args.runs.rglob(name):
            destination = args.output / source.relative_to(args.runs)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
    (args.output / 'export.json').write_text(json.dumps({'runs': exported, 'raw_logs_exported': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
