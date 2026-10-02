"""Export an explicit public subset; never copy logs, prompts, or credentials."""
import argparse
import json
from pathlib import Path


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
        runtime = value['runtime_profile']
        public['runtime_profile'] = {'versions': runtime['versions'],
            'gpu': {key: runtime['gpu'][key] for key in ('name', 'memory_mib', 'driver')}}
        (out / 'run.json').write_text(json.dumps(public, indent=2) + '\n', encoding='utf-8')
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
    for name in ('plan.json', 'progress.json', 'analysis.json', 'boundary-analysis.json', 'pressure-analysis.json'):
        for source in args.runs.rglob(name):
            destination = args.output / source.relative_to(args.runs)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
    (args.output / 'export.json').write_text(json.dumps({'runs': exported, 'raw_logs_exported': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
