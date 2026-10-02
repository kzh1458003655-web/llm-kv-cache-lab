"""One bounded Linux vLLM run, with explicit calibration or frozen settings."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import hashlib
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from nineb_artifacts import sha256, verify_model
from nineb_online_client import run_client

ROOT = Path(__file__).resolve().parents[1]
NATIVE_OPTIONS = {'mamba_cache_mode', 'prefix_match_unit', 'prefix_cache_retention_interval'}
MEASUREMENT_COUNTERS = ('selection_calls', 'selection_cpu_ns', 'reordered_calls', 'max_scanned_blocks')


def measurement_reset(base_url):
    request = urllib.request.Request(base_url + '/reset_prefix_cache', data=b'', method='POST')
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read())
            if isinstance(result, dict) and result.get('success') is True:
                return {'status': 'confirmed', 'success': True, 'attempts': attempt + 1}
        except Exception as exc:
            error_type = type(exc).__name__
        else:
            error_type = 'ResetNotConfirmed'
        time.sleep(.2)
    return {'status': 'failed', 'success': False, 'attempts': 5,
            'error_type': error_type}


def collect_policy_measurement(evidence, markers, reset_result):
    engines = []
    for index, marker in enumerate(markers):
        pid = marker.get('pid')
        path = Path(evidence) / f'policy-measurement-{pid}.json' if pid else None
        item = {'engine_index': index, 'status': 'missing'}
        if reset_result.get('success') and path and path.exists():
            try:
                value = json.loads(path.read_text(encoding='utf-8'))
                if value.get('pid') != pid or value.get('policy') != marker.get('policy'):
                    raise ValueError('measurement identity mismatch')
                counts = value.get('diagnostics', {})
                if not all(name in counts for name in MEASUREMENT_COUNTERS):
                    raise ValueError('measurement counters incomplete')
                item = {'engine_index': index, 'status': 'available',
                        'policy': value['policy'],
                        'diagnostics': {name: counts[name] for name in MEASUREMENT_COUNTERS}}
            except Exception as exc:
                item = {'engine_index': index, 'status': 'invalid',
                        'error_type': type(exc).__name__}
        engines.append(item)
    if not reset_result.get('success'):
        status = 'reset_failed'
    elif not engines or any(item['status'] != 'available' for item in engines):
        status = 'missing_or_invalid'
    else:
        status = 'available'
    return {'status': status, 'engines': engines}


def profile(protocol):
    versions = {name: importlib.metadata.version(name) for name in ('vllm', 'torch', 'transformers')}
    for name, actual in versions.items():
        expected = protocol['runtime'][name]
        if actual.split('+')[0] != expected:
            raise ValueError(f'{name} version differs: expected {expected}, got {actual}')
    command = ['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader,nounits']
    output = subprocess.check_output(command, text=True, timeout=20)
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError('fixed protocol requires one visible GPU')
    name, uuid, memory, driver = [part.strip() for part in lines[0].split(',')]
    return {'versions': versions, 'gpu': {'name': name, 'uuid': uuid, 'memory_mib': int(memory), 'driver': driver}}


def stop_owned(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    # The parent may exit while spawned engine workers remain in its group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('empty request manifest')
    for row in rows:
        if len(row['prompt_token_ids']) != row['expected_prompt_tokens']:
            raise ValueError('input token count mismatch')
    return rows


def boundary_rows(rows, evidence, max_model_len):
    groups = [json.loads(path.read_text()) for path in evidence.glob('cache-groups-*.json')]
    if not groups:
        raise RuntimeError('native cache-group evidence missing')
    base = max(rows, key=lambda row: len(row['prompt_token_ids']))['prompt_token_ids']
    requests = []
    # Both sides of every actual group boundary, with one unrelated suffix.
    lengths = sorted({multiple * group['block_size'] + delta
                      for config in groups for group in config['groups']
                      for multiple in (1, 2) for delta in (-1, 0, 1)
                      if 8 < multiple * group['block_size'] + delta < min(len(base), max_model_len - 128)})
    for length in lengths:
        for suffix in (101, 102):
            ids = base[:length] + [suffix]
            digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
            requests.append({'request_id': f'boundary-{length}-suffix{suffix}',
                             'doc_id': f'prefix-length-{length}', 'prompt_sha256': None,
                             'prompt_token_ids': ids, 'input_token_ids_sha256': digest,
                             'shared_prefix_tokens': length,
                             'cache_salt': f'nineb-boundary-{length}',
                             'expected_prompt_tokens': len(ids), 'arrival_units': 0})
    if not requests:
        raise ValueError('no group boundaries fit in the prepared prefix')
    return requests


def apply_workload_variant(rows, variant):
    if variant == 'shared':
        return rows
    if variant != 'no_reuse':
        raise ValueError('unknown workload variant')
    return [{**row, 'cache_salt': 'nineb-isolated-' + hashlib.sha256(
        (row['request_id'] + ':' + row['input_token_ids_sha256']).encode()).hexdigest()} for row in rows]


def run(args):
    run_started = time.monotonic()
    if os.name != 'posix':
        raise ValueError('GPU runner requires Linux')
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    runtime = profile(protocol)
    verification = verify_model(args.manifest, args.model, cache_receipt=True)
    rows = load_rows(args.requests)
    settings = {'native_cache_options': {'mamba_cache_mode': 'align'}, 'enforce_eager': False,
                'request_rate': args.rate, 'concurrency': args.concurrency,
                'kv_cache_memory_bytes': args.kv_cache_bytes}
    if args.lock:
        lock = json.loads(args.lock.read_text(encoding='utf-8'))
        if lock.get('status') != 'frozen' or lock['protocol_sha256'] != sha256(args.protocol):
            raise ValueError('missing frozen calibration or protocol hash mismatch')
        if lock['model_revision'] != verification['revision'] or lock['runtime_profile'] != runtime:
            raise ValueError('hardware, dependency, or model changed from frozen calibration')
        if sha256(args.requests) != lock['request_files'].get(args.requests.name):
            raise ValueError('request file differs from the frozen calibration')
        settings.update({'native_cache_options': lock['native_cache_options'],
                         'enforce_eager': lock['enforce_eager'],
                         'request_rate': lock['request_rate'],
                         'kv_cache_memory_bytes': lock['cache_budgets'][args.budget]})
        if args.boundary_probe and args.mode == 'formal':
            raise ValueError('boundary diagnostics run separately from formal latency runs')
        if args.mode == 'formal' and args.concurrency not in protocol['online_workload']['client_concurrency_levels']:
            raise ValueError('formal concurrency is outside the fixed protocol')
    elif args.mode == 'formal':
        raise ValueError('formal run requires --lock; initial runs must use --mode calibration')
    else:
        if args.mode == 'calibration' and args.policy != 'stock':
            raise ValueError('calibrate with stock only')
        if args.count:
            rows = rows[:args.count]
        if args.native_options:
            settings['native_cache_options'] = json.loads(args.native_options.read_text(encoding='utf-8'))
        settings['enforce_eager'] = args.enforce_eager
    if args.mode == 'diagnostic' and args.count:
        rows = rows[:args.count]
    rows = apply_workload_variant(rows, args.workload_variant)
    settings['workload_variant'] = args.workload_variant
    if not settings['request_rate'] or settings['request_rate'] <= 0:
        raise ValueError('a positive calibrated request rate is required')
    if set(settings['native_cache_options']) - NATIVE_OPTIONS:
        raise ValueError('unsupported native cache option in settings')
    # Refuse to benchmark an unrelated server already using this port.
    with socket.socket() as port_check:
        # Closed connections can remain in TIME_WAIT between successive runs.
        # Reuse that address without permitting a second active listener.
        port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_check.bind(('127.0.0.1', args.port))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = out / 'evidence'
    evidence.mkdir()
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(args.model.resolve()), '--served-model-name', 'thesis-9b',
        '--host', '127.0.0.1', '--port', str(args.port), '--language-model-only',
        '--dtype', 'bfloat16', '--max-model-len', str(protocol['runtime']['max_model_len']),
        '--max-num-seqs', str(protocol['runtime']['max_num_seqs']),
        '--max-num-batched-tokens', str(protocol['runtime']['max_num_batched_tokens']),
        '--gpu-memory-utilization', '0.90', '--enable-prefix-caching',
        '--enable-prompt-tokens-details', '--safetensors-load-strategy', 'lazy',
        '--generation-config', 'vllm', '--disable-uvicorn-access-log']
    for name, value in sorted(settings['native_cache_options'].items()):
        if value is not None:
            command.extend(['--' + name.replace('_', '-'), str(value)])
    if settings['kv_cache_memory_bytes'] is not None:
        command.extend(['--kv-cache-memory-bytes', str(settings['kv_cache_memory_bytes'])])
    if settings['enforce_eager']:
        command.append('--enforce-eager')
    env = dict(os.environ)
    env.update({'NINEB_POLICY': args.policy, 'NINEB_EVIDENCE_DIR': str(evidence),
                'VLLM_SERVER_DEV_MODE': '1',
                'VLLM_USE_V2_MODEL_RUNNER': '0', 'VLLM_USE_FLASHINFER_SAMPLER': '0',
                'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'HF_HUB_OFFLINE': '1',
                'PYTHONPATH': os.pathsep.join([str(ROOT / 'server-kit/nineb-bootstrap'), str(ROOT),
                                             env.get('PYTHONPATH', '')])})
    if args.boundary_probe or args.trace_diagnostic:
        env['NINEB_DIAGNOSTIC_EVENTS'] = str(out / 'diagnostic-events.jsonl')
        settings['concurrency'] = 1
    record = {'status': 'starting', 'mode': args.mode, 'policy': args.policy,
              'settings': settings, 'runtime_profile': runtime,
              'model_revision': verification['revision'], 'model_verification': verification,
              'protocol_sha256': sha256(args.protocol), 'request_file_sha256': sha256(args.requests),
              'request_count': len(rows), 'command': command,
              'environment_overrides': {key: env[key] for key in ('VLLM_USE_V2_MODEL_RUNNER', 'VLLM_USE_FLASHINFER_SAMPLER', 'VLLM_WORKER_MULTIPROC_METHOD', 'VLLM_SERVER_DEV_MODE')},
              'shutdown_note': 'owned serving process is stopped; instance must be shut down after download verification'}
    record['boundary_probe'] = args.boundary_probe
    record['trace_diagnostic'] = args.trace_diagnostic
    record['generated_tokens_per_request'] = 16 if args.boundary_probe else protocol['online_workload']['output_tokens']
    if (ROOT / 'CODE_VERSION.json').exists():
        record['code_version'] = json.loads((ROOT / 'CODE_VERSION.json').read_text())
    source_paths = ('scripts/run_9b_online.py', 'server-kit/nineb-bootstrap/sitecustomize.py',
                    'kv_cache_lab/hybrid_retention_prototype.py', 'kv_cache_lab/hybrid_observer.py',
                    'kv_cache_lab/events.py')
    record.setdefault('code_version', {})['source_sha256'] = {
        name: sha256(ROOT / name) for name in source_paths if (ROOT / name).is_file()}
    provenance = ROOT / 'execution-provenance.json'
    if provenance.is_file():
        record['code_version']['execution_provenance_sha256'] = sha256(provenance)
    (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    process = None
    base = f'http://127.0.0.1:{args.port}'
    try:
        with (out / 'server-private.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + args.startup_seconds
            while True:
                if process.poll() is not None:
                    raise RuntimeError('server exited during startup; inspect private log')
                try:
                    with urllib.request.urlopen(base + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('server startup time limit exceeded')
                time.sleep(1)
            if args.boundary_probe:
                rows = boundary_rows(rows, evidence, protocol['runtime']['max_model_len'])
                record['request_count'] = len(rows)
                record['boundary_inputs'] = [{key: value for key, value in row.items()
                                              if key != 'prompt_token_ids'} for row in rows]
            record['startup_elapsed_s'] = time.monotonic() - run_started
            summary = asyncio.run(asyncio.wait_for(
                run_client(base, rows, settings['concurrency'], settings['request_rate'],
                           record['generated_tokens_per_request'], out / 'client'),
                timeout=args.measurement_seconds))
            reset_result = measurement_reset(base)
            record['measurement_reset'] = reset_result
            markers = [json.loads(path.read_text()) for path in evidence.glob('engine-active-*.json')]
            if not markers or any(marker['policy'] != args.policy for marker in markers):
                raise RuntimeError('no actual engine allocation evidence for the selected policy')
            if args.policy == 'reuse2' and not all(marker['reuse_hook_installed'] for marker in markers):
                raise RuntimeError('candidate hook was not installed in the allocating engine')
            record['policy_measurement'] = collect_policy_measurement(evidence, markers, reset_result)
            record['status'] = 'completed' if summary['error_count'] == 0 else 'request_errors'
            record['client_summary'] = summary
            record['engine_evidence'] = markers
    except BaseException as exc:
        record.update({'status': 'failed', 'error_type': type(exc).__name__})
        raise
    finally:
        stop_owned(process)
        record['owned_server_stopped'] = True
        record['total_elapsed_s'] = time.monotonic() - run_started
        (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return 0 if record['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'manifest', 'protocol', 'requests', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=('calibration', 'diagnostic', 'formal'), default='formal')
    parser.add_argument('--policy', choices=('stock', 'reuse2'), default='stock')
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--budget', choices=('high_pressure', 'moderate_pressure', 'roomy'), default='moderate_pressure')
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--rate', type=float)
    parser.add_argument('--kv-cache-bytes', type=int)
    parser.add_argument('--count', type=int)
    parser.add_argument('--native-options', type=Path)
    parser.add_argument('--enforce-eager', action='store_true')
    parser.add_argument('--boundary-probe', action='store_true')
    parser.add_argument('--trace-diagnostic', action='store_true')
    parser.add_argument('--workload-variant', choices=('shared', 'no_reuse'), default='shared')
    parser.add_argument('--port', type=int, default=8019)
    parser.add_argument('--startup-seconds', type=int, default=480)
    parser.add_argument('--measurement-seconds', type=int, default=600)
    args = parser.parse_args()
    if (args.boundary_probe or args.trace_diagnostic) and args.mode != 'diagnostic':
        parser.error('event diagnostics require --mode diagnostic')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
