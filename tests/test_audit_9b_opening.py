import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_9b_opening import CORE_SOURCES, COUNTERS, audit_run


class AuditNineBOpeningTest(unittest.TestCase):
    def fixture(self, root):
        directory = root / 'run-00'
        (directory / 'client').mkdir(parents=True)
        (directory / 'evidence').mkdir()
        runtime = {'versions': {'vllm': '0.30.0'}, 'gpu': {'name': 'GPU', 'memory_mib': 32000}}
        source = {name: f'{index:064x}' for index, name in enumerate(CORE_SOURCES)}
        item = {'scene': 'hot_scan', 'context': 4096, 'rep': 0,
                'budget': 'moderate_pressure', 'concurrency': 8,
                'policy': 'stock', 'workload_variant': 'shared'}
        lock = {'cache_budgets': {'moderate_pressure': 100},
                'request_files': {'hot_scan-ctx4096-rep0.jsonl': 'request-file-hash'},
                'request_rate': 4.0,
                'native_cache_options': {'mamba_cache_mode': 'align', 'prefix_cache_retention_interval': 0},
                'enforce_eager': False,
                'model_revision': 'revision', 'runtime_profile': runtime}
        run = {'status': 'completed', 'mode': 'formal', 'policy': 'stock',
            'settings': {'concurrency': 8, 'kv_cache_memory_bytes': 100, 'workload_variant': 'shared',
                'request_rate': 4.0,
                'native_cache_options': {'mamba_cache_mode': 'align', 'prefix_cache_retention_interval': 0},
                'enforce_eager': False},
            'protocol_sha256': 'protocol-hash', 'model_revision': 'revision',
            'runtime_profile': runtime, 'request_file_sha256': 'request-file-hash',
            'request_count': 2, 'generated_tokens_per_request': 128,
            'owned_server_stopped': True,
            'client_summary': {'request_count': 2, 'success_count': 2, 'error_count': 0},
            'code_version': {'source_sha256': source},
            'engine_evidence': [{'policy': 'stock', 'num_gpu_blocks': 10,
                                 'reuse_hook_installed': False}],
            'policy_measurement': {'status': 'available', 'engines': [{
                'status': 'available', 'policy': 'stock',
                'diagnostics': {name: 0 for name in COUNTERS}}]}}
        run['measurement_reset'] = {'status': 'confirmed', 'success': True, 'attempts': 1}
        (directory / 'run.json').write_text(json.dumps(run), encoding='utf-8')
        (directory / 'evidence/cache-groups-00.json').write_text(json.dumps({
            'total_tensor_bytes': 90, 'num_blocks': 10}), encoding='utf-8')
        (directory / 'client/requests.jsonl').write_text(''.join(json.dumps({
            'request_id': f'q{i}', 'status': 'ok', 'output_tokens': 128,
            'input_token_ids_sha256': f'{i + 10:064x}', 'output_token_ids_sha256': f'{i + 20:064x}'}) + '\n'
            for i in range(2)), encoding='utf-8')
        return directory, item, lock, source

    def test_valid_run_passes_core_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, item, lock, source = self.fixture(Path(temp))
            findings, summary = audit_run(0, item, directory, lock, 'protocol-hash', source, 2)
            self.assertEqual(findings, [])
            self.assertEqual(summary['budget_bytes'], 100)
            self.assertEqual(summary['request_count'], 2)

    def test_capacity_and_measurement_failures_are_findings(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, item, lock, source = self.fixture(Path(temp))
            (directory / 'evidence/cache-groups-00.json').write_text(json.dumps({
                'total_tensor_bytes': 101, 'num_blocks': 10}), encoding='utf-8')
            run_path = directory / 'run.json'
            run = json.loads(run_path.read_text(encoding='utf-8'))
            run['policy_measurement']['status'] = 'missing_or_invalid'
            run['policy_measurement']['engines'][0]['diagnostics']['selection_calls'] = -1
            run['policy_measurement']['engines'][0]['diagnostics']['max_scanned_blocks'] = 33
            run_path.write_text(json.dumps(run), encoding='utf-8')
            findings, _ = audit_run(0, item, directory, lock, 'protocol-hash', source, 2)
            codes = {entry['code'] for entry in findings}
            self.assertIn('physical_cache_bytes_exceed_budget_or_missing', codes)
            self.assertIn('policy_measurement_unavailable', codes)
            self.assertIn('policy_measurement_counters_invalid', codes)


if __name__ == '__main__':
    unittest.main()
