import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import export_9b_results
from run_9b_online import collect_policy_measurement


class PolicyMeasurementTest(unittest.TestCase):
    def test_collects_only_engine_pid_and_reports_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence = Path(temp)
            counts = {'selection_calls': 2, 'selection_cpu_ns': 101,
                      'reordered_calls': 1, 'max_scanned_blocks': 32}
            (evidence / 'policy-measurement-41.json').write_text(json.dumps({
                'pid': 41, 'policy': 'reuse2', 'diagnostics': counts}), encoding='utf-8')
            result = collect_policy_measurement(evidence, [
                {'pid': 41, 'policy': 'reuse2'}, {'pid': 77, 'policy': 'reuse2'}],
                {'success': True})
            self.assertEqual(result['status'], 'missing_or_invalid')
            self.assertEqual(result['engines'][0]['diagnostics'], counts)
            self.assertEqual(result['engines'][1]['status'], 'missing')
            self.assertNotIn('pid', json.dumps(result))
            failed = collect_policy_measurement(evidence, [{'pid': 41, 'policy': 'reuse2'}],
                                                {'success': False})
            self.assertEqual(failed['status'], 'reset_failed')
            self.assertEqual(failed['engines'][0]['status'], 'missing')

    def test_export_filters_measurement_pid_and_uuid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / 'runs' / 'one'
            run.mkdir(parents=True)
            (run / 'run.json').write_text(json.dumps({
                'status': 'completed', 'runtime_profile': {'versions': {}, 'gpu': {
                    'name': 'GPU', 'memory_mib': 8, 'driver': 'x', 'uuid': 'SECRET-UUID'}},
                'measurement_reset': {'status': 'confirmed', 'success': True,
                                      'attempts': 1, 'pid': 41},
                'policy_measurement': {'status': 'available', 'pid': 41,
                    'engines': [{'engine_index': 0, 'status': 'available', 'pid': 41,
                        'policy': 'reuse2', 'diagnostics': {'selection_calls': 2},
                        'uuid': 'SECRET-UUID'}]}}), encoding='utf-8')
            output = root / 'public'
            with patch.object(sys, 'argv', ['export_9b_results.py', '--runs',
                                             str(root / 'runs'), '--output', str(output)]):
                export_9b_results.main()
            text = (output / 'one/run.json').read_text(encoding='utf-8')
            public = json.loads(text)
            self.assertEqual(public['policy_measurement']['engines'][0]['diagnostics']['selection_calls'], 2)
            self.assertNotIn('pid', text)
            self.assertNotIn('SECRET-UUID', text)


if __name__ == '__main__':
    unittest.main()
