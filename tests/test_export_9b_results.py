import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import export_9b_results


class ExportNineBResultsTest(unittest.TestCase):
    def test_exports_cache_capacity_and_filtered_engine_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            runs, output = base / 'runs', base / 'public'
            run = runs / 'run-00'
            evidence = run / 'evidence'
            evidence.mkdir(parents=True)
            (evidence / 'cache-groups-9876.json').write_text(json.dumps({
                'pid': 9876, 'policy': 'reuse2', 'vllm': '0.30.0',
                'num_blocks': 321, 'resolved_prefix_cache_retention_interval': 2,
                'groups': [{'spec': 'FullAttentionSpec', 'block_size': 16,
                            'page_size_bytes': 4096, 'layers': ['layer.0'],
                            'gpu_uuid': 'PRIVATE-GPU-UUID'}],
                'tensor_bytes': [123456], 'tensor_descriptor_size_sum': 123456,
                'total_tensor_bytes': 123456, 'allocation_note': 'shared buffer',
                'tensor_descriptors': [{'size': 123456, 'layers': ['layer.0'],
                    'offset': 0, 'layer_stride': 128, 'block_stride': 4096,
                    'pid': 9876}],
                'gpu_uuid': 'PRIVATE-GPU-UUID'}), encoding='utf-8')
            (evidence / 'engine-active-9876.json').write_text(json.dumps({
                'pid': 9876, 'policy': 'reuse2', 'vllm': '0.30.0',
                'num_gpu_blocks': 321, 'hash_block_size': 16,
                'reuse_hook_installed': True, 'gpu_uuid': 'PRIVATE-GPU-UUID'}), encoding='utf-8')
            (run / 'server-private.log').write_text('private log marker', encoding='utf-8')
            (run / 'run.json').write_text(json.dumps({
                'status': 'completed', 'mode': 'formal', 'policy': 'reuse2',
                'settings': {'kv_cache_memory_bytes': 123456},
                'runtime_profile': {'versions': {}, 'gpu': {
                    'name': 'GPU', 'memory_mib': 1, 'driver': 'x', 'uuid': 'PRIVATE-GPU-UUID'}},
                'engine_evidence': [{'pid': 9876, 'policy': 'reuse2', 'vllm': '0.30.0',
                    'num_gpu_blocks': 321, 'hash_block_size': 16,
                    'reuse_hook_installed': True, 'gpu_uuid': 'PRIVATE-GPU-UUID'}]}),
                encoding='utf-8')
            with patch.object(sys, 'argv', ['export_9b_results.py',
                    '--runs', str(runs), '--output', str(output)]):
                export_9b_results.main()

            cache = json.loads((output / 'run-00/evidence/cache-groups-00.json').read_text())
            engine = json.loads((output / 'run-00/evidence/engine-active-00.json').read_text())
            self.assertEqual(cache['total_tensor_bytes'], 123456)
            self.assertEqual(cache['num_blocks'], 321)
            self.assertEqual(cache['groups'][0]['layers'], ['layer.0'])
            self.assertEqual(cache['tensor_descriptors'][0]['block_stride'], 4096)
            self.assertEqual(engine['num_gpu_blocks'], 321)
            self.assertNotIn('pid', cache)
            self.assertNotIn('pid', engine)
            self.assertNotIn('uuid', json.loads((output / 'run-00/run.json').read_text())['runtime_profile']['gpu'])
            self.assertFalse((output / 'run-00/server-private.log').exists())
            for path in output.rglob('*'):
                if path.is_file():
                    self.assertNotIn('PRIVATE-GPU-UUID', path.read_text(encoding='utf-8'))
                    self.assertNotIn('private log marker', path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
