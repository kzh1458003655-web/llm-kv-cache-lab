import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_9b_aa_hashes import analyze


def make_run(root, index, policy, hashes, cached, label='main-hot-0'):
    directory = root / f'run-{index:02d}'
    (directory / 'client').mkdir(parents=True)
    run = {'status': 'completed', 'request_count': 2, 'policy': policy,
           'settings': {'concurrency': 8, 'workload_variant': 'shared'},
           'model_revision': 'fixed-revision',
           'runtime_profile': {'versions': {'vllm': '0.30.0'},
                              'gpu': {'name': 'GPU', 'memory_mib': 32768}}}
    (directory / 'run.json').write_text(json.dumps(run), encoding='utf-8')
    rows = []
    for offset in range(2):
        rows.append({'request_id': f'request-{offset}', 'status': 'ok',
            'input_token_ids_sha256': f'input-{offset}', 'cache_salt_sha256': f'salt-{offset}',
            'output_tokens': 16, 'output_token_ids_sha256': hashes[offset],
            'usage': {'prompt_tokens_details': {'cached_tokens': cached[offset]}}})
    (directory / 'client/requests.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


class AnalyzeNineBAAHashesTest(unittest.TestCase):
    def test_stock_pairs_and_ab_cache_strata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items = []
            def add(label, policy):
                items.append({'label': label, 'policy': policy})
            add('main-hot-0', 'stock'); add('main-hot-0', 'reuse2')
            add('AA-stock-0', 'stock'); add('AA-stock-1', 'stock'); add('AA-stock-2', 'stock')
            add('main-mixed-0', 'stock'); add('main-mixed-0', 'reuse2')
            (root / 'plan.json').write_text(json.dumps({'items': items}), encoding='utf-8')
            make_run(root, 0, 'stock', ['h0', 'h1'], [10, 20])
            make_run(root, 1, 'reuse2', ['h0', 'candidate'], [10, 30])
            make_run(root, 2, 'stock', ['h0', 'a1'], [10, 20], 'AA-stock-0')
            make_run(root, 3, 'stock', ['different', 'h1'], [10, 20], 'AA-stock-1')
            make_run(root, 4, 'stock', ['h0', 'h1'], [10, 20], 'AA-stock-2')
            make_run(root, 5, 'stock', ['m0', 'm1'], [10, 20], 'main-mixed-0')
            make_run(root, 6, 'reuse2', ['m0', 'candidate'], [10, 30], 'main-mixed-0')
            result = analyze(root)
            aa = result['stock_reruns']['pairwise_output_hashes']
            self.assertEqual(len(aa), 6)
            self.assertTrue(any(pair['mismatch_count'] for pair in aa))
            strata = result['ab_cached_token_strata'][1]['cached_token_strata']
            self.assertEqual(strata['same']['request_count'], 1)
            self.assertEqual(strata['different']['request_count'], 1)
            public = json.dumps(result)
            self.assertNotIn('prompt_sha', public)
            self.assertIn('AA differences show rerun variation in stock', public)

    def test_rejects_mismatched_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left = {'index': 1, 'run': {'settings': {}, 'model_revision': 'r', 'runtime_profile': {}},
                    'rows': {'q': {'input_token_ids_sha256': 'one', 'cache_salt_sha256': None,
                                   'output_tokens': 16}}}
            right = {'index': 2, 'run': {'settings': {}, 'model_revision': 'r', 'runtime_profile': {}},
                     'rows': {'q': {'input_token_ids_sha256': 'two', 'cache_salt_sha256': None,
                                    'output_tokens': 16}}}
            from analyze_9b_aa_hashes import same_requests
            with self.assertRaises(ValueError):
                same_requests(left, right)


if __name__ == '__main__':
    unittest.main()
