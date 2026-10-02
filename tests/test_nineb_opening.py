"""CPU regressions for experiment validity; no model, GPU, or remote service."""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nineb_online_client import run_client
from run_9b_online import apply_workload_variant, boundary_rows
from run_9b_round import diagnostics_plan, plan, run_bounded
from analyze_9b_traces import summarize
from freeze_9b_calibration import main as freeze
from nineb_artifacts import sha256


def input_row(index=0):
    return {'request_id': str(index), 'doc_id': str(index), 'prompt_sha256': 'text-hash',
            'input_token_ids_sha256': 'ids-' + str(index), 'prompt_token_ids': [1, 2],
            'expected_prompt_tokens': 2, 'arrival_units': 0}


class DesignChecks(unittest.TestCase):
    def test_balanced_pairs_and_fixed_controls(self):
        items = plan()
        self.assertEqual(len(items), 25)
        for prefix in ('main-hot-', 'main-mixed-'):
            self.assertEqual({item['rep'] for item in items if item['label'].startswith(prefix)}, {0, 1, 2})
        for label in {item['label'] for item in items if not item['label'].startswith('AA-')}:
            pair = [item for item in items if item['label'] == label]
            self.assertEqual({item['policy'] for item in pair}, {'stock', 'reuse2'})
            self.assertEqual({k: v for k, v in pair[0].items() if k != 'policy'},
                             {k: v for k, v in pair[1].items() if k != 'policy'})
        self.assertEqual(len(diagnostics_plan()), 4)

    def test_no_reuse_and_boundary_cases_are_isolated(self):
        rows = [input_row(index) for index in range(4)]
        transformed = apply_workload_variant(rows, 'no_reuse')
        self.assertEqual(len({row['cache_salt'] for row in transformed}), 4)
        self.assertEqual(transformed, apply_workload_variant(rows, 'no_reuse'))
        self.assertTrue(all('cache_salt' not in row for row in rows))
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            (evidence / 'cache-groups-1.json').write_text(json.dumps({'groups': [{'block_size': 16}, {'block_size': 128}]}))
            cases = boundary_rows([{'prompt_token_ids': [7] * 600}], evidence, 8192)
        lengths = {row['shared_prefix_tokens'] for row in cases}
        self.assertEqual(len({row['cache_salt'] for row in cases}), len(lengths))
        for length in lengths:
            pair = [row for row in cases if row['shared_prefix_tokens'] == length]
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0]['cache_salt'], pair[1]['cache_salt'])
            self.assertEqual(pair[0]['prompt_token_ids'][:-1], pair[1]['prompt_token_ids'][:-1])
            self.assertNotEqual(pair[0]['prompt_token_ids'][-1], pair[1]['prompt_token_ids'][-1])

    def test_exhausted_budget_does_not_start_a_process(self):
        with patch('run_9b_round.subprocess.Popen') as process:
            with self.assertRaises(TimeoutError):
                run_bounded(['unused'], 0)
            process.assert_not_called()

    def test_trace_ignores_warmup_and_counts_reappearing_removed_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run.json').write_text(json.dumps({'status': 'completed', 'trace_diagnostic': True,
                'settings': {}, 'request_count': 2, 'client_summary': {'cached_tokens': 16, 'input_tokens': 64}}))
            events = [
                {'type': 'lookup', 'request_id': 'cmpl-measure-warmup-0'},
                {'type': 'cache_hash_removed', 'during_allocation': True, 'entries': [{'key': 'warm', 'group_id': 0}]},
                {'type': 'lookup', 'request_id': 'cmpl-measure-0'},
                {'type': 'cache_hash_removed', 'during_allocation': True, 'entries': [{'key': 'A', 'group_id': 1}]},
                {'type': 'cache_registration_observed', 'entries': [{'key': 'A', 'group_id': 1}]},
                {'type': 'cache_registration_observed', 'entries': [{'key': 'A', 'group_id': 1}]},
            ]
            (root / 'diagnostic-events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
            value = summarize(root)
            self.assertEqual(value['allocation_removed_hash_entries_by_group'], {'1': 1})
            self.assertEqual(value['removed_hash_registered_again_by_group'], {'1': 1})

    def test_freeze_rejects_client_concurrency_without_server_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'protocol.json').write_text('{}')
            (root / 'prepared').mkdir()
            for index in range(20):
                (root / f'prepared/{index}.jsonl').write_text('{}\n')
            settings = {'cache_budgets': {'high_pressure': 1, 'moderate_pressure': 2, 'roomy': 3},
                'request_rate': 2, 'native_cache_options': {'mamba_cache_mode': 'align'}, 'enforce_eager': False,
                'native_options_review': 'review fixture', 'rate_choice_reason': 'rate fixture'}
            (root / 'settings.json').write_text(json.dumps(settings))
            for budget in (1, 2, 3):
                folder = root / f'calibration/{budget}'
                folder.mkdir(parents=True)
                value = {'mode': 'calibration', 'status': 'completed', 'policy': 'stock', 'runtime_profile': {},
                    'protocol_sha256': sha256(root / 'protocol.json'), 'model_revision': 'fixture',
                    'settings': {'kv_cache_memory_bytes': budget, 'request_rate': 2, 'concurrency': 16,
                        'native_cache_options': settings['native_cache_options'], 'enforce_eager': False},
                    'client_summary': {'peak_client_active': 16, 'server_activity_metrics_available': True,
                        'max_server_running': 8, 'max_server_waiting': 0}}
                (folder / 'run.json').write_text(json.dumps(value))
            argv = ['freeze', '--protocol', str(root / 'protocol.json'), '--prepared', str(root / 'prepared'),
                '--calibration-root', str(root / 'calibration'), '--settings', str(root / 'settings.json'),
                '--output', str(root / 'lock.json')]
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'no observed server queue'):
                freeze()
            self.assertFalse((root / 'lock.json').exists())
            path = root / 'calibration/2/run.json'
            value = json.loads(path.read_text())
            value['client_summary']['max_server_waiting'] = 2
            path.write_text(json.dumps(value))
            with patch.object(sys, 'argv', argv):
                freeze()
            self.assertEqual(json.loads((root / 'lock.json').read_text())['status'], 'frozen')


class StreamingChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp import web
        self.web = web
        self.directory = tempfile.TemporaryDirectory()
        self.resets = 0
        self.allow_reset = False
        self.payloads = []
        app = web.Application()
        app.router.add_post('/reset_prefix_cache', self.reset)
        app.router.add_post('/v1/completions', self.complete)
        app.router.add_get('/metrics', self.metrics)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.directory.cleanup()

    async def reset(self, request):
        self.resets += 1
        return self.web.json_response({'success': self.allow_reset and self.resets >= 2})

    async def metrics(self, request):
        return self.web.Response(text='vllm:num_requests_running{model_name="thesis-9b"} 2\nvllm:num_requests_waiting{model_name="thesis-9b"} 1\n')

    async def complete(self, request):
        payload = await request.json()
        self.payloads.append(payload)
        response = self.web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        for token in range(payload['max_tokens']):
            value = {'choices': [{'token_ids': [token], 'text': 'x'}]}
            await response.write(('data: ' + json.dumps(value) + '\n\n').encode())
            await asyncio.sleep(.001)
        value = {'choices': [], 'usage': {'prompt_tokens': 2, 'completion_tokens': payload['max_tokens'],
                                          'prompt_tokens_details': {'cached_tokens': 0}}}
        await response.write(('data: ' + json.dumps(value) + '\n\ndata: [DONE]\n\n').encode())
        await response.write_eof()
        return response

    async def test_false_reset_never_starts_measurement(self):
        with self.assertRaisesRegex(RuntimeError, 'did not confirm'):
            await run_client(self.url, [input_row()], 1, 1, 2, Path(self.directory.name) / 'failed', warmup=False)
        self.assertEqual(self.resets, 5)
        self.assertEqual(self.payloads, [])

    async def test_reset_retry_and_no_reuse_payload_are_measured(self):
        self.allow_reset = True
        rows = apply_workload_variant([input_row(i) for i in range(4)], 'no_reuse')
        value = await run_client(self.url, rows, 2, 100, 2, Path(self.directory.name) / 'success', warmup=False)
        self.assertEqual(self.resets, 2)
        self.assertEqual(value['success_count'], 4)
        self.assertEqual(value['cached_tokens'], 0)
        self.assertEqual(len({item['cache_salt'] for item in self.payloads}), 4)
        self.assertTrue(value['server_activity_metrics_available'])
        self.assertGreater(value['max_server_waiting'], 0)
        self.assertGreater(value['duration_s'], 0)


if __name__ == '__main__':
    unittest.main()
