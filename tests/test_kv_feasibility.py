import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from scripts.kv_feasibility import (
    build_manifest,
    calculate_counter_deltas,
    choice_event_metadata,
    _get_url,
    parse_prometheus_counters,
    request_order,
    resolve_first_token_timestamp,
    summarize_run,
)


class FakeTokenizer:
    """A deterministic whitespace tokenizer for CPU-only workload tests."""

    def __init__(self):
        self.token_ids = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        result = []
        for token in text.split():
            if token not in self.token_ids:
                self.token_ids[token] = len(self.token_ids) + 1
            result.append(self.token_ids[token])
        return result


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        del format, args


class KVFeasibilityTests(unittest.TestCase):
    def test_empty_text_choice_is_preserved_as_safe_metadata(self):
        metadata = choice_event_metadata({"text": "", "finish_reason": "length"})

        self.assertEqual(metadata["text_length"], 0)
        self.assertTrue(metadata["text_empty"])
        self.assertEqual(metadata["finish_reason"], "length")
        self.assertNotIn("text", metadata)

    def test_first_choice_chunk_counts_as_token_when_usage_confirms_one(self):
        timestamp, source = resolve_first_token_timestamp(
            first_choice_ns=123,
            first_nonempty_text_ns=None,
            output_tokens=1,
            finish_reason="length",
        )

        self.assertEqual(timestamp, 123)
        self.assertEqual(source, "first_choice_chunk")

    def test_zero_output_tokens_does_not_get_a_synthetic_ttft(self):
        with self.assertRaisesRegex(ValueError, "no generated token"):
            resolve_first_token_timestamp(
                first_choice_ns=123,
                first_nonempty_text_ns=None,
                output_tokens=0,
                finish_reason="stop",
            )

    def test_local_http_opener_disables_all_proxies(self):
        server = HTTPServer(("127.0.0.1", 0), HealthHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            proxy_environment = {
                "HTTP_PROXY": "http://127.0.0.1:1",
                "http_proxy": "http://127.0.0.1:1",
                "NO_PROXY": "",
                "no_proxy": "",
            }
            with patch.dict(os.environ, proxy_environment):
                status, body = _get_url(f"http://127.0.0.1:{server.server_port}/health")
            self.assertEqual(status, 200)
            self.assertEqual(body, b"ok")
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_manifest_is_reproducible_and_has_ordered_workload(self):
        first = build_manifest(FakeTokenizer(), cold_count=14, seed=20260927)
        second = build_manifest(FakeTokenizer(), cold_count=14, seed=20260927)

        self.assertEqual(first, second)
        self.assertEqual([row["name"] for row in first["requests"]], request_order(14))
        self.assertEqual(len(first["requests"]), 18)
        hot = [row["prompt"] for row in first["requests"] if row["name"].startswith("hot_")]
        self.assertEqual(len(set(hot)), 1)
        self.assertTrue(all(352 <= row["token_count"] <= 416 for row in first["requests"]))
        self.assertTrue(all(row["token_count"] <= 512 for row in first["requests"]))
        self.assertTrue(all(len(row["sha256"]) == 64 for row in first["requests"]))

    def test_cold_prompts_have_distinct_first_cache_blocks(self):
        tokenizer = FakeTokenizer()
        manifest = build_manifest(tokenizer, cold_count=14, seed=7)
        cold = [row for row in manifest["requests"] if row["name"].startswith("cold_")]
        first_blocks = [tuple(tokenizer.encode(row["prompt"])[:16]) for row in cold]

        self.assertTrue(all(len(block) == 16 for block in first_blocks))
        self.assertEqual(len(first_blocks), len(set(first_blocks)))

    def test_request_order_supports_expanded_scan(self):
        order = request_order(24)
        self.assertEqual(len(order), 28)
        self.assertEqual(order[0:2], ["hot_first", "hot_immediate"])
        self.assertEqual(order[2], "cold_00")
        self.assertEqual(order[25], "cold_23")
        self.assertEqual(order[-2:], ["hot_after_scan", "hot_repeated_again"])

    def test_prometheus_parser_sums_labelled_series_and_total_suffix(self):
        text = """
        # HELP vllm:prefix_cache_hits Prefix hits
        vllm:prefix_cache_hits{model_name="a"} 11
        vllm:prefix_cache_hits_total{model_name="b"} 7
        vllm:prefix_cache_queries{model_name="a"} 40
        vllm:prompt_tokens 50
        vllm:prompt_tokens_cached 18
        """
        parsed = parse_prometheus_counters(text)

        self.assertEqual(parsed["vllm:prefix_cache_hits"], 18)
        self.assertEqual(parsed["vllm:prefix_cache_queries"], 40)
        self.assertEqual(parsed["vllm:prompt_tokens"], 50)
        self.assertEqual(parsed["vllm:prompt_tokens_cached"], 18)

    def test_counter_deltas_reject_missing_and_decreasing_counters(self):
        before = {"vllm:prefix_cache_hits": 8}
        after = {"vllm:prefix_cache_hits": 12}
        self.assertEqual(
            calculate_counter_deltas(before, after, ["vllm:prefix_cache_hits"]),
            {"vllm:prefix_cache_hits": 4},
        )
        with self.assertRaisesRegex(ValueError, "[Mm]issing"):
            calculate_counter_deltas(before, {}, ["vllm:prefix_cache_hits"])
        with self.assertRaisesRegex(ValueError, "decreased"):
            calculate_counter_deltas(after, before, ["vllm:prefix_cache_hits"])

    def test_summary_uses_token_hit_fraction_and_ideal_retention_formulas(self):
        rows = [
            {"name": "hot_after_scan", "status": "ok", "ttft_ms": 120.0, "elapsed_ms": 150.0, "metrics_delta": {"vllm:prefix_cache_hits": 20, "vllm:prefix_cache_queries": 100}},
            {"name": "hot_repeated_again", "status": "ok", "ttft_ms": 70.0, "elapsed_ms": 90.0, "metrics_delta": {"vllm:prefix_cache_hits": 80, "vllm:prefix_cache_queries": 100}},
            {"name": "other", "status": "ok", "ttft_ms": 60.0, "elapsed_ms": 75.0, "metrics_delta": {"vllm:prefix_cache_hits": 5, "vllm:prefix_cache_queries": 50}},
        ]
        summary = summarize_run(rows)

        self.assertEqual(summary["paired_ttft_gap_ms_signed"], 50.0)
        self.assertEqual(summary["ideal_retention_proxy_ms"], 50.0)
        self.assertAlmostEqual(summary["affected_hot_request_fraction"], 50 / 120)
        self.assertAlmostEqual(summary["full_mix_ttft_fraction"], 50 / 250)
        self.assertEqual(summary["total_ttft_ms"], 250.0)
        self.assertAlmostEqual(summary["hit_fraction_by_request"]["hot_after_scan"], 0.2)

    def test_summary_preserves_negative_signed_gap_and_clamps_proxy(self):
        rows = [
            {"name": "hot_after_scan", "status": "ok", "ttft_ms": 60.0, "elapsed_ms": 70.0, "metrics_delta": {"vllm:prefix_cache_hits": 0, "vllm:prefix_cache_queries": 100}},
            {"name": "hot_repeated_again", "status": "ok", "ttft_ms": 80.0, "elapsed_ms": 90.0, "metrics_delta": {"vllm:prefix_cache_hits": 90, "vllm:prefix_cache_queries": 100}},
        ]
        summary = summarize_run(rows)

        self.assertEqual(summary["paired_ttft_gap_ms_signed"], -20.0)
        self.assertEqual(summary["ideal_retention_proxy_ms"], 0.0)
        self.assertEqual(summary["affected_hot_request_fraction"], 0.0)
        self.assertEqual(summary["full_mix_ttft_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
