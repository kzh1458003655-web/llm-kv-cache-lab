import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_server_round import analyze


class AnalyzeServerRoundTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_run(self, name, status, rows, **summary_fields):
        run_dir = self.root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {"status": status, **summary_fields}
        (run_dir / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        return run_dir

    @staticmethod
    def row(request_id, prompt_hash, prompt_tokens, cached_tokens, elapsed_ms, output_hash):
        return {
            "request_id": request_id,
            "prompt_sha256": prompt_hash,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "elapsed_ms": elapsed_ms,
            "output_token_sha256": output_hash,
        }

    def test_completed_matched_pair_reports_cache_gain_elapsed_and_output_hashes(self):
        stock_rows = [
            self.row("r1", "p1", 100, 10, 20.0, "o1"),
            self.row("r2", "p2", 200, 20, 30.0, "o2"),
        ]
        reuse_rows = [
            self.row("r1", "p1", 100, 60, 18.0, "o1"),
            self.row("r2", "p2", 200, 70, 27.0, "different-output"),
        ]
        self.write_run("case-stock", "completed", stock_rows)
        self.write_run("case-reuse2", "completed", reuse_rows)

        result = analyze(self.root)

        self.assertEqual(len(result["pairs"]), 1)
        pair = result["pairs"][0]
        self.assertEqual(pair["stock"], "case-stock")
        self.assertEqual(pair["reuse2"], "case-reuse2")
        self.assertEqual(pair["cached_token_gain"], 100)
        self.assertAlmostEqual(pair["elapsed_change_percent"], -10.0)
        self.assertEqual(pair["output_token_hash_matches"], 1)
        self.assertEqual(pair["n"], 2)

    def test_partial_or_failed_run_is_not_in_matched_pairs(self):
        self.write_run(
            "case-stock", "completed",
            [self.row("r1", "p1", 100, 10, 20.0, "o1")],
        )
        self.write_run(
            "case-reuse2", "failed",
            [self.row("r1", "p1", 100, 60, 18.0, "o1")],
        )

        result = analyze(self.root)

        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["runs"]["case-stock"]["status"], "completed")
        self.assertEqual(result["runs"]["case-reuse2"]["status"], "failed")

    def test_manifest_mismatch_raises(self):
        self.write_run(
            "case-stock", "completed",
            [self.row("r1", "p1", 100, 10, 20.0, "o1")],
        )
        self.write_run(
            "case-reuse2", "completed",
            [self.row("r1", "different-prompt", 100, 60, 18.0, "o1")],
        )

        with self.assertRaisesRegex(ValueError, "manifests/tokens differ"):
            analyze(self.root)

    def test_empty_directory_returns_empty_runs_and_pairs(self):
        result = analyze(self.root)

        self.assertEqual(result["runs"], {})
        self.assertEqual(result["pairs"], [])
        self.assertTrue(result["limitations"])

    def test_completed_empty_run_is_invalid_and_not_paired(self):
        self.write_run("case-stock", "completed", [])
        self.write_run(
            "case-reuse2", "completed",
            [self.row("r1", "p1", 100, 60, 18.0, "o1")],
        )

        result = analyze(self.root)

        self.assertEqual(result["runs"]["case-stock"]["status"], "invalid_completed_artifact")
        self.assertEqual(result["pairs"], [])

    def test_completed_run_with_request_count_mismatch_is_invalid(self):
        self.write_run(
            "case-stock", "completed",
            [self.row("r1", "p1", 100, 10, 20.0, "o1")],
            request_count=2,
            success_count=1,
        )
        self.write_run(
            "case-reuse2", "completed",
            [self.row("r1", "p1", 100, 60, 18.0, "o1")],
        )

        result = analyze(self.root)

        self.assertEqual(result["runs"]["case-stock"]["status"], "invalid_completed_artifact")
        self.assertEqual(result["pairs"], [])


if __name__ == "__main__":
    unittest.main()
