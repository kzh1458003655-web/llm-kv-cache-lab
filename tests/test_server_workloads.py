import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.server_workloads import build_workload


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


class ServerWorkloadTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source = Path(self.temp_dir.name) / "shortdep_qa.jsonl"
        self.tokenizer = FakeTokenizer()
        records = []
        # Ten documents with twenty unique questions each provide enough rows
        # for 128 no-replacement requests and multiple hot-scan segments.
        for doc_number in range(10):
            for question_number in range(20):
                records.append({
                    "id": f"record-{doc_number}-{question_number}",
                    "doc_id": f"doc-{doc_number}",
                    "context": f"Context for document {doc_number}. " * 5,
                    "question": f"Question {question_number} for document {doc_number}?",
                })
        self.source.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def build(self, scene, seed=20260930, count=128):
        return build_workload(
            self.source,
            self.tokenizer,
            scene=scene,
            seed=seed,
            count=count,
            context_tokens=80,
        )

    def assert_unique_rows(self, rows, expected_count):
        self.assertEqual(len(rows), expected_count)
        self.assertEqual(len({row["request_id"] for row in rows}), expected_count)
        self.assertEqual(len({row["dataset_record_id"] for row in rows}), expected_count)
        self.assertEqual(len({row["question_sha256"] for row in rows}), expected_count)
        self.assertEqual(len({row["prompt"] for row in rows}), expected_count)
        self.assertEqual(len({row["prompt_sha256"] for row in rows}), expected_count)
        for row in rows:
            self.assertEqual(
                row["prompt_sha256"],
                hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
            )
            self.assertEqual(
                row["expected_prompt_tokens"],
                len(self.tokenizer.encode(row["prompt"], add_special_tokens=False)),
            )

    def test_hot_scan_128_has_unique_ids_and_expected_segments(self):
        rows, metadata = self.build("hot_scan", count=128)
        self.assert_unique_rows(rows, 128)
        self.assertEqual(metadata["scene"], "hot_scan")
        self.assertEqual(
            [row["request_id"] for row in rows[:8]],
            [
                "segment000-warm1", "segment000-warm2", "segment000-warm3",
                "segment000-scan0", "segment000-scan1", "segment000-scan2",
                "segment000-hot_return", "segment000-hot_verify",
            ],
        )
        self.assertEqual(rows[0]["doc_id"], rows[1]["doc_id"])
        self.assertNotEqual(rows[0]["doc_id"], rows[3]["doc_id"])
        self.assertEqual(rows[0]["doc_id"], rows[6]["doc_id"])

    def test_zipf_scenes_return_128_unique_rows(self):
        for scene in ("mixed_zipf", "roomy_control"):
            with self.subTest(scene=scene):
                rows, metadata = self.build(scene, count=128)
                self.assert_unique_rows(rows, 128)
                self.assertEqual(metadata["scene"], scene)

    def test_seed_reproduces_each_scene_and_changes_selection(self):
        for scene in ("hot_scan", "mixed_zipf", "roomy_control"):
            with self.subTest(scene=scene):
                rows_a, _ = self.build(scene, seed=17, count=16 if scene == "hot_scan" else 128)
                rows_b, _ = self.build(scene, seed=17, count=16 if scene == "hot_scan" else 128)
                rows_other, _ = self.build(scene, seed=18, count=16 if scene == "hot_scan" else 128)
                self.assertEqual(
                    [(r["request_id"], r["doc_id"], r["dataset_record_id"], r["prompt"]) for r in rows_a],
                    [(r["request_id"], r["doc_id"], r["dataset_record_id"], r["prompt"]) for r in rows_b],
                )
                self.assertNotEqual(
                    [(r["doc_id"], r["dataset_record_id"]) for r in rows_a],
                    [(r["doc_id"], r["dataset_record_id"]) for r in rows_other],
                )

    def test_mixed_zipf_and_roomy_control_match_for_same_seed(self):
        mixed, _ = self.build("mixed_zipf", seed=123, count=128)
        roomy, _ = self.build("roomy_control", seed=123, count=128)
        self.assertEqual([row["prompt"] for row in mixed], [row["prompt"] for row in roomy])

    def test_shortage_raises_instead_of_repeating_records(self):
        tiny = Path(self.temp_dir.name) / "tiny.jsonl"
        tiny.write_text(
            json.dumps({
                "id": "only-one", "doc_id": "only-doc", "context": "context",
                "question": "one question?",
            }) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            build_workload(tiny, self.tokenizer, scene="mixed_zipf", count=2)


if __name__ == "__main__":
    unittest.main()
