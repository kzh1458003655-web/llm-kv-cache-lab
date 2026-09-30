import contextlib
import io
import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from scripts import run_server_round


class RunServerRoundTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.output = self.root / "round-output"
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def invoke(self, *, repeats=1, minutes=45):
        argv = [
            "run_server_round.py",
            "--model", str(self.root / "model"),
            "--prepared", str(self.prepared),
            "--output", str(self.output),
            "--minutes", str(minutes),
            "--repeats", str(repeats),
        ]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            run_server_round.main()

    @staticmethod
    def completed_process(pid, return_code=0):
        process = Mock()
        process.pid = pid
        process.wait.return_value = return_code
        return process

    def test_timeout_kills_only_owned_process_group(self):
        process = self.completed_process(43121)
        process.wait.side_effect = [
            subprocess.TimeoutExpired("benchmark", 1),
            subprocess.TimeoutExpired("benchmark", 10),
            0,
        ]
        with patch("scripts.run_server_round.subprocess.Popen", return_value=process) as popen, \
             patch("scripts.run_server_round.os.killpg", create=True) as killpg, \
             patch("scripts.run_server_round.signal.SIGKILL", 9, create=True), \
             patch("scripts.run_server_round.subprocess.run"):
            self.invoke(repeats=1, minutes=7)

        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(
            killpg.call_args_list,
            [call(process.pid, signal.SIGTERM), call(process.pid, 9)],
        )
        self.assertEqual(process.wait.call_count, 3)
        outcomes = json.loads((self.output / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(outcomes[0]["status"], "interrupted_or_timeout")

    def test_nonzero_exit_stops_later_runs(self):
        process = self.completed_process(43122, return_code=9)
        with patch("scripts.run_server_round.subprocess.Popen", return_value=process) as popen, \
             patch("scripts.run_server_round.subprocess.run") as run:
            self.invoke(repeats=1)

        self.assertEqual(popen.call_count, 1)
        outcomes = json.loads((self.output / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["exit_code"], 9)
        self.assertEqual(run.call_count, 1)  # Final offline analyzer only.

    def test_each_stock_reuse_pair_uses_same_manifest(self):
        processes = [self.completed_process(44000 + index) for index in range(6)]
        with patch("scripts.run_server_round.subprocess.Popen", side_effect=processes) as popen, \
             patch("scripts.run_server_round.subprocess.run"):
            self.invoke(repeats=1)

        self.assertEqual(popen.call_count, 6)
        plan = json.loads((self.output / "plan.json").read_text(encoding="utf-8"))
        pairs = {}
        for item in plan:
            key = (item["rep"], item["scene"], item["kv_mib"])
            pairs.setdefault(key, {})[item["policy"]] = item["manifest"]
        self.assertEqual(len(pairs), 3)
        for manifests in pairs.values():
            self.assertEqual(set(manifests), {"stock", "reuse2"})
            self.assertEqual(manifests["stock"], manifests["reuse2"])

        # The actual benchmark invocations consume the manifest associated
        # with their planned pair, not merely a matching plan annotation.
        invoked_manifests = []
        for call_args in popen.call_args_list:
            command = call_args.args[0]
            invoked_manifests.append(command[command.index("--manifest") + 1])
        self.assertEqual(
            invoked_manifests,
            [str(self.prepared / item["manifest"]) for item in plan],
        )

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        with patch("scripts.run_server_round.subprocess.Popen") as popen, \
             patch("scripts.run_server_round.subprocess.run"):
            with self.assertRaises(FileExistsError):
                self.invoke(repeats=1)

        popen.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
