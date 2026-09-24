import json
import tempfile
import unittest
from pathlib import Path

from kv_cache_lab.events import CacheEvent, EventType, read_events, write_events


class CacheEventTests(unittest.TestCase):
    def test_event_round_trip(self) -> None:
        event = CacheEvent(
            timestamp_ns=123,
            event_type=EventType.PREFIX_HIT,
            request_id="req-1",
            block_id=7,
            tokens=16,
        )
        self.assertEqual(CacheEvent.from_dict(event.to_dict()), event)

    def test_rejects_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            CacheEvent(timestamp_ns=-1, event_type=EventType.BLOCK_ALLOCATED)
        with self.assertRaises(ValueError):
            CacheEvent(timestamp_ns=1, event_type=EventType.BLOCK_ALLOCATED, block_id=-1)
        with self.assertRaises(ValueError):
            CacheEvent(timestamp_ns=1, event_type=EventType.PREFIX_HIT, tokens=-1)

    def test_rejects_unknown_event_type(self) -> None:
        with self.assertRaises(ValueError):
            CacheEvent.from_dict({"timestamp_ns": 1, "event_type": "unknown"})

    def test_ndjson_append_and_read(self) -> None:
        events = [
            CacheEvent(timestamp_ns=1, event_type=EventType.REQUEST_STARTED, request_id="a"),
            CacheEvent(timestamp_ns=2, event_type=EventType.BLOCK_ALLOCATED, request_id="a", block_id=0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            write_events(path, events[:1])
            write_events(path, events[1:])
            self.assertEqual(read_events(path), events)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_read_reports_bad_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(json.dumps({"event_type": "bad"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                read_events(path)


if __name__ == "__main__":
    unittest.main()
