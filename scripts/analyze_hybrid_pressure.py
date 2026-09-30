"""Analyze one completed synthetic hybrid cache-pressure run offline.

This mechanism trace is synthetic and does not establish realistic workload
behavior or optimization benefit. The analysis does not compute E/P and does
not treat cache misses alone as eviction evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUEST_FIELDS = (
    "engine_request_id",
    "num_cached_tokens",
    "prompt_tokens",
    "output_text",
    "elapsed_ms",
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(event, dict):
                    raise ValueError(f"event at {path}:{line_number} is not an object")
                events.append(event)
    except OSError as exc:
        raise ValueError(f"cannot read events file {path}: {exc}") from exc
    return events


def analyze(run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    if summary.get("completed") is not True and summary.get("status") != "completed":
        raise ValueError("summary does not mark this run completed")

    order = summary.get("request_order")
    if not isinstance(order, list) or len(order) != 8 or any(not isinstance(x, str) for x in order):
        raise ValueError("summary.request_order must contain the eight request names")
    required_names = {"warm1", "warm2", "warm3", "scan0", "scan1", "scan2", "hot_return", "hot_verify"}
    if set(order) != required_names:
        raise ValueError("summary.request_order does not contain the expected pressure sequence")
    if order.index("warm3") >= order.index("hot_return"):
        raise ValueError("warm3 must precede hot_return in summary.request_order")

    requests: list[dict[str, Any]] = []
    request_by_name: dict[str, dict[str, Any]] = {}
    for name in order:
        row = load_json(run_dir / f"{name}.json")
        missing = [field for field in REQUEST_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{name}.json is missing required fields: {', '.join(missing)}")
        row["name"] = name
        request_by_name[name] = row
        requests.append({field: row[field] for field in ("name", *REQUEST_FIELDS)})

    events = read_events(run_dir / "events.jsonl")
    lookup_ids: list[tuple[str, int]] = []
    for index, event in enumerate(events):
        if event.get("type") == "lookup" and isinstance(event.get("request_id"), str):
            lookup_ids.append((event["request_id"], index))

    def lookup_index(engine_request_id: Any, name: str) -> int:
        if not isinstance(engine_request_id, (str, int)):
            raise ValueError(f"{name} engine_request_id must be a string or integer")
        identifier = str(engine_request_id)
        exact = [index for event_id, index in lookup_ids if event_id == identifier]
        # vLLM's scheduler may append an internal suffix (for example, `2-ab12`)
        # to the API-side engine request id (`2`). Accept that only when unique.
        matches = exact or [
            index for event_id, index in lookup_ids
            if event_id.startswith(identifier + "-")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"events.jsonl has no unique {name} lookup matching engine_request_id {identifier!r}"
            )
        return matches[0]

    warm3_id = request_by_name["warm3"]["engine_request_id"]
    hot_return_id = request_by_name["hot_return"]["engine_request_id"]
    warm3_index = lookup_index(warm3_id, "warm3")
    hot_return_index = lookup_index(hot_return_id, "hot_return")
    if warm3_index >= hot_return_index:
        raise ValueError("events warm3 lookup must precede hot_return lookup")

    warm3_event = events[warm3_index]
    hit_keys = warm3_event.get("hit_keys")
    if not isinstance(hit_keys, list) or any(not isinstance(key, str) for key in hit_keys):
        raise ValueError("warm3 lookup is missing a valid hit_keys list")
    hot_key_set = set(hit_keys)

    removed_hot_keys: list[dict[str, Any]] = []
    for event in events[warm3_index + 1 : hot_return_index]:
        if event.get("type") != "cache_hash_removed" or event.get("during_allocation") is not True:
            continue
        entries = event.get("entries")
        if not isinstance(entries, list):
            raise ValueError("cache_hash_removed event is missing entries list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("cache_hash_removed entry is not an object")
            key = entry.get("key")
            if not isinstance(key, str):
                raise ValueError("cache_hash_removed entry is missing a string key")
            if key in hot_key_set:
                group_id = entry.get("group_id", event.get("group_id"))
                if group_id is None or "ref_count_before" not in event:
                    raise ValueError("matching removal is missing group_id or ref_count_before")
                removed_hot_keys.append({
                    "group_id": group_id,
                    "key": key,
                    "ref_count_before": event["ref_count_before"],
                })

    hot_return_cached_tokens = request_by_name["hot_return"]["num_cached_tokens"]
    return {
        "status": "analyzed",
        "workload_note": ("Public LooGLE excerpts with curated hot/scan arrivals; not production traffic or original QA evaluation."
                          if summary.get("workload") == "loogle" else
                          "Synthetic mechanism pressure only; this does not prove real-world performance or optimization benefit."),
        "interpretation_note": "The removal list contains only keys observed in warm3.lookup.hit_keys that were removed during allocation between the warm3 and hot_return lookup events. This does not compute E/P; misses alone are not counted as evictions.",
        "request_order": order,
        "requests": requests,
        "hot_return_num_cached_tokens": hot_return_cached_tokens,
        "hot_keys_removed_during_allocation_before_hot_return_lookup": removed_hot_keys,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.run_dir)
        output = args.run_dir / "analysis.json"
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        parser.exit(1, f"analysis failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
