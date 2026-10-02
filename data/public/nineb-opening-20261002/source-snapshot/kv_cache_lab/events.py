"""A small, engine-independent event format for future vLLM instrumentation."""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


class EventType(str, Enum):
    REQUEST_STARTED = "request_started"
    BLOCK_ALLOCATED = "block_allocated"
    BLOCK_RELEASED = "block_released"
    PREFIX_HIT = "prefix_hit"
    BLOCK_EVICTED = "block_evicted"
    REQUEST_FINISHED = "request_finished"


@dataclass(frozen=True)
class CacheEvent:
    timestamp_ns: int
    event_type: EventType
    request_id: str | None = None
    block_id: int | None = None
    tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("timestamp_ns", "block_id", "tokens"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "event_type": self.event_type.value,
            "request_id": self.request_id,
            "block_id": self.block_id,
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CacheEvent":
        return cls(
            timestamp_ns=data["timestamp_ns"],
            event_type=EventType(data["event_type"]),
            request_id=data.get("request_id"),
            block_id=data.get("block_id"),
            tokens=data.get("tokens"),
        )


def write_events(path: str | Path, events: Iterable[CacheEvent]) -> None:
    """Append events as newline-delimited JSON without replacing prior runs."""
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        for event in events:
            stream.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


def read_events(path: str | Path) -> list[CacheEvent]:
    """Read a single event file and report the line of malformed records."""
    result: list[CacheEvent] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                result.append(CacheEvent.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid cache event on line {line_number}") from error
    return result
