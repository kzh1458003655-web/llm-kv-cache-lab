"""Serial synthetic smoke probe for hybrid-model prefix-cache availability.

This probe checks that repeated document prefixes can be served and records
vLLM cache metrics. It does not establish eviction loss or realistic workload
benefit; its prompts are synthetic mechanism-smoke inputs only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _paragraph(seed: str, number: int) -> str:
    """Produce varied, deterministic synthetic English prose."""
    subjects = [
        "The archive team", "A careful reviewer", "The catalog system",
        "Several local historians", "The preservation group", "A visiting student",
    ]
    actions = [
        "checks each record against a dated inventory",
        "compares handwritten notes with the public index",
        "stores a second copy before changing the description",
        "marks uncertain details for a later human review",
        "keeps the original wording beside a normalized summary",
        "records which room and shelf held the material",
    ]
    objects = [
        "maps, letters, and meeting minutes", "photographs from the river district",
        "shipping ledgers from the old station", "oral-history transcripts and maps",
        "field notebooks from the northern valley", "small publications from local clubs",
    ]
    consequences = [
        "so another reader can repeat the same search",
        "without treating an uncertain guess as a fact",
        "while keeping unrelated collections separate",
        "and without changing the source document",
        "before the material is used in a later report",
        "even when two entries share a similar title",
    ]
    i = number
    return (
        f"{subjects[i % len(subjects)]} {actions[(i + 1) % len(actions)]} "
        f"for {objects[(i + 2) % len(objects)]}. The note {seed}-{i + 1:02d} "
        f"describes a fictional collection assembled for this software check. "
        f"Staff preserve dates, labels, and source boundaries, {consequences[(i + 3) % len(consequences)]}. "
        "A short description is useful, but it does not replace the underlying "
        "record. When two sources disagree, the catalog keeps both statements "
        "and records where each one came from. The example contains no real "
        "person, institution, event, or archival item."
    )


def make_document(seed: str) -> str:
    # About 800 English words (roughly 1,000–1,500 model tokens, depending on
    # tokenizer). The script deliberately records the full text so this
    # synthetic input can be reproduced; token count is not asserted here.
    return "\n\n".join(_paragraph(seed, i) for i in range(14))


def make_requests() -> list[dict[str, str]]:
    doc_a = make_document("DOC-A")
    doc_b = make_document("DOC-B")
    specs = [
        ("docA-question1", doc_a, "Which materials are mentioned in the fictional collection?"),
        ("docA-question2", doc_a, "What does the catalog do when two sources disagree?"),
        ("docB-question1", doc_b, "Which materials are mentioned in the fictional collection?"),
        ("docA-question3", doc_a, "Why does the team preserve source boundaries?"),
    ]
    rows = []
    for request_id, document, question in specs:
        prompt = (
            "Read the following fictional archive description and answer the question briefly.\n\n"
            f"{document}\n\nQuestion: {question}\nAnswer:"
        )
        rows.append({
            "request_id": request_id,
            "document_id": "docA" if request_id.startswith("docA-") else "docB",
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "question": question,
        })
    return rows


def _metrics_snapshot(base_url: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/metrics"
    request = Request(url, headers={"Accept": "text/plain"}, method="GET")
    try:
        with urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
        lines = [
            line for line in body.splitlines()
            if not line.lstrip().startswith("#")
            and re.search(r"vllm", line, re.IGNORECASE)
            and re.search(r"prefix|cache|preemption", line, re.IGNORECASE)
        ]
        return {"status": "ok", "http_status": status, "url": url, "lines": lines}
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return {
            "status": "error", "url": url,
            "error": f"{type(exc).__name__}: {exc}", "lines": [],
        }


def _send_one(base_url: str, model: str, row: dict[str, str]) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": row["prompt"],
        "temperature": 0,
        "max_tokens": 16,
        "stream": False,
    }
    request = Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    start = time.perf_counter_ns()
    result: dict[str, Any] = {
        **row,
        "status": "error",
        "http_status": None,
        "elapsed_ms": None,
        "usage": None,
        "cached_usage_fields": {},
        "output_text": None,
        "error": None,
    }
    try:
        with urlopen(request, timeout=180) as response:
            result["http_status"] = response.status
            body = response.read().decode("utf-8", errors="replace")
        decoded = json.loads(body)
        choices = decoded.get("choices") or []
        result["usage"] = decoded.get("usage")
        usage = result["usage"] or {}
        result["cached_usage_fields"] = {
            key: value for key, value in usage.items()
            if "cached" in key.lower()
        }
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            result["cached_usage_fields"].update({
                f"prompt_tokens_details.{key}": value
                for key, value in details.items() if "cached" in key.lower()
            })
        result["output_text"] = "".join(
            str(choice.get("text") or "") for choice in choices
        )
        result["response_id"] = decoded.get("id")
        result["status"] = "ok"
    except HTTPError as exc:
        result["http_status"] = exc.code
        try:
            result["error"] = exc.read().decode("utf-8", errors="replace")[:4000]
        except OSError:
            result["error"] = f"HTTPError: {exc}"
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # Keep unexpected per-request failures in the trace.
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_ms"] = (time.perf_counter_ns() - start) / 1_000_000
    return result


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    requests = make_requests()
    rows: list[dict[str, Any]] = []
    for request_row in requests:
        before = _metrics_snapshot(args.base_url)
        response = _send_one(args.base_url, args.model, request_row)
        after = _metrics_snapshot(args.base_url)
        rows.append({
            **response,
            "metrics_before": before,
            "metrics_after": after,
            "metrics_note": "Raw filtered snapshots only; this probe does not infer eviction or causality.",
        })
        # Persist each request immediately so a later failure cannot erase earlier evidence.
        with (output_dir / f"{request_row['request_id']}.json").open("x", encoding="utf-8") as stream:
            json.dump(rows[-1], stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    successes = sum(row["status"] == "ok" for row in rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "request_order": [row["request_id"] for row in rows],
        "request_count": len(rows),
        "success_count": successes,
        "error_count": len(rows) - successes,
        "elapsed_ms_is": "end-to-end HTTP request duration; not TTFT",
        "workload": "synthetic mechanism smoke: one repeated fictional document with distinct questions plus one different fictional document",
        "interpretation": "The first-to-later shared prefix checks cache availability only. This does not establish eviction loss, improvement potential, or realistic workload behavior.",
        "output_files": [f"{row['request_id']}.json" for row in rows],
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if successes == len(rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", default="hybrid-pilot")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
