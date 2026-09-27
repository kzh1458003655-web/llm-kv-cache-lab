"""Local, repeatable HTTP feasibility probe for vLLM prefix-cache pressure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


COUNTER_NAMES = (
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:prompt_tokens_cached",
    "vllm:prompt_tokens",
)
MIN_PROMPT_TOKENS = 352
MAX_PROMPT_TOKENS = 416
MAX_ALLOWED_TOKENS = 512
FILLER = "这是用于本地推理缓存可行性评估的中文技术文档片段。 "
PROMETHEUS_SAMPLE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[+-]?Inf|NaN)(?:\s+\S+)?\s*$"
)


def request_order(cold_count: int) -> list[str]:
    if cold_count < 1:
        raise ValueError("cold_count must be at least 1")
    return [
        "hot_first",
        "hot_immediate",
        *(f"cold_{index:02d}" for index in range(cold_count)),
        "hot_after_scan",
        "hot_repeated_again",
    ]


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _make_prompt(tokenizer: Any, identifier: str, target_tokens: int) -> tuple[str, list[int]]:
    prompt = f"{identifier}\n"
    token_ids = _encode(tokenizer, prompt)
    # Grow in short increments so the final token count stays inside the fixed range.
    while len(token_ids) < target_tokens:
        prompt += FILLER
        token_ids = _encode(tokenizer, prompt)
        if len(token_ids) > MAX_PROMPT_TOKENS:
            raise ValueError(
                f"Cannot build a prompt in {MIN_PROMPT_TOKENS}–{MAX_PROMPT_TOKENS} tokens; "
                f"identifier={identifier!r}, got {len(token_ids)}"
            )
    if not MIN_PROMPT_TOKENS <= len(token_ids) <= MAX_PROMPT_TOKENS:
        raise ValueError(f"Prompt length outside allowed range: {len(token_ids)}")
    if len(token_ids) > MAX_ALLOWED_TOKENS:
        raise ValueError(f"Prompt exceeds {MAX_ALLOWED_TOKENS} tokens")
    return prompt, token_ids


def build_manifest(tokenizer: Any, cold_count: int = 14, seed: int = 20260927) -> dict[str, Any]:
    if not 1 <= cold_count <= 100:
        raise ValueError("cold_count must be between 1 and 100")
    rng = random.Random(seed)
    prompts: dict[str, tuple[str, list[int]]] = {}
    prompts["hot"] = _make_prompt(tokenizer, "HOT_SHARED_PREFIX_20260927", 384)
    for index in range(cold_count):
        target = rng.randint(MIN_PROMPT_TOKENS, 400)
        prompts[f"cold_{index:02d}"] = _make_prompt(
            tokenizer, f"UNIQUE_COLD_PREFIX_{index:03d}_SEED_{seed}", target
        )

    cold_blocks = [tuple(prompts[f"cold_{i:02d}"][1][:16]) for i in range(cold_count)]
    if any(len(block) < 16 for block in cold_blocks):
        raise ValueError("Every cold prompt must have at least 16 tokens")
    if len(set(cold_blocks)) != len(cold_blocks):
        raise ValueError("Cold prompts share a first 16-token cache block")

    requests = []
    for name in request_order(cold_count):
        prompt_key = "hot" if name.startswith("hot_") else name
        prompt, token_ids = prompts[prompt_key]
        requests.append(
            {
                "name": name,
                "prompt": prompt,
                "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "token_count": len(token_ids),
            }
        )
    return {
        "schema_version": 1,
        "seed": seed,
        "cold_count": cold_count,
        "requests": requests,
    }


def parse_prometheus_counters(text: str) -> dict[str, int]:
    values = {name: 0 for name in COUNTER_NAMES}
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_SAMPLE.match(line)
        if not match:
            continue
        metric_name, raw_value = match.groups()
        canonical_name = metric_name[:-6] if metric_name.endswith("_total") else metric_name
        if canonical_name not in values:
            continue
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid counter value for {metric_name}: {raw_value}")
        values[canonical_name] += int(value)
        seen.add(canonical_name)
    missing = sorted(set(COUNTER_NAMES) - seen)
    if missing:
        raise ValueError("Missing required Prometheus counters: " + ", ".join(missing))
    return values


def calculate_counter_deltas(
    before: dict[str, int], after: dict[str, int], required: Iterable[str] = COUNTER_NAMES
) -> dict[str, int]:
    missing = [name for name in required if name not in before or name not in after]
    if missing:
        raise ValueError("Missing required counters: " + ", ".join(missing))
    deltas = {name: after[name] - before[name] for name in required}
    decreased = {name: delta for name, delta in deltas.items() if delta < 0}
    if decreased:
        raise ValueError("Counters decreased during request: " + repr(decreased))
    return deltas


def choice_event_metadata(choice: dict[str, Any]) -> dict[str, Any]:
    """Return non-content SSE metadata suitable for request logs."""
    text = choice.get("text") or ""
    return {
        "text_length": len(text),
        "text_empty": not bool(text),
        "finish_reason": choice.get("finish_reason"),
    }


def resolve_first_token_timestamp(
    first_choice_ns: int | None,
    first_nonempty_text_ns: int | None,
    output_tokens: int | None,
    finish_reason: str | None,
) -> tuple[int, str]:
    if output_tokens == 0:
        raise ValueError("response contained no generated token")
    if first_choice_ns is not None and (output_tokens is not None or finish_reason == "length"):
        return first_choice_ns, "first_choice_chunk"
    if first_nonempty_text_ns is not None:
        return first_nonempty_text_ns, "first_nonempty_text_chunk"
    raise ValueError("stream ended without evidence of a generated token")


def summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["name"]: row for row in rows if row.get("status") == "ok"}
    missing = [name for name in ("hot_after_scan", "hot_repeated_again") if name not in by_name]
    if missing:
        raise ValueError("Cannot summarize run; missing successful requests: " + ", ".join(missing))
    ttft_by_request = {
        row["name"]: float(row["ttft_ms"])
        for row in rows
        if row.get("status") == "ok" and row.get("ttft_ms") is not None
    }
    hit_fraction_by_request: dict[str, float | None] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        delta = row.get("metrics_delta", {})
        queries = int(delta.get("vllm:prefix_cache_queries", 0))
        hits = int(delta.get("vllm:prefix_cache_hits", 0))
        hit_fraction_by_request[row["name"]] = hits / queries if queries else None

    after_scan = float(by_name["hot_after_scan"]["ttft_ms"])
    final_warm = float(by_name["hot_repeated_again"]["ttft_ms"])
    signed_gap = after_scan - final_warm
    proxy = max(0.0, signed_gap)
    total_ttft = sum(ttft_by_request.values())
    return {
        "request_count": len([row for row in rows if row.get("status") == "ok"]),
        "failed_requests": [row.get("name", "unknown") for row in rows if row.get("status") != "ok"],
        "ttft_ms_by_request": ttft_by_request,
        "elapsed_ms_by_request": {
            row["name"]: float(row["elapsed_ms"])
            for row in rows
            if row.get("status") == "ok" and row.get("elapsed_ms") is not None
        },
        "hit_fraction_by_request": hit_fraction_by_request,
        "total_ttft_ms": total_ttft,
        "paired_ttft_gap_ms_signed": signed_gap,
        "ideal_retention_proxy_ms": proxy,
        "affected_hot_request_fraction": proxy / after_scan if after_scan > 0 else None,
        "full_mix_ttft_fraction": proxy / total_ttft if total_ttft > 0 else None,
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_tokenizer(model_path: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for prepare") from exc
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def command_prepare(args: argparse.Namespace) -> int:
    tokenizer = _load_tokenizer(args.model)
    manifest = build_manifest(tokenizer, cold_count=args.cold_count, seed=args.seed)
    manifest["model_path"] = args.model
    _write_json(Path(args.output), manifest)
    lengths = [row["token_count"] for row in manifest["requests"]]
    print(
        json.dumps(
            {
                "output": args.output,
                "request_count": len(manifest["requests"]),
                "cold_count": args.cold_count,
                "token_count_min": min(lengths),
                "token_count_max": max(lengths),
                "seed": args.seed,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _get_url(url: str, timeout: float = 10.0) -> tuple[int, bytes]:
    with _local_opener().open(url, timeout=timeout) as response:
        return response.status, response.read()


def _local_opener() -> urllib.request.OpenerDirector:
    """Build an opener that never sends localhost traffic through configured proxies."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _scrape_metrics(base_url: str) -> dict[str, int]:
    _, body = _get_url(base_url.rstrip("/") + "/metrics")
    return parse_prometheus_counters(body.decode("utf-8", errors="replace"))


def _consume_completion(base_url: str, model: str, prompt: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    start_ns = time.monotonic_ns()
    first_choice_ns: int | None = None
    first_nonempty_text_ns: int | None = None
    first_choice_metadata: dict[str, Any] | None = None
    choice_chunk_count = 0
    final_finish_reason: str | None = None
    status: int | None = None
    usage: dict[str, Any] | None = None
    output_tokens: int | None = None
    try:
        with _local_opener().open(request, timeout=180) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError("vLLM stream error: " + json.dumps(event["error"])[:1000])
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices:
                    for choice in choices:
                        choice_chunk_count += 1
                        choice_ns = time.monotonic_ns()
                        metadata = choice_event_metadata(choice)
                        if first_choice_ns is None:
                            first_choice_ns = choice_ns
                            first_choice_metadata = metadata
                        if first_nonempty_text_ns is None and not metadata["text_empty"]:
                            first_nonempty_text_ns = choice_ns
                        if metadata["finish_reason"] is not None:
                            final_finish_reason = metadata["finish_reason"]
                    if final_finish_reason is not None:
                        output_tokens = (usage or {}).get("completion_tokens")
        end_ns = time.monotonic_ns()
    except urllib.error.HTTPError as exc:
        end_ns = time.monotonic_ns()
        detail = exc.read(4000).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    if status is None or status < 200 or status >= 300:
        raise RuntimeError(f"Unexpected HTTP status: {status}")
    if usage:
        output_tokens = int(usage.get("completion_tokens", output_tokens or 0))
    if output_tokens is None and final_finish_reason == "length":
        output_tokens = 1
        output_token_count_source = "finish_reason_length"
    else:
        output_token_count_source = "server_usage" if output_tokens is not None else None
    try:
        first_token_ns, ttft_source = resolve_first_token_timestamp(
            first_choice_ns,
            first_nonempty_text_ns,
            output_tokens,
            final_finish_reason,
        )
    except ValueError as exc:
        safe_sse_metadata = {
            "choice_chunk_count": choice_chunk_count,
            "first_choice": first_choice_metadata,
            "final_finish_reason": final_finish_reason,
            "server_output_tokens": output_tokens,
        }
        raise RuntimeError(f"{exc}; SSE metadata={json.dumps(safe_sse_metadata)}") from exc
    return {
        "http_status": status,
        "start_monotonic_ns": start_ns,
        "first_token_monotonic_ns": first_token_ns,
        "end_monotonic_ns": end_ns,
        "ttft_ms": (first_token_ns - start_ns) / 1_000_000,
        "elapsed_ms": (end_ns - start_ns) / 1_000_000,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "output_tokens": output_tokens,
        "output_token_count_source": output_token_count_source,
        "ttft_source": ttft_source,
        "sse_choice_chunk_count": choice_chunk_count,
        "first_choice_metadata": first_choice_metadata,
        "final_finish_reason": final_finish_reason,
    }


def _wait_for_metrics(base_url: str, before: dict[str, int], timeout_s: float = 4.0) -> tuple[dict[str, int], dict[str, int]]:
    deadline = time.monotonic() + timeout_s
    last_after: dict[str, int] | None = None
    last_error: Exception | None = None
    while time.monotonic() <= deadline:
        try:
            last_after = _scrape_metrics(base_url)
            delta = calculate_counter_deltas(before, last_after)
            if delta["vllm:prompt_tokens"] > 0 or delta["vllm:prefix_cache_queries"] > 0:
                return last_after, delta
        except Exception as exc:  # Retry transient scrape lag, but report the final error.
            last_error = exc
        time.sleep(0.2)
    if last_after is not None:
        delta = calculate_counter_deltas(before, last_after)
        raise RuntimeError(f"Counters did not reflect completed request: {delta}")
    raise RuntimeError(f"Could not read post-request metrics: {last_error}")


def command_run(args: argparse.Namespace) -> int:
    manifest = _read_json(Path(args.manifest))
    rows: list[dict[str, Any]] = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8", buffering=1) as output:
            for request_spec in manifest["requests"]:
                row: dict[str, Any] = {
                    "name": request_spec["name"],
                    "prompt_sha256": request_spec["sha256"],
                    "expected_prompt_tokens": request_spec["token_count"],
                    "expected_request_count": len(manifest["requests"]),
                    "status": "failed",
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    before = _scrape_metrics(args.base_url)
                    result = _consume_completion(args.base_url, args.model, request_spec["prompt"])
                    after, delta = _wait_for_metrics(args.base_url, before)
                    row.update(result)
                    row["metrics_before"] = before
                    row["metrics_after"] = after
                    row["metrics_delta"] = delta
                    row["cache_hit_fraction"] = (
                        delta["vllm:prefix_cache_hits"] / delta["vllm:prefix_cache_queries"]
                        if delta["vllm:prefix_cache_queries"]
                        else None
                    )
                    row["status"] = "ok"
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    print(json.dumps({"name": row["name"], "status": "failed", "error": row["error"]}), file=sys.stderr)
                    return 2
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    json.dumps(
                        {
                            "name": row["name"],
                            "status": row["status"],
                            "ttft_ms": round(row["ttft_ms"], 3),
                            "elapsed_ms": round(row["elapsed_ms"], 3),
                            "cache_hit_fraction": row["cache_hit_fraction"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    except OSError as exc:
        print(f"Could not write run output {output_path}: {exc}", file=sys.stderr)
        return 2
    return 0


def _read_run(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
    return rows


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def command_summarize(args: argparse.Namespace) -> int:
    run_results: list[dict[str, Any]] = []
    for path_text in args.runs:
        rows = _read_run(Path(path_text))
        try:
            summary = summarize_run(rows)
            expected_count = rows[0].get("expected_request_count", 18) if rows else 18
            summary["run_status"] = (
                "complete"
                if len(rows) == expected_count and not summary["failed_requests"]
                else "incomplete"
            )
        except ValueError as exc:
            summary = {
                "request_count": len([row for row in rows if row.get("status") == "ok"]),
                "failed_requests": [row.get("name", "unknown") for row in rows if row.get("status") != "ok"],
                "run_status": "incomplete",
                "summary_error": str(exc),
            }
        summary["path"] = path_text
        run_results.append(summary)

    phase_names = sorted({name for run in run_results for name in run.get("ttft_ms_by_request", {})})
    phase_summary: dict[str, Any] = {}
    for phase in phase_names:
        values = [run["ttft_ms_by_request"][phase] for run in run_results if phase in run.get("ttft_ms_by_request", {})]
        phase_summary[phase] = {
            "per_run_ms": values,
            "median_ms": _median(values),
            "range_ms": max(values) - min(values) if values else None,
        }

    pressure_passes = len(run_results) == 3 and all(run.get("run_status") == "complete" for run in run_results)
    if pressure_passes:
        for run in run_results:
            fractions = run.get("hit_fraction_by_request", {})
            pressure_passes = pressure_passes and (
                fractions.get("hot_immediate") is not None
                and fractions["hot_immediate"] >= 0.70
                and fractions.get("hot_after_scan") is not None
                and fractions["hot_after_scan"] <= 0.20
                and fractions.get("hot_repeated_again") is not None
                and fractions["hot_repeated_again"] >= 0.70
                and not run.get("failed_requests")
            )
    final_warm_values = [
        run.get("ttft_ms_by_request", {}).get("hot_repeated_again")
        for run in run_results
    ]
    gaps = [run.get("paired_ttft_gap_ms_signed") for run in run_results]
    latency_passes = (
        pressure_passes
        and all(value is not None for value in final_warm_values)
        and all(value is not None for value in gaps)
        and all(gap >= 0.10 * final for gap, final in zip(gaps, final_warm_values))
        and min(gaps) > 2 * (max(final_warm_values) - min(final_warm_values))
    )
    if latency_passes:
        decision = "go"
    elif pressure_passes:
        decision = "pressure only"
    else:
        decision = "no-go/inconclusive"

    result = {
        "schema_version": 1,
        "interpretation": "ideal-retention proxy only; no retention policy was implemented",
        "decision": decision,
        "pressure_passes": pressure_passes,
        "latency_gap_passes": latency_passes,
        "runs": run_results,
        "ttft_by_phase": phase_summary,
        "limitations": [
            "Only three independent service starts; no P95/P99 is reported.",
            "The prompts are synthetic and conclusions apply only to this model, hardware, cache budget, and request mix.",
            "The upper-bound control is a best-case proxy, not a strict mathematical bound or an implemented policy result.",
        ],
    }
    _write_json(Path(args.output), result)
    print(json.dumps({"output": args.output, "decision": decision, "pressure_passes": pressure_passes, "latency_gap_passes": latency_passes}))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Build a deterministic prompt manifest")
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--cold-count", type=int, default=14)
    prepare.add_argument("--seed", type=int, default=20260927)
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(handler=command_prepare)

    run = commands.add_parser("run", help="Replay a manifest against a local vLLM HTTP service")
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument("--model", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--output", required=True)
    run.set_defaults(handler=command_run)

    summarize = commands.add_parser("summarize", help="Summarize raw repetitions and apply the go/no-go rule")
    summarize.add_argument("--runs", nargs="+", required=True)
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(handler=command_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
