"""Compare saved short traces; never infer statistically reliable speedup."""
import argparse
import json
from pathlib import Path


def compare(root, baseline, candidate):
    dirs = [root / baseline, root / candidate]
    summaries = [json.loads((d / "summary.json").read_text()) for d in dirs]
    if any(s["status"] != "completed" for s in summaries):
        raise ValueError("Only completed traces can be compared")
    if summaries[0]["request_order"] != summaries[1]["request_order"]:
        raise ValueError("Request order differs")
    rows = [[json.loads((d / (n + ".json")).read_text())
             for n in summaries[0]["request_order"]] for d in dirs]
    if any(a["prompt_sha256"] != b["prompt_sha256"] or a["prompt_tokens"] != b["prompt_tokens"]
           for a, b in zip(*rows)):
        raise ValueError("Compared request inputs differ")
    totals = []
    for name, records in zip([baseline, candidate], rows):
        input_tokens = sum(r["prompt_tokens"] for r in records)
        cached_tokens = sum(r["num_cached_tokens"] for r in records)
        totals.append({"run": name, "request_count": len(records),
                       "input_tokens": input_tokens, "cached_tokens": cached_tokens,
                       "cached_token_fraction": cached_tokens / input_tokens,
                       "sum_elapsed_ms": sum(r["elapsed_ms"] for r in records)})
    return {"runs": totals,
            "outputs_equal_in_this_trace": all(a["output_text"] == b["output_text"] for a, b in zip(*rows)),
            "hot_return_cached_tokens": [records[6]["num_cached_tokens"] for records in rows],
            "sum_elapsed_change_percent": 100 * (totals[1]["sum_elapsed_ms"] / totals[0]["sum_elapsed_ms"] - 1),
            "limitations": "One short pair per scene; diagnostic hooks and first-use JIT included. Not TTFT, throughput, E/P or statistically reliable performance evidence. Matching 16-token outputs is only a smoke check."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = {"synthetic": compare(args.root, "baseline", "reuse2"),
              "loogle": compare(args.root, "loogle-stock-v2", "loogle-reuse2"),
              "capacity_control": compare(args.root, "loogle-stock-v2", "loogle-stock-256mib")}
    with (args.root / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
