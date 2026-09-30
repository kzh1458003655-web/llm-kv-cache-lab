"""Bounded single-process hybrid smoke, separate from HTTP latency benchmarks."""
import argparse
import json
import os
from pathlib import Path
import time

from hybrid_prefix_probe import make_requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    # Use vLLM's existing PyTorch sampler; this WSL environment has no nvcc.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["HYBRID_PILOT_EVENTS"] = str(out / "events.jsonl")
    from kv_cache_lab.hybrid_observer import install
    install()
    from vllm import LLM, SamplingParams
    summary = {"mode": "offline single process; diagnostic hooks enabled",
               "status": "initializing", "success_count": 0,
               "interpretation": "Synthetic availability smoke only; durations are not TTFT or stable performance measurements."}
    try:
        llm = LLM(model=args.model_dir, language_model_only=True,
                  dtype="bfloat16", enforce_eager=True,
                  max_model_len=2048, max_num_seqs=1,
                  max_num_batched_tokens=1024, gpu_memory_utilization=0.60,
                  kv_cache_memory_bytes=134217728, enable_prefix_caching=True,
                  mamba_cache_mode="align", safetensors_load_strategy="lazy")
        for row in make_requests():
            start = time.perf_counter_ns()
            result = llm.generate([row["prompt"]], SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)[0]
            record = {**row, "engine_request_id": result.request_id,
                      "prompt_tokens": len(result.prompt_token_ids),
                      "num_cached_tokens": getattr(result, "num_cached_tokens", None),
                      "output_text": result.outputs[0].text,
                      "output_tokens": len(result.outputs[0].token_ids),
                      "elapsed_ms": (time.perf_counter_ns() - start) / 1e6,
                      "status": "ok"}
            (out / (row["request_id"] + ".json")).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            summary["success_count"] += 1
        summary["status"] = "completed"
    except Exception as exc:
        summary["status"] = "failed"
        # Full traceback stays in the local raw log; publish a bounded message.
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc).replace(args.model_dir, "<MODEL_DIR>")[:2000]
        raise
    finally:
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
