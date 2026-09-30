"""One isolated, warmed, serial GPU run; timings are not streaming TTFT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--policy', choices=['stock', 'reuse2'], required=True)
    p.add_argument('--kv-mib', type=int, required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(s) for s in Path(a.manifest).read_text().splitlines() if s.strip()]
    if not rows or len({r['request_id'] for r in rows}) != len(rows):
        raise ValueError('empty manifest or duplicate request IDs')
    for key in ['HYBRID_PILOT_EVENTS']:
        os.environ.pop(key, None)
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_V2_MODEL_RUNNER='0',
                      VLLM_USE_FLASHINFER_SAMPLER='0')
    summary = {'status': 'initializing', 'policy': a.policy, 'kv_mib': a.kv_mib,
               'mode': 'serial offline, diagnostic event hooks disabled',
               'timing_definition': 'synchronized generate call duration; NOT streaming TTFT',
               'success_count': 0, 'request_count': len(rows),
               'manifest_sha256': hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()}
    diagnostics = None
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        if vllm.__version__ != '0.30.0':
            raise RuntimeError('requires vLLM 0.30.0')
        summary['versions'] = {'vllm': vllm.__version__, 'torch': torch.__version__}
        summary['gpu'] = torch.cuda.get_device_name(0)
        if a.policy == 'reuse2':
            from kv_cache_lab.hybrid_retention_prototype import install
            diagnostics = install()
        llm = LLM(model=a.model, language_model_only=True, dtype='bfloat16',
                  enforce_eager=True, max_model_len=2048, max_num_seqs=1,
                  max_num_batched_tokens=1024, gpu_memory_utilization=0.60,
                  kv_cache_memory_bytes=a.kv_mib * 1024**2,
                  enable_prefix_caching=True, mamba_cache_mode='align',
                  safetensors_load_strategy='lazy')
        params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
        # Same-length warmup before measurement; remove warmup cache and counters.
        llm.generate([rows[0]['prompt']], params, use_tqdm=False)
        llm.generate([rows[0]['prompt']], params, use_tqdm=False)
        if not llm.reset_prefix_cache():
            raise RuntimeError('warmup prefix cache reset failed')
        if diagnostics is not None:
            for key in diagnostics:
                diagnostics[key] = 0
        start_run = time.perf_counter_ns()
        with (out / 'requests.jsonl').open('x', encoding='utf-8') as stream:
            for row in rows:
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                result = llm.generate([row['prompt']], params, use_tqdm=False)[0]
                torch.cuda.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                generated = result.outputs[0]
                record = {k: v for k, v in row.items() if k != 'prompt'}
                record.update(status='ok', prompt_tokens=len(result.prompt_token_ids),
                              cached_tokens=getattr(result, 'num_cached_tokens', None),
                              output_tokens=len(generated.token_ids), elapsed_ms=elapsed,
                              output_token_sha256=hashlib.sha256(json.dumps(generated.token_ids).encode()).hexdigest())
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                summary['success_count'] += 1
        summary['wall_ms'] = (time.perf_counter_ns() - start_run) / 1e6
        summary['status'] = 'completed'
    except Exception as exc:
        summary.update(status='failed', error_type=type(exc).__name__,
                       error=str(exc).replace(a.model, '<MODEL>')[:1200])
        raise
    finally:
        summary['policy_diagnostics'] = diagnostics
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
