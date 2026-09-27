# Local KV Cache feasibility check — 2026-09-27

## Decision

**Go to a switchable-policy experiment.** Stock vLLM repeatedly lost a previously warm prefix after a scan of one-off requests, and the resulting per-request latency gap was measurable in three independent service starts. This experiment **did not implement or benchmark an improved cache policy**.

## Setup

- RTX 4060 Laptop GPU (8188 MiB); WSL Ubuntu 24.04; installed, unmodified vLLM 0.30.0; local Qwen3 model (`Qwen3ForCausalLM`, about 4.06 GB BF16 weights).
- Fixed GPU KV cache budget: 268,435,456 bytes (256 MiB); block size 16; max model length 1024; sequential requests (concurrency 1); one generated token per request.
- Per fresh server: cold hot prompt, immediate repeat, 14 distinct one-off prompts, post-scan hot prompt, immediate repeat. All prompts were 352–416 tokens. Three measured runs each had 18/18 successful requests.
- The server was restarted for every measured run. `hot_first` had zero cached-prefix hits each time. Metrics were read from stock vLLM `/metrics` token counters before and after each request.
- TTFT here means time to the first completion *choice* SSE event. Its visible text was empty, but server usage reported one generated token. This is a local measurement convention, not a general end-user visible-text TTFT.

## Measurements

| Run | Hot after scan: hit / TTFT | Immediate repeat: hit / TTFT | Paired gap | No-cost ideal-retention proxy as fraction of this 18-request mix's total TTFT |
| --- | ---: | ---: | ---: | ---: |
| 01 | 0% / 86.49 ms | 96.7% / 37.48 ms | 49.01 ms | 3.74% |
| 02 | 0% / 75.46 ms | 96.7% / 43.43 ms | 32.03 ms | 2.05% |
| 03 | 0% / 67.27 ms | 96.7% / 38.68 ms | 28.59 ms | 1.93% |

Every immediate repeat before the scan also hit 96.7% of its prompt tokens, while every one-off scan request had zero prefix-cache hits. The post-scan request lost the hot prefix in all three runs. The smallest paired gap (28.59 ms) exceeded twice the across-run range of final-warm TTFT (11.90 ms).

The paired gap is only a **best-case proxy** for how much a future, cost-free retention policy might save on that affected hot request. It is not a strict mathematical bound, a measured strategy improvement, or evidence that overall service throughput will increase. The proxy's full-mix median was 2.05%; an actual policy may gain less, do nothing, or regress after its overhead and effects on other requests are included.

## Limitations and next gate

Three runs, a small synthetic workload, one model, one GPU, an intentionally tight KV budget, and no concurrency cannot establish a general performance result. The first-request TTFT varied substantially (102–415 ms), so it was not used as the warm/cold comparison. An initial client run failed to recognize an empty-text first choice event; a separate setup run used a previously warm cache; one service-start attempt failed to become reachable. These attempts were retained locally and excluded from the three valid runs rather than silently counted.

The next gate is to implement **one switchable policy** in the real cache management path and rerun the *same* workload against stock vLLM and the modified version, including a no-sharing control. Only then can this project claim an actual optimization result.

The [implementation plan](superpowers/plans/2026-09-27-kv-cache-feasibility.md) describes the protocol. The client and CPU tests are in `scripts/kv_feasibility.py` and `tests/test_kv_feasibility.py`. Raw prompts, request traces, service logs, `config.json`, and `summary.json` are stored locally under `data/runs/kv-feasibility/` and intentionally ignored by Git.
