# Fixed 9B Experiment Kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare reproducible Qwen3.5-9B online experiments locally while the GPU instance remains off.

**Architecture:** Keep the historical 0.8B runner intact. Use a pinned model manifest, fixed private requests with public audits, an explicit policy bootstrap, and a bounded Linux runner. Freeze hardware-dependent settings only after the first GPU calibration.

**Tech Stack:** Python 3.12, standard-library artifact tools, Transformers 5.17.0, aiohttp, vLLM 0.30.0, Torch 2.13.0.

## File responsibilities and steps

- [x] Lock model revision and protocol in `configs/thesis-9b-protocol.json` and `docs/固定实验配置.md`.
- [x] Delegate only manifest preparation to the user-requested Luna agent: `scripts/prepare_9b_model_manifest.py`.
- [x] Add `scripts/nineb_artifacts.py` for immutable file verification/download and bundle checks.
- [x] Add `scripts/prepare_9b_workloads.py` for 9B tokenization and saved Poisson schedules; preserve the locked LooGLE source.
- [x] Add `server-kit/nineb-bootstrap/sitecustomize.py` to install the selected policy in the actual engine process and write startup evidence.
- [x] Add `scripts/nineb_online_client.py` for streamed token timing, hashes, errors, and server metrics.
- [x] Add `scripts/run_9b_online.py` for one bounded independent server run, calibration records, warmup/reset, and owned-process cleanup.
- [x] Add `scripts/run_9b_round.py` for a selective A/A and A/B plan using a frozen calibration file.
- [x] Add `scripts/prepare_9b_bundle.py` and `server-kit/nineb/README.md` for local packaging and server commands.
- [x] Check syntax, artifact hashes, fixed workload counts, and CPU-only execution paths. No remote connection or GPU execution in this preparation stage.
- [x] Publish allowlisted code/config/audits, sync them to the user's main local folder, and record preparation status.

## Deferred until the instance is opened

Verify real GPU/driver, model load, native cache groups and checkpoint options, CUDA graph mode, activation memory, streaming compatibility, and actual policy invocation. Save all calibration attempts before freezing cache budgets and arrival rate. Performance and correctness conclusions require the resulting 9B runs.
