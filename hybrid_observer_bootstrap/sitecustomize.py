"""Opt-in vLLM 0.30 diagnostic hooks; no cache policy is changed."""
import os

if os.environ.get("HYBRID_PILOT_EVENTS"):
    from kv_cache_lab.hybrid_observer import install
    install()
