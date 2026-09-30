"""Group-aware diagnostic events, not recompute accounting or latency tooling."""
import functools
import inspect
import json
import os
import time


def install():
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_group_id
    if getattr(KVCacheManager, "_hybrid_pilot_installed", False):
        return
    path = os.environ["HYBRID_PILOT_EVENTS"]
    origin = time.monotonic_ns()

    def emit(event):
        event["relative_ns"] = time.monotonic_ns() - origin
        # One engine writer; append also allows process startup records.
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    init = KVCacheManager.__init__
    lookup = KVCacheManager.get_computed_blocks
    cache = BlockPool.cache_full_blocks
    cache_signature = inspect.signature(cache)

    @functools.wraps(init)
    def init_hook(self, *args, **kwargs):
        result = init(self, *args, **kwargs)
        emit({"type": "cache_config", "num_pool_blocks": self.block_pool.num_gpu_blocks,
              "hash_block_size": self.block_pool.hash_block_size,
              "groups": [{"id": i, "spec": type(g.kv_cache_spec).__name__,
                          "block_size": g.kv_cache_spec.block_size,
                          "layer_count": len(g.layer_names)}
                         for i, g in enumerate(self.kv_cache_config.kv_cache_groups)]})
        return result

    @functools.wraps(lookup)
    def lookup_hook(self, request):
        result = lookup(self, request)
        emit({"type": "lookup", "request_id": request.request_id,
              "prompt_tokens": request.num_tokens,
              "cached_tokens": result[1], "shared_prefix_boundary": result[2]})
        return result

    @functools.wraps(cache)
    def cache_hook(self, *args, **kwargs):
        bound = cache_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        result = cache(self, *args, **kwargs)
        entries = []
        for index in range(values["num_cached_blocks"], values["num_full_blocks"]):
            block = values["blocks"][index]
            key = block.block_hash
            if key is not None and not block.is_null:
                entries.append({"block_id": block.block_id, "key": bytes(key).hex(),
                                "group_id": get_group_id(key),
                                "prefix_end": (index + 1) * values["block_size"]})
        emit({"type": "cache_registration_observed",
              "request_id": values["request"].request_id,
              "group_id": values["kv_cache_group_id"], "entries": entries})
        return result

    KVCacheManager.__init__ = init_hook
    KVCacheManager.get_computed_blocks = lookup_hook
    BlockPool.cache_full_blocks = cache_hook
    KVCacheManager._hybrid_pilot_installed = True
