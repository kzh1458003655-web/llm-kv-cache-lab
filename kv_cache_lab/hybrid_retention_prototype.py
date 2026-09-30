"""Opt-in bounded reuse-frequency eviction prototype for vLLM 0.30.0.

This is an existing-cache retention experiment, not a new checkpoint-placement
algorithm or a complete Marconi reproduction. No active block is selectable.
"""
import functools
import time


def install(max_scan=32, reuse_threshold=2):
    import vllm
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    if vllm.__version__ != "0.30.0":
        raise RuntimeError("Prototype supports only the inspected vLLM 0.30.0")
    if getattr(KVCacheManager, "_hybrid_retention_installed", False):
        return
    lookup = KVCacheManager.get_computed_blocks
    allocate = BlockPool.get_new_blocks
    remove = BlockPool._remove_cached_block_hashes
    diagnostics = {"selection_calls": 0, "selection_cpu_ns": 0,
                   "reordered_calls": 0, "max_scanned_blocks": 0}

    def counts(pool):
        if not hasattr(pool, "_hybrid_reuse_counts"):
            pool._hybrid_reuse_counts = {}
        return pool._hybrid_reuse_counts

    @functools.wraps(lookup)
    def lookup_hook(self, request):
        result = lookup(self, request)
        seen = set()
        counter = counts(self.block_pool)
        for group in result[0].blocks:
            for block in group:
                key = block.block_hash
                if not block.is_null and key is not None and key not in seen:
                    counter[key] = min(reuse_threshold, counter.get(key, 0) + 1)
                    seen.add(key)
        # A conservative bound for diagnostic prototype metadata, including
        # duplicate and partial entries. Clearing only loses protection.
        if len(counter) > 2 * self.block_pool.num_gpu_blocks:
            counter.clear()
        return result

    @functools.wraps(remove)
    def remove_hook(self, block):
        keys = remove(self, block)
        counter = counts(self)
        for key in keys:
            counter.pop(key, None)
        return keys

    @functools.wraps(allocate)
    def allocate_hook(self, num_blocks):
        # Keep native handling for oversized allocations and live readable
        # blocks maintained by reuse watchers; this pilot has no MTP/connector.
        available = self.get_num_free_blocks()
        if (0 < num_blocks <= min(max_scan, available)
                and self.enable_caching and not self._reuse_watchers):
            start = time.perf_counter_ns()
            queue = self.free_block_queue
            candidates = queue.popleft_n(min(max_scan, available))
            counter = counts(self)
            preferred = [block for block in candidates
                         if counter.get(block.block_hash, 0) < reuse_threshold]
            preferred_ids = {block.block_id for block in preferred}
            chosen = (preferred + [block for block in candidates
                                   if block.block_id not in preferred_ids])[:num_blocks]
            selected_ids = {block.block_id for block in chosen}
            remainder = [block for block in candidates if block.block_id not in selected_ids]
            # Restore all candidates before delegating allocation, refcounts,
            # eviction notifications and hash removal to the original engine.
            queue.prepend_n(chosen + remainder)
            assert queue.num_free_blocks == available
            assert all(block.ref_cnt == 0 and not block.is_null for block in chosen)
            diagnostics["selection_calls"] += 1
            diagnostics["max_scanned_blocks"] = max(diagnostics["max_scanned_blocks"], len(candidates))
            diagnostics["reordered_calls"] += int([b.block_id for b in chosen] != [b.block_id for b in candidates[:num_blocks]])
            diagnostics["selection_cpu_ns"] += time.perf_counter_ns() - start
        return allocate(self, num_blocks)

    KVCacheManager.get_computed_blocks = lookup_hook
    BlockPool.get_new_blocks = allocate_hook
    BlockPool._remove_cached_block_hashes = remove_hook
    KVCacheManager._hybrid_retention_installed = True
    return diagnostics
