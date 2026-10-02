"""Diagnostic future-informed retention intervention; never a deployable policy.

Capture only the selected measured source request's prefix. Reorder free blocks
without pinning, changing reference counts, or increasing the cache budget.
"""
import functools
import inspect
import json
import os
import time


def reorder(candidates, protected, num_blocks):
    """Stable partition; preserve native progress if protection is infeasible."""
    safe = [b for b in candidates if b.block_hash not in protected]
    held = [b for b in candidates if b.block_hash in protected]
    return safe + held, max(0, num_blocks - len(safe))


def install(mode):
    import vllm
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import get_group_id
    if vllm.__version__ != '0.30.0' or mode not in ('stock', 'all', 'attention', 'state'):
        raise ValueError('unsupported diagnostic mode/runtime')
    cache, allocate, reset = (BlockPool.cache_full_blocks,
                              BlockPool.get_new_blocks, BlockPool.reset_prefix_cache)
    signature = inspect.signature(cache)
    path = os.environ['NINEB_DIAGNOSTIC_EVENTS']

    def emit(event):
        with open(path, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps({'type': 'retention_probe', 'mode': mode,
                                    'monotonic_ns': time.monotonic_ns(), **event}) + '\n')

    def keys(pool):
        if not hasattr(pool, '_probe_keys'):
            pool._probe_keys = set()
        return pool._probe_keys

    @functools.wraps(cache)
    def register(self, *args, **kwargs):
        values = signature.bind(self, *args, **kwargs)
        values.apply_defaults()
        v = values.arguments
        result = cache(self, *args, **kwargs)
        rid = v['request'].request_id
        if rid.startswith('cmpl-measure-segment002-scan2-'):
            added = []
            for i in range(v['num_cached_blocks'], v['num_full_blocks']):
                block = v['blocks'][i]
                if (i + 1) * v['block_size'] > 3696 or block.is_null or block.block_hash is None:
                    continue
                key = block.block_hash
                group = get_group_id(key)
                selected = mode == 'all' or (mode == 'attention' and group == 3) or (mode == 'state' and group in (0, 1, 2))
                if selected and key not in keys(self):
                    keys(self).add(key)
                    added.append({'key': bytes(key).hex(), 'group_id': group})
            if added:
                emit({'action': 'capture', 'request_id': rid, 'entries': added})
        return result

    @functools.wraps(allocate)
    def get_new(self, num_blocks):
        available = self.get_num_free_blocks()
        if keys(self) and 0 < num_blocks <= available:
            if self._reuse_watchers:
                raise RuntimeError('unexpected reuse watchers in diagnostic')
            queue = self.free_block_queue
            candidates = queue.popleft_n(available)
            ordered, forced = reorder(candidates, keys(self), num_blocks)
            queue.prepend_n(ordered)
            assert queue.num_free_blocks == available
            assert all(b.ref_cnt == 0 and not b.is_null for b in ordered)
            emit({'action': 'selection', 'free_blocks': available,
                  'requested_blocks': num_blocks, 'forced_protected_selections': forced,
                  'selected_protected': sum(b.block_hash in keys(self) for b in ordered[:num_blocks])})
        return allocate(self, num_blocks)

    @functools.wraps(reset)
    def clear(self, *args, **kwargs):
        result = reset(self, *args, **kwargs)
        if result:
            keys(self).clear()
            emit({'action': 'reset'})
        return result

    BlockPool.cache_full_blocks = register
    BlockPool.get_new_blocks = get_new
    BlockPool.reset_prefix_cache = clear
    emit({'action': 'installed', 'source_request': 'segment002-scan2',
          'target_request': 'segment007-scan0', 'prefix_end': 3696})
