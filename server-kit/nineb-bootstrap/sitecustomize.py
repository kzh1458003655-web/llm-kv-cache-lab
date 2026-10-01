"""Opt-in policy bootstrap inherited by vLLM's spawned Python processes."""
import functools
import json
import os
from pathlib import Path


def activate():
    policy = os.environ.get('NINEB_POLICY')
    if policy is None:
        return
    if policy not in ('stock', 'reuse2'):
        raise ValueError('unknown 9B policy')
    import vllm
    if vllm.__version__ != '0.30.0':
        raise RuntimeError('9B policy bootstrap requires vLLM 0.30.0')
    diagnostics = None
    if policy == 'reuse2':
        from kv_cache_lab.hybrid_retention_prototype import install
        diagnostics = install(max_scan=32, reuse_threshold=2)
    if os.environ.get('NINEB_DIAGNOSTIC_EVENTS'):
        os.environ['HYBRID_PILOT_EVENTS'] = os.environ['NINEB_DIAGNOSTIC_EVENTS']
        from kv_cache_lab.hybrid_observer import install as install_observer
        install_observer()
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.block_pool import BlockPool
    output = Path(os.environ['NINEB_EVIDENCE_DIR'])
    output.mkdir(parents=True, exist_ok=True)

    def write(kind, value):
        path = output / f'{kind}-{os.getpid()}.json'
        if not path.exists():
            with path.open('x', encoding='utf-8') as stream:
                json.dump({'pid': os.getpid(), 'policy': policy, 'vllm': vllm.__version__, **value}, stream, indent=2)

    original_init = KVCacheManager.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        config = self.kv_cache_config
        groups = []
        for group in config.kv_cache_groups:
            spec = group.kv_cache_spec
            groups.append({'spec': type(spec).__name__, 'block_size': spec.block_size,
                           'page_size_bytes': spec.page_size_bytes,
                           'layers': list(group.layer_names)})
        tensors = getattr(config, 'kv_cache_tensors', [])
        write('cache-groups', {'num_blocks': config.num_blocks,
                              'groups': groups,
                              'tensor_bytes': [tensor.size for tensor in tensors],
                              'total_tensor_bytes': sum(tensor.size for tensor in tensors)})

    original_allocate = BlockPool.get_new_blocks

    @functools.wraps(original_allocate)
    def allocate(self, *args, **kwargs):
        result = original_allocate(self, *args, **kwargs)
        if not getattr(self, '_nineb_first_allocation', False):
            write('engine-active', {'num_gpu_blocks': self.num_gpu_blocks,
                  'hash_block_size': self.hash_block_size,
                  'reuse_hook_installed': bool(getattr(KVCacheManager, '_hybrid_retention_installed', False))})
            self._nineb_first_allocation = True
        return result

    KVCacheManager.__init__ = init
    BlockPool.get_new_blocks = allocate
    # Normal shutdown diagnostics are best effort; startup evidence is required.
    if diagnostics is not None:
        import atexit
        atexit.register(lambda: write('policy-diagnostics', {'diagnostics': diagnostics}))


try:
    activate()
except Exception as exc:
    # Python normally suppresses a sitecustomize error. Explicitly fail closed
    # rather than silently measuring stock while labeling the run reuse2.
    raise SystemExit(f'9B policy bootstrap failed: {type(exc).__name__}: {exc}')
