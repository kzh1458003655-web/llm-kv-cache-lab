"""Verify free-queue intervention conserves blocks and falls back under pressure."""
from types import SimpleNamespace
from kv_cache_lab.retention_probe import reorder


def test_protection_preserves_order_and_inventory():
    blocks = [SimpleNamespace(block_hash=k) for k in ('hot', 'cold1', 'hot', None, 'cold2')]
    result, forced = reorder(blocks, {'hot'}, 3)
    assert [b.block_hash for b in result] == ['cold1', None, 'cold2', 'hot', 'hot']
    assert sorted(map(id, result)) == sorted(map(id, blocks))
    assert forced == 0


def test_infeasible_protection_preserves_allocation_progress():
    blocks = [SimpleNamespace(block_hash=k) for k in ('hot', 'cold', 'hot')]
    result, forced = reorder(blocks, {'hot'}, 3)
    assert len(result[:3]) == 3
    assert forced == 2


def test_no_protection_is_identity():
    blocks = [SimpleNamespace(block_hash=k) for k in ('a', None, 'b')]
    result, forced = reorder(blocks, set(), 2)
    assert result == blocks and forced == 0
