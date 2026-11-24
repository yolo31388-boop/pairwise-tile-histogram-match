import pytest
import imaging.tile_histogram_match as mod

CASES = [
    ("overlap_blend", (1,)),
    ("mask_exclude", (1,)),
    ("merge_order", (1,)),
    ("ref_map", (1,)),
    ("gain_limit", (1,)),
    ("boundary_check", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("tile_stats", (1,)),
    ("metrics", (1,)),
    ("match_lut", (1,)),
    ("apply_lut", (1,)),
    ("compose", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
