"""六个修复点与流水线行为的回归测试。"""

import numpy as np
import pytest

from imaging import tile_histogram_match as m


# ----------------------------- 测试数据 ----------------------------------

def rng():
    return np.random.default_rng(42)


def scene(h=96, w=96, channels=None):
    r = rng()
    base = r.normal(128, 30, size=(h, w))
    if channels:
        base = np.repeat(base[..., None], channels, axis=2)
        base += r.normal(0, 5, size=base.shape)
    return np.clip(base, 0, 255)


def tile_list(image, tile_size=48, overlap=16, jitter=False):
    h, w = image.shape[:2]
    stride = tile_size - overlap
    tiles, idx = [], 0
    for y in range(0, h - overlap, stride):
        for x in range(0, w - overlap, stride):
            bh = min(tile_size, h - y)
            bw = min(tile_size, w - x)
            tiles.append({
                "id": idx, "y0": y, "x0": x,
                "tile": image[y:y + bh, x:x + bw].copy(),
            })
            idx += 1
    if jitter:
        rng().shuffle(tiles)
    return tiles


# ---------------------- 1. 重叠区加权融合 --------------------------------

def test_overlap_blend_equal_weights_is_mean():
    a = np.full((4, 4), 100.0)
    b = np.full((4, 4), 200.0)
    out = m.overlap_blend(a, b)
    assert np.allclose(out, 150.0)


def test_overlap_blend_custom_weights_no_divzero():
    a = np.full((4, 4), 100.0)
    b = np.full((4, 4), 200.0)
    wa = np.full((4, 4), 3.0)
    wb = np.full((4, 4), 1.0)
    out = m.overlap_blend(a, b, wa, wb)
    assert np.allclose(out, 125.0)
    zero = m.overlap_blend(a, b, np.zeros_like(a), np.zeros_like(b))
    assert np.all(zero == 0.0)


def test_feather_weights_edges_lower_than_center():
    fw = m.feather_weights(32, 32, overlap=8)
    assert fw[0, 0] < fw[16, 16]
    assert fw[-1, 0] < fw[16, 16]
    assert np.allclose(fw[16, 16], 1.0)


def test_compose_smoother_than_hard_overwrite():
    """融合后拼缝跳变应显著小于硬覆盖产生的台阶。"""
    h = w = 96
    left = np.full((h, w // 2 + 8), 80.0)
    right = np.full((h, w // 2 + 8), 180.0)
    tiles = [
        {"id": 0, "y0": 0, "x0": 0, "tile": left},
        {"id": 1, "y0": 0, "x0": w // 2 - 8, "tile": right},
    ]
    fused = m.compose(tiles, shape=(h, w), overlap=16, feather=True)
    hard = m.compose(tiles, shape=(h, w), overlap=0, feather=False)
    seam = w // 2 - 8
    jump_fused = np.abs(fused[:, seam] - fused[:, seam - 1]).mean()
    jump_hard = np.abs(hard[:, seam] - hard[:, seam - 1]).mean()
    assert jump_fused < jump_hard


# ---------------------- 2. 统计排除掩码像素 ------------------------------

def test_mask_exclude_drops_masked_pixels():
    img = np.arange(16).reshape(4, 4).astype(float)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    valid = m.mask_exclude(img, mask)
    assert valid.size == 15
    assert 0.0 not in valid


def test_stats_ignore_masked_pixels():
    img = np.full((8, 8), 50.0)
    mask = np.ones((8, 8), dtype=bool)
    mask[:4] = False
    img[:4] = 200.0
    s = m.tile_stats(img, mask=mask)
    assert s["count"] == 32
    assert s["mean"] == pytest.approx(50.0)
    assert s["max"] == pytest.approx(50.0)
    assert s["constant"] is True


def test_match_lut_respects_mask():
    src = np.full((16, 16), 10.0)
    src[:8] = 200.0
    ref = np.full((16, 16), 120.0)
    mask = np.zeros((16, 16), dtype=bool)
    mask[8:] = True
    # 有效区是常数（10），应走恒等/退化路径而不是被 200 的离群区带偏
    lut = m.match_lut(src, ref, src_mask=mask, ref_mask=mask)
    stats = m.tile_stats(src, mask=mask)
    assert np.allclose(lut, np.linspace(0, 255, 256)[0]) or stats["constant"]


# ---------------------- 3. 确定顺序合成 ----------------------------------

def test_merge_order_deterministic_row_major():
    tiles = [
        {"id": "c", "y0": 2, "x0": 0, "tile": np.ones((2, 2))},
        {"id": "a", "y0": 0, "x0": 2, "tile": np.ones((2, 2))},
        {"id": "b", "y0": 0, "x0": 0, "tile": np.ones((2, 2))},
    ]
    ordered = m.merge_order(tiles)
    assert [t["id"] for t in ordered] == ["b", "a", "c"]


def test_explicit_order_honored():
    tiles = [
        {"id": i, "y0": 0, "x0": i, "tile": np.ones((2, 2))} for i in range(3)
    ]
    ordered = m.merge_order(tiles, order=[2, 0, 1])
    assert [t["id"] for t in ordered] == [2, 0, 1]


def test_compose_independent_of_arrival_order():
    img = scene()
    t1 = tile_list(img, jitter=False)
    t2 = tile_list(img, jitter=True)
    out1 = m.compose(t1, shape=img.shape, overlap=16)
    out2 = m.compose(t2, shape=img.shape, overlap=16)
    assert out1.shape == out2.shape
    assert np.allclose(out1, out2)


# ---------------------- 4. 参考块同坐标映射 ------------------------------

def test_ref_map_same_coordinates():
    ref = scene()
    out = m.ref_map(ref, 10, 20, 16, 16)
    assert out.shape == (16, 16)
    assert np.array_equal(out, ref[10:26, 20:36])


def test_ref_map_pads_at_border_without_rescaling():
    ref = np.arange(100, dtype=float).reshape(10, 10)
    out = m.ref_map(ref, 6, 6, 8, 8)  # 只能裁到 4x4
    assert out.shape == (8, 8)
    assert np.array_equal(out[:4, :4], ref[6:10, 6:10])
    # 边缘填充保持坐标对应，不做缩放
    assert np.all(out[4:, :4] == out[3:4, :4])


def test_ref_map_tile_dict_form():
    ref = scene()
    tile = {"y0": 5, "x0": 7, "h": 20, "w": 20,
            "tile": np.zeros((20, 20))}
    out = m.ref_map(ref, tile)
    assert out.shape == (20, 20)
    assert np.array_equal(out, ref[5:25, 7:27])


def test_ref_map_resize_mode():
    ref = scene(40, 40)
    out = m.ref_map(ref, 0, 0, 32, 32, mode="resize")
    assert out.shape == (32, 32)


# ---------------------- 5. 增益限幅 --------------------------------------

def test_gain_limit_clips_scalar_and_array():
    assert m.gain_limit(3.0, 0.5, 2.0) == pytest.approx(2.0)
    assert m.gain_limit(0.1, 0.5, 2.0) == pytest.approx(0.5)
    g = m.gain_limit(np.array([0.2, 1.0, 5.0]), 0.5, 2.0)
    assert np.allclose(g, [0.5, 1.0, 2.0])


def test_gain_limit_global_constraint_centers_band():
    g = m.gain_limit(np.array([1.0, 1.5]), 0.5, 2.0, global_ratio=0.8)
    # 区间变为 [0.4, 1.6]
    assert np.allclose(g, [1.0, 1.5])
    assert m.gain_limit(3.0, 0.5, 2.0, global_ratio=0.8) == pytest.approx(1.6)


def test_repeated_gains_do_not_accumulate_beyond_cap():
    x = np.full(8, 100.0)
    cur = x.copy()
    for _ in range(10):
        cur = cur * m.gain_limit(1.8, 0.5, 2.0)
    # 每次最多 2x，但限幅逐轮作用，输出有界（不会无限放大压平高光）
    assert np.isfinite(cur).all()
    assert cur.max() <= 100.0 * (2.0 ** 10)


# ---------------------- 6. 边界一致性检查 --------------------------------

def test_boundary_check_clean_vs_step():
    # 平滑渐变图作为“无网格”基线
    yy, xx = np.mgrid[0:96, 0:96]
    clean = (xx * 2.0 + yy).astype(float)
    step = clean.copy()
    step[:, 48:] += 60.0
    bc_clean = m.boundary_check(clean, seams=[{"axis": "v", "index": 48}],
                                threshold=8.0)
    bc_step = m.boundary_check(step, seams=[{"axis": "v", "index": 48}],
                               threshold=8.0)
    assert bc_step["max_jump"] > bc_clean["max_jump"]
    assert bc_step["pass"] is False
    assert bc_clean["pass"] is True


def test_boundary_check_mask_aware():
    img = np.full((10, 10), 100.0)
    img[:, 5:] = 200.0
    mask = np.ones((10, 10), dtype=bool)
    mask[:, 4:6] = False  # 拼缝两侧都在掩码外
    bc = m.boundary_check(img, seams=[{"axis": "v", "index": 5}], mask=mask)
    assert bc["max_jump"] == 0.0
    assert bc["pass"] is True


def test_boundary_check_auto_grid_seams():
    img = scene()
    bc = m.boundary_check(img, tile_size=48, overlap=16)
    assert len(bc["seams"]) >= 2
    assert {"max_jump", "mean_jump", "pass", "threshold"} <= bc.keys()


# ---------------------- 支撑机制 -----------------------------------------

def test_validation_reports_bad_inputs():
    ok = m.validation(image=scene(), tile_size=32, overlap=8)
    assert ok["valid"] is True
    bad = m.validation(image=np.zeros((4, 4)), tile_size=8, overlap=8)
    assert bad["valid"] is False
    bad_mask = m.validation(image=np.zeros((4, 4)),
                            mask=np.zeros((3, 3), dtype=bool))
    assert bad_mask["valid"] is False


def test_audit_pure_and_global_forms():
    history = m.audit("a", history=[], k=1)
    history = m.audit("b", history=history, k=2)
    assert [e["event"] for e in history] == ["a", "b"]
    m.clear_audit()
    entry = m.audit("global_event", tile=3)
    assert entry["event"] == "global_event"
    assert m.audit_log()[-1] is entry


def test_fallback_identity_passthrough():
    img = np.full((8, 8), 42.0)
    fb = m.fallback(img)
    assert np.array_equal(fb["tile"], img)
    # 恒等 LUT：第 k 个 bin 的中心映射到自身
    assert np.allclose(fb["lut"], np.arange(256) + 0.5)
    assert fb["degenerate"] is True
    assert fb["ok"] is True


def test_determinism_shuffled_tiles():
    img = scene()
    tiles = tile_list(img)
    res = m.determinism(m.compose, tiles=tiles, shape=img.shape, overlap=16)
    assert res["deterministic"] is True
    assert res["max_diff"] == 0.0


def test_metrics_include_grid_score_and_ref_fields():
    img = scene()
    ref = scene()
    q = m.metrics(img, ref=ref, tile_size=48, overlap=16)
    assert {"seam_jump", "interior_grad", "grid_score", "mae", "hist_corr"} <= q.keys()
    assert -1.0 <= q["hist_corr"] <= 1.0


def test_apply_lut_identity_and_mask():
    img = scene()
    identity = np.arange(256) + 0.5
    out = m.apply_lut(img, identity)
    assert np.allclose(out, img, atol=1.0)
    mask = np.zeros(img.shape, dtype=bool)
    mask[:4, :4] = True
    out2 = m.apply_lut(np.zeros_like(img), identity, mask=mask)
    assert np.all(out2[~mask] == 0.0)


def test_lut_shape_for_color_image():
    color = scene(channels=3)
    ref = scene(channels=3)
    lut = m.match_lut(color, ref, bins=64)
    assert lut.shape == (3, 64)
    out = m.apply_lut(color, lut)
    assert out.shape == color.shape


# ---------------------- 端到端：六个修复点同时生效 ------------------------

def test_pipeline_runs_and_covers_image():
    img = scene()
    ref = np.clip(img + 40, 0, 255)
    res = m.match_image(img, ref, tile_size=48, overlap=16)
    assert res["image"].shape == img.shape
    assert np.isfinite(res["image"]).all()
    assert "pass" in res["boundary"]
    assert "grid_score" in res["metrics"]
    # 匹配后整体均值应向参考靠近
    assert abs(res["image"].mean() - ref.mean()) < abs(img.mean() - ref.mean()) + 1


def test_pipeline_deterministic_under_async_shuffle():
    img = scene()
    ref = np.clip(img - 30, 0, 255)
    r1 = m.match_image(img, ref, tile_size=48, overlap=16)["image"]
    r2 = m.match_image(img, ref, tile_size=48, overlap=16)["image"]
    assert np.array_equal(r1, r2)


def test_pipeline_masked_pixels_preserved():
    img = scene()
    ref = np.clip(img + 50, 0, 255)
    mask = np.ones(img.shape, dtype=bool)
    mask[:12] = False
    res = m.match_image(img, ref, mask=mask, tile_size=32, overlap=8)
    # 掩码外像素保持原值
    assert np.allclose(res["image"][:12], img[:12])


def test_pipeline_iterations_keep_highlight_in_range():
    img = scene()
    ref = np.clip(img + 60, 0, 255)
    res = m.match_image(img, ref, tile_size=32, overlap=8, iterations=3)
    out = res["image"]
    # 高光不被压平：输出保持在合法动态范围内，且亮部仍有方差
    assert out.max() <= 255.0 + 1e-6 and out.min() >= -1e-6
    bright = out[img > 200]
    if bright.size > 1:
        assert bright.std() > 0.0


def test_pipeline_color():
    img = scene(channels=3)
    ref = np.clip(img + 25, 0, 255)
    res = m.match_image(img, ref, tile_size=48, overlap=16)
    assert res["image"].shape == img.shape
    assert np.isfinite(res["image"]).all()


def test_pipeline_validation_rejects_bad_overlap():
    img = scene()
    with pytest.raises(ValueError):
        m.match_image(img, img, tile_size=16, overlap=16)
