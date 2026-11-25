"""大图分块直方图匹配（tiled histogram matching）。

修复点与函数对应关系：

1. ``overlap_blend``  块间重叠区加权融合（羽化权重，消除块边界亮度台阶/网格）
2. ``mask_exclude``   直方图统计时排除掩码外像素
3. ``merge_order``    异步分块结果按确定的行主序合成，重叠区不被后完成的块覆盖
4. ``ref_map``        参考块按目标块同一坐标裁剪，尺寸不一致时填充/重采样对齐
5. ``gain_limit``     迭代增益按全局约束限幅，避免高光被累积增益压平
6. ``boundary_check`` 多轮迭代后做块边界一致性检查

其余函数为流水线支撑：``tile_stats`` / ``match_lut`` / ``apply_lut`` 负责
统计与映射，``compose`` 负责整块拼合，``determinism`` 验证乱序到达下结果
一致，``validation`` 做入参校验，``audit`` 留审计轨迹，``fallback`` 处理
退化块，``metrics`` 给出网格伪影量化指标。顶层入口为 :func:`match_image`。

掩码约定：``True`` 表示有效像素，统计与映射只使用有效像素。
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

BINS = 256
VLO, VHI = 0.0, 256.0  # 256 个单位宽 bin：bin k 覆盖 [k, k+1)，中心 k+0.5
EPS = 1e-8
DEFAULT_GAIN_LO = 0.5
DEFAULT_GAIN_HI = 2.0
DEFAULT_SEAM_THRESHOLD = 8.0

__all__ = [
    "overlap_blend",
    "mask_exclude",
    "merge_order",
    "ref_map",
    "gain_limit",
    "boundary_check",
    "determinism",
    "validation",
    "audit",
    "audit_log",
    "clear_audit",
    "fallback",
    "tile_stats",
    "metrics",
    "match_lut",
    "apply_lut",
    "compose",
    "match_image",
    "feather_weights",
]


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _as_float(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _edges_centers(bins: int, value_range: Sequence[float]):
    lo, hi = float(value_range[0]), float(value_range[1])
    # bin k 覆盖等宽区间 [lo+k*step, lo+(k+1)*step)，中心为 k+0.5 个步长；
    # 对 8 位图 (bins=256, range=[0,256)) 中心恰为 k+0.5
    step = (hi - lo) / bins
    edges = lo + np.arange(bins + 1, dtype=np.float64) * step
    centers = lo + (np.arange(bins, dtype=np.float64) + 0.5) * step
    return edges, centers


def _tile_box(tile: Dict[str, Any]) -> tuple:
    """从块描述符取出 (y0, x0, h, w)，兼容 row/col 命名。"""
    block = np.asarray(tile["tile"])
    y0 = int(tile.get("y0", tile.get("row", 0)))
    x0 = int(tile.get("x0", tile.get("col", 0)))
    h = int(tile.get("h", block.shape[0]))
    w = int(tile.get("w", block.shape[1]))
    return y0, x0, h, w


def feather_weights(h: int, w: int, overlap: int) -> np.ndarray:
    """生成块内逐像素融合权重。

    距块边缘 ``overlap`` 范围内权重由小线性升到 1，块内部保持 1，
    二维权重为两个方向斜坡的乘积，保证重叠区两侧块的贡献平滑过渡。
    """

    h, w = int(h), int(w)
    overlap = max(0, int(overlap))
    if overlap <= 0:
        return np.ones((h, w), dtype=np.float64)

    def ramp(n: int) -> np.ndarray:
        r = np.ones(n, dtype=np.float64)
        k = min(overlap, n // 2)
        if k <= 0:
            return r
        ramp_up = (np.arange(k, dtype=np.float64) + 1.0) / (k + 1.0)
        r[:k] = ramp_up
        r[n - k:] = ramp_up[::-1]
        return r

    return ramp(h)[:, None] * ramp(w)[None, :]


def _resize_bilinear(image: np.ndarray, h: int, w: int) -> np.ndarray:
    """无 cv2 依赖的双线性重采样，用于参考块尺寸对齐。"""

    src_h, src_w = image.shape[:2]
    if src_h == h and src_w == w:
        return image.copy()
    ys = np.linspace(0.0, src_h - 1.0, h)
    xs = np.linspace(0.0, src_w - 1.0, w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = image[y0[:, None], x0[None, :]] * (1 - wx) + image[y0[:, None], x1[None, :]] * wx
    bottom = image[y1[:, None], x0[None, :]] * (1 - wx) + image[y1[:, None], x1[None, :]] * wx
    return top * (1 - wy) + bottom * wy


# ---------------------------------------------------------------------------
# 2. 掩码排除
# ---------------------------------------------------------------------------

def mask_exclude(values: Any, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """返回掩码内有效像素组成的数组，掩码外像素不参与任何统计。

    灰度图返回一维 ``(N,)``；多通道图返回 ``(N, C)``。``mask`` 为 None
    时退化为全部像素展平。
    """

    values = np.asarray(values)
    if mask is None:
        if values.ndim <= 1:
            return values.reshape(-1)
        return values.reshape(-1, values.shape[-1])
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != values.shape[:2]:
        raise ValueError(
            f"mask 空间尺寸 {mask.shape} 与图像 {values.shape[:2]} 不一致"
        )
    if values.ndim == 2:
        return values[mask]
    return values[mask]


# ---------------------------------------------------------------------------
# 块统计
# ---------------------------------------------------------------------------

def _stats_1d(valid: np.ndarray, bins: int, lo: float, hi: float) -> Dict[str, Any]:
    valid = np.asarray(valid, dtype=np.float64).reshape(-1)
    valid = valid[np.isfinite(valid)]
    edges, centers = _edges_centers(bins, (lo, hi))
    hist = np.zeros(bins, dtype=np.float64)
    if valid.size:
        hist, _ = np.histogram(valid, bins=bins, range=(lo, hi))
        hist = hist.astype(np.float64)
    cdf = np.cumsum(hist)
    total = cdf[-1] if cdf.size else 0.0
    cdf = cdf / total if total > 0 else np.zeros_like(cdf)
    count = int(valid.size)
    if count:
        mean = float(valid.mean())
        std = float(valid.std())
        vmin = float(valid.min())
        vmax = float(valid.max())
        constant = bool(count < 2 or (vmax - vmin) <= EPS)
    else:
        mean = std = vmin = vmax = 0.0
        constant = True
    return {
        "hist": hist,
        "cdf": cdf,
        "centers": centers,
        "edges": edges,
        "mean": mean,
        "std": std,
        "min": vmin,
        "max": vmax,
        "count": count,
        "constant": constant,
    }


def tile_stats(
    tile: Any,
    mask: Optional[np.ndarray] = None,
    bins: int = BINS,
    value_range: Sequence[float] = (VLO, VHI),
) -> Dict[str, Any]:
    """计算单块统计量（直方图/CDF/均值/方差…），统计时排除掩码外像素。"""

    tile = np.asarray(tile)
    lo, hi = float(value_range[0]), float(value_range[1])
    valid = mask_exclude(tile, mask)

    if tile.ndim == 3:
        channels = tile.shape[2]
        valid = valid.reshape(-1, channels) if valid.size else np.zeros((0, channels))
        per = [_stats_1d(valid[:, c], bins, lo, hi) for c in range(channels)]
        return {
            "hist": np.stack([p["hist"] for p in per]),
            "cdf": np.stack([p["cdf"] for p in per]),
            "centers": per[0]["centers"],
            "edges": per[0]["edges"],
            "mean": np.array([p["mean"] for p in per]),
            "std": np.array([p["std"] for p in per]),
            "min": np.array([p["min"] for p in per]),
            "max": np.array([p["max"] for p in per]),
            "count": min(p["count"] for p in per),
            "constant": all(p["constant"] for p in per),
            "bins": bins,
            "range": (lo, hi),
            "channels": channels,
        }

    d = _stats_1d(valid, bins, lo, hi)
    d.update({"bins": bins, "range": (lo, hi), "channels": 1})
    return d


# ---------------------------------------------------------------------------
# 直方图匹配 LUT
# ---------------------------------------------------------------------------

def _stats_from(x: Any, m: Optional[np.ndarray], bins: int,
                value_range: Sequence[float]) -> Dict[str, Any]:
    if isinstance(x, dict) and "cdf" in x:
        return x
    return tile_stats(x, mask=m, bins=bins, value_range=value_range)


def match_lut(
    src: Any,
    ref: Any = None,
    src_mask: Optional[np.ndarray] = None,
    ref_mask: Optional[np.ndarray] = None,
    bins: int = BINS,
    value_range: Sequence[float] = (VLO, VHI),
) -> np.ndarray:
    """由源块与参考块的 CDF 构造直方图匹配查找表。

    掩码外像素不进入两侧统计；参考块缺失、有效像素为空或参考块为常数
    （匹配无意义）时返回恒等 LUT，交由调用方走退化兜底。
    灰度返回 ``(bins,)``，多通道返回 ``(C, bins)``。
    """

    lo, hi = float(value_range[0]), float(value_range[1])
    _, centers = _edges_centers(bins, (lo, hi))
    s = _stats_from(src, src_mask, bins, (lo, hi))

    if ref is None:
        return centers.copy()
    r = _stats_from(ref, ref_mask, bins, (lo, hi))

    if s.get("count", 0) == 0 or r.get("count", 0) == 0 or r.get("constant"):
        return np.repeat(centers[None, :], s.get("channels", 1), axis=0).squeeze()

    if s["channels"] > 1:
        luts = []
        for c in range(s["channels"]):
            luts.append(np.interp(s["cdf"][c], r["cdf"][c], r["centers"]))
        return np.stack(luts)
    return np.interp(s["cdf"], r["cdf"], r["centers"])


def apply_lut(
    tile: Any,
    lut: Optional[np.ndarray] = None,
    value_range: Sequence[float] = (VLO, VHI),
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """按 LUT 对块做连续值映射；掩码外像素保持原值不变。"""

    tile = np.asarray(tile, dtype=np.float64)
    lo, hi = float(value_range[0]), float(value_range[1])
    bins = BINS if lut is None else int(np.asarray(lut).shape[-1])
    _, centers = _edges_centers(bins, (lo, hi))
    if lut is None:
        lut = centers
    lut = np.asarray(lut, dtype=np.float64)

    if lut.ndim == 2:
        outs = [
            np.interp(tile[..., c], centers, lut[c],
                      left=float(lut[c, 0]), right=float(lut[c, -1]))
            for c in range(lut.shape[0])
        ]
        out = np.stack(outs, axis=-1)
    else:
        out = np.interp(tile, centers, lut,
                        left=float(lut[0]), right=float(lut[-1]))

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        out[~m] = tile[~m]
    return out.astype(np.float64)


# ---------------------------------------------------------------------------
# 5. 增益限幅
# ---------------------------------------------------------------------------

def gain_limit(
    gains: Any,
    lo: float = DEFAULT_GAIN_LO,
    hi: float = DEFAULT_GAIN_HI,
    global_ratio: Optional[float] = None,
    base: Any = None,
    value_range: Sequence[float] = (VLO, 255.0),
) -> Any:
    """把逐像素/逐块增益约束在 ``[lo, hi]`` 内。

    ``global_ratio`` 为全局目标均值与当前均值之比，给出时以它为中心
    平移允许区间，使限幅同时服从全局亮度约束。

    提供 ``base``（增益作用前的像素值）时做高光保护：越靠近动态范围
    上界的像素允许的增益上界越小，保证 ``base*gain`` 不超过上界，多轮
    迭代中每轮增益都被截断，避免累积增益把高光压平。
    """

    scalar = np.isscalar(gains) or np.asarray(gains).ndim == 0
    g = np.asarray(gains, dtype=np.float64)
    if global_ratio is not None:
        ratio = float(global_ratio)
        lo, hi = lo * ratio, hi * ratio
    if lo > hi:
        lo, hi = hi, lo
    clipped = np.clip(g, lo, hi)

    if base is not None:
        lo_v, hi_v = float(value_range[0]), float(value_range[1])
        b = np.asarray(base, dtype=np.float64)
        # 像素到动态范围上界的剩余空间决定其增益上界；headroom 充足时
        # 保持全局 hi 不变，只对接近上界的高光像素收缩（不影响暗部）
        headroom = (hi_v - b) / np.maximum(b - lo_v, EPS)
        local_hi = np.minimum(hi, headroom)
        clipped = np.clip(clipped, lo, local_hi)

    return float(clipped) if scalar else clipped


# ---------------------------------------------------------------------------
# 1. 重叠区加权融合
# ---------------------------------------------------------------------------

def overlap_blend(a: Any, b: Any = None, wa: Any = None,
                  wb: Any = None) -> np.ndarray:
    """两个块在重叠区的加权平均融合。

    权重缺省时等权融合；任一点权重和为 0 时输出 0，不会产生除零。
    相比硬覆盖，加权融合消去块边界的亮度台阶。
    """

    a = _as_float(a)
    if b is None:
        return a
    b = _as_float(b)
    if wa is None:
        wa = np.ones_like(a) * 0.5
    if wb is None:
        wb = np.ones_like(b) * 0.5
    wa = _as_float(wa)
    wb = _as_float(wb)
    denom = wa + wb
    safe = np.where(denom > EPS, denom, 1.0)
    out = (wa * a + wb * b) / safe
    return np.where(denom > EPS, out, 0.0)


# ---------------------------------------------------------------------------
# 3. 确定顺序合成
# ---------------------------------------------------------------------------

def merge_order(tiles: Any, order: Optional[Sequence[Any]] = None) -> List[Any]:
    """返回确定的块合成顺序（默认行主序：y0、x0、id）。

    异步任务完成先后不影响结果：compose 前一律按本顺序排列，且重叠区
    采用加权累加（交换律），而非后到块直接覆盖。``order`` 可显式给出
    id 序列；非列表输入原样返回。
    """

    if not isinstance(tiles, (list, tuple)):
        return tiles

    if order is not None:
        id_to_tile = {
            t.get("id", i): t
            for i, t in enumerate(tiles)
            if isinstance(t, dict)
        }
        head = [id_to_tile[k] for k in order if k in id_to_tile]
        used = {id(k) for k in head}
        tail = [t for t in tiles if id(t) not in used]
        return head + tail

    def sort_key(item):
        i, t = item
        if isinstance(t, dict):
            y0 = int(t.get("y0", t.get("row", i)))
            x0 = int(t.get("x0", t.get("col", i)))
            return (y0, x0, str(t.get("id", i)))
        return (0, 0, str(i))

    return [t for _, t in sorted(enumerate(tiles), key=sort_key)]


# ---------------------------------------------------------------------------
# 4. 参考块同坐标映射
# ---------------------------------------------------------------------------

def ref_map(
    ref: Any,
    y0: Any = 0,
    x0: int = 0,
    h: Optional[int] = None,
    w: Optional[int] = None,
    tile: Optional[Dict[str, Any]] = None,
    mode: str = "pad",
) -> np.ndarray:
    """按目标块的同一坐标从参考图裁出参考块。

    坐标与目标块严格一致（``ref[y0:y0+h, x0:x0+w]``），裁出的块再对齐
    到 ``(h, w)``：边界处越界部分默认边缘填充（保持坐标对应，不产生
    映射错位）；``mode="resize"`` 时改用双线性重采样。
    """

    if isinstance(y0, dict):  # ref_map(ref, tile_dict) 调用形式
        tile = y0
        y0 = 0
    if tile is not None:
        y0, x0, h, w = _tile_box(tile)

    ref = np.asarray(ref)
    if ref.ndim < 2:
        return np.atleast_2d(ref).copy()

    H, W = ref.shape[:2]
    h = H if h is None else int(h)
    w = W if w is None else int(w)
    y0, x0 = max(0, int(y0)), max(0, int(x0))
    y1, x1 = min(H, y0 + h), min(W, x0 + w)
    crop = ref[y0:y1, x0:x1]

    if crop.shape[0] == h and crop.shape[1] == w:
        return crop.copy()

    if mode == "resize" and crop.size:
        return _resize_bilinear(crop.astype(np.float64), h, w)

    # 默认 pad：坐标对应关系不被缩放破坏
    pad = ((0, max(0, h - crop.shape[0])), (0, max(0, w - crop.shape[1])))
    if crop.ndim == 3:
        pad += ((0, 0),)
    if crop.size:
        return np.pad(crop, pad, mode="edge")
    return np.zeros((h, w) + ref.shape[2:], dtype=ref.dtype)


# ---------------------------------------------------------------------------
# 拼合
# ---------------------------------------------------------------------------

def compose(
    tiles: Any,
    shape: Optional[Sequence[int]] = None,
    overlap: int = 0,
    feather: bool = True,
    background: float = 0.0,
    base: Optional[np.ndarray] = None,
) -> np.ndarray:
    """按确定顺序把分块加权累加到画布，重叠区做羽化融合。

    块描述符：``{"y0","x0","tile","mask"(可选),"weight"(可选),"id"(可选)}``。
    加权累加对合成顺序满足交换律，配合 :func:`merge_order` 保证异步乱序
    完成时结果逐位一致，且不存在后到块覆盖重叠区的问题。未被任何有效
    像素覆盖的位置取 ``base``（缺省取 ``background``），因此掩码外区域
    可以原样保留。
    """

    if not isinstance(tiles, (list, tuple)):
        return np.atleast_2d(_as_float(tiles))

    tiles = merge_order(tiles)
    if not tiles:
        if base is not None:
            return _as_float(base).copy()
        return np.zeros(tuple(shape) if shape else (0, 0), dtype=np.float64)

    channels = np.asarray(tiles[0]["tile"]).ndim == 3
    if shape is None:
        H = max(_tile_box(t)[0] + _tile_box(t)[2] for t in tiles)
        W = max(_tile_box(t)[1] + _tile_box(t)[3] for t in tiles)
    else:
        H, W = int(shape[0]), int(shape[1])
    C = np.asarray(tiles[0]["tile"]).shape[2] if channels else 1

    if base is None:
        canvas = np.full((H, W, C), background, dtype=np.float64) if channels \
            else np.full((H, W), background, dtype=np.float64)
    else:
        canvas = _as_float(base)
        if canvas.shape != ((H, W, C) if channels else (H, W)):
            canvas = np.broadcast_to(canvas, (H, W, C) if channels else (H, W)).copy()
    numer = np.zeros((H, W, C), dtype=np.float64) if channels else np.zeros(
        (H, W), dtype=np.float64)
    accum = np.zeros((H, W), dtype=np.float64)

    for t in tiles:
        y0, x0, bh, bw = _tile_box(t)
        block = _as_float(t["tile"])
        y1, x1 = min(H, y0 + bh), min(W, x0 + bw)
        bh, bw = y1 - y0, x1 - x0
        if bh <= 0 or bw <= 0:
            continue
        block = block[:bh, :bw]

        if t.get("mask") is None:
            m = np.ones((bh, bw), dtype=np.float64)
        else:
            m = np.asarray(t["mask"], dtype=bool)[:bh, :bw].astype(np.float64)
        fw = feather_weights(bh, bw, overlap) if feather else np.ones((bh, bw))
        wmap = fw * m * float(t.get("weight", 1.0))

        if channels:
            numer[y0:y1, x0:x1] += wmap[..., None] * block
        else:
            numer[y0:y1, x0:x1] += wmap * block
        accum[y0:y1, x0:x1] += wmap

    covered = accum > EPS
    safe = np.where(covered, accum, 1.0)
    out = numer / safe[..., None] if channels else numer / safe
    if channels:
        out[~covered] = canvas[~covered]
    else:
        out[~covered] = canvas[~covered]
    return out


# ---------------------------------------------------------------------------
# 6. 边界一致性检查 / 指标
# ---------------------------------------------------------------------------

def _seam_jumps(image: np.ndarray, seams: Sequence[Dict[str, Any]],
                mask: Optional[np.ndarray]) -> List[Dict[str, Any]]:
    results = []
    for seam in seams:
        axis = seam.get("axis", "v")
        idx = int(seam.get("index", seam.get("position", 0)))
        if axis == "v":
            if not (0 < idx < image.shape[1]):
                continue
            left, right = image[:, idx - 1], image[:, idx]
            m_left = None if mask is None else mask[:, idx - 1]
            m_right = None if mask is None else mask[:, idx]
        else:
            if not (0 < idx < image.shape[0]):
                continue
            left, right = image[idx - 1, :], image[idx, :]
            m_left = None if mask is None else mask[idx - 1, :]
            m_right = None if mask is None else mask[idx, :]
        diff = np.abs(_as_float(left) - _as_float(right))
        if diff.ndim == 2:
            diff = diff.mean(axis=-1)
        valid = np.ones(diff.shape, dtype=bool)
        if m_left is not None:
            valid &= np.asarray(m_left, dtype=bool)
        if m_right is not None:
            valid &= np.asarray(m_right, dtype=bool)
        jump = float(diff[valid].mean()) if valid.any() else 0.0
        results.append({"axis": axis, "index": idx, "jump": jump})
    return results


def _default_seams(h: int, w: int, tile_size: int, overlap: int) -> List[Dict[str, Any]]:
    step = max(1, int(tile_size) - int(overlap))
    seams = [{"axis": "v", "index": x} for x in range(step, w, step)]
    seams += [{"axis": "h", "index": y} for y in range(step, h, step)]
    return seams


def boundary_check(
    image: Any,
    seams: Optional[Sequence[Dict[str, Any]]] = None,
    tile_size: Optional[int] = None,
    overlap: int = 0,
    mask: Optional[np.ndarray] = None,
    threshold: float = DEFAULT_SEAM_THRESHOLD,
) -> Dict[str, Any]:
    """检查块拼缝两侧的亮度跳变，返回每条拼缝的跳变与是否达标。

    拼缝可显式给出，也可由 ``tile_size/overlap`` 推出的网格线生成。
    """

    image = np.asarray(image)
    if image.ndim < 2:
        return {
            "pass": True, "max_jump": 0.0, "mean_jump": 0.0,
            "seams": [], "threshold": float(threshold),
        }
    h, w = image.shape[:2]
    if seams is None:
        if tile_size is None:
            seams = []
        else:
            seams = _default_seams(h, w, int(tile_size), int(overlap))
    jumps = _seam_jumps(image, seams, mask)
    values = [j["jump"] for j in jumps]
    max_jump = max(values) if values else 0.0
    mean_jump = float(np.mean(values)) if values else 0.0
    return {
        "pass": bool(max_jump <= threshold),
        "max_jump": float(max_jump),
        "mean_jump": float(mean_jump),
        "seams": jumps,
        "threshold": float(threshold),
    }


def metrics(
    image: Any,
    ref: Any = None,
    seams: Optional[Sequence[Dict[str, Any]]] = None,
    tile_size: Optional[int] = None,
    overlap: int = 0,
    mask: Optional[np.ndarray] = None,
    value_range: Sequence[float] = (VLO, VHI),
) -> Dict[str, Any]:
    """网格伪影与匹配质量指标。

    - ``seam_jump``：拼缝处平均邻域跳变；
    - ``interior_grad``：非拼缝处的平均邻域梯度（自然纹理基线）；
    - ``grid_score``：两者之比，≈1 表示没有额外网格伪影，越大越严重；
    - 提供参考图时另给 MAE 与直方图相关系数。
    """

    image = np.asarray(image)
    if image.ndim < 2:
        return {
            "seam_jump": 0.0, "interior_grad": 0.0, "grid_score": 0.0,
            "mae": 0.0, "hist_corr": 1.0,
        }

    h, w = image.shape[:2]
    if seams is None and tile_size is not None:
        seams = _default_seams(h, w, int(tile_size), int(overlap))
    seams = seams or []
    seam_jumps = _seam_jumps(image, seams, mask)
    seam_jump = float(np.mean([j["jump"] for j in seam_jumps])) if seam_jumps else 0.0

    grad_x = np.abs(np.diff(_as_float(image), axis=1))
    grad_y = np.abs(np.diff(_as_float(image), axis=0))
    if grad_x.ndim == 3:
        grad_x, grad_y = grad_x.mean(-1), grad_y.mean(-1)
    interior = np.concatenate([grad_x.reshape(-1), grad_y.reshape(-1)])
    interior_grad = float(interior.mean()) if interior.size else 0.0
    grid_score = float(seam_jump / (interior_grad + EPS))

    out = {
        "seam_jump": seam_jump,
        "interior_grad": interior_grad,
        "grid_score": grid_score,
    }

    if ref is not None:
        ref_a = _as_float(ref)
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            out["mae"] = float(np.abs(_as_float(image) - ref_a)[m].mean())
        else:
            out["mae"] = float(np.abs(_as_float(image) - ref_a).mean())
        s = tile_stats(image, mask=mask, value_range=value_range)
        r = tile_stats(ref_a, mask=mask, value_range=value_range)
        if s["channels"] > 1:
            corrs = [
                np.corrcoef(s["hist"][c], r["hist"][c])[0, 1]
                for c in range(s["channels"])
            ]
            out["hist_corr"] = float(np.nanmean(corrs))
        else:
            out["hist_corr"] = float(np.corrcoef(s["hist"], r["hist"])[0, 1])
    return out


# ---------------------------------------------------------------------------
# 确定性验证
# ---------------------------------------------------------------------------

def _same(a: Any, b: Any) -> tuple:
    a_arr = isinstance(a, np.ndarray) or isinstance(b, np.ndarray)
    if a_arr:
        aa, bb = np.asarray(a), np.asarray(b)
        if aa.shape != bb.shape:
            return False, float("inf")
        if aa.size == 0:
            return True, 0.0
        diff = float(np.max(np.abs(aa.astype(float) - bb.astype(float))))
        return np.allclose(aa, bb, equal_nan=True), diff
    return a == b, 0.0


def determinism(fn: Any = None, *args: Any, repeats: int = 2, **kwargs: Any) -> Dict[str, Any]:
    """验证可调用对象重复执行（含块乱序到达）结果完全一致。

    - ``fn`` 不可调用时，把所有位置参数当作待比较的多份输出；
    - ``fn`` 可调用且 kwargs 含 ``tiles`` 列表时，额外用固定随机种子
      打乱块顺序再执行一次，模拟异步任务完成顺序不同。
    """

    if fn is None:
        return {"deterministic": True, "max_diff": 0.0, "outputs": []}
    if not callable(fn):
        outputs = [fn] + list(args)
    else:
        outputs = [fn(*args, **kwargs) for _ in range(max(1, repeats))]
        if isinstance(kwargs.get("tiles"), list) and len(kwargs["tiles"]) > 1:
            shuffled = list(kwargs["tiles"])
            random.Random(0).shuffle(shuffled)
            kw2 = dict(kwargs)
            kw2["tiles"] = shuffled
            outputs.append(fn(*args, **kw2))

    deterministic = True
    max_diff = 0.0
    base = outputs[0]
    for other in outputs[1:]:
        same, diff = _same(base, other)
        deterministic &= same
        max_diff = max(max_diff, diff)
    return {
        "deterministic": bool(deterministic),
        "max_diff": float(max_diff),
        "output": base,
        "outputs": outputs,
    }


# ---------------------------------------------------------------------------
# 入参校验 / 审计 / 兜底
# ---------------------------------------------------------------------------

def validation(image: Any = None, mask: Any = None, ref: Any = None,
               tile_size: Any = None, overlap: Any = None,
               value_range: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """校验流水线入参，返回结构化报告（不抛异常，便于上层决策）。"""

    errors: List[str] = []
    warnings: List[str] = []

    image_arr = np.asarray(image) if isinstance(image, np.ndarray) else None
    if image_arr is None or image_arr.ndim not in (2, 3):
        errors.append("image 必须是二维灰度或三维多通道 numpy 数组")
    elif not np.isfinite(image_arr.astype(np.float64)).all():
        errors.append("image 含 NaN/Inf")

    if mask is not None:
        mask_arr = np.asarray(mask)
        if image_arr is not None and mask_arr.shape != image_arr.shape[:2]:
            errors.append(
                f"mask 尺寸 {mask_arr.shape} 与 image 空间尺寸 "
                f"{image_arr.shape[:2]} 不一致"
            )
        if not mask_arr.any():
            warnings.append("mask 内没有有效像素")

    if ref is not None:
        ref_arr = np.asarray(ref)
        if image_arr is not None and ref_arr.shape != image_arr.shape:
            errors.append(f"ref 形状 {ref_arr.shape} 与 image {image_arr.shape} 不一致")

    if tile_size is not None and int(tile_size) <= 0:
        errors.append("tile_size 必须为正整数")
    if overlap is not None:
        if int(overlap) < 0:
            errors.append("overlap 不能为负")
        elif tile_size is not None and int(overlap) >= int(tile_size):
            errors.append("overlap 必须小于 tile_size")

    if value_range is not None and float(value_range[0]) >= float(value_range[1]):
        errors.append("value_range 下界必须小于上界")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "shape": None if image_arr is None else tuple(image_arr.shape),
    }


_AUDIT_LOG: List[Dict[str, Any]] = []


def audit(event: Any = None, history: Optional[List[Dict[str, Any]]] = None,
          **fields: Any) -> Dict[str, Any]:
    """追加一条审计记录。

    传入 ``history`` 列表时走纯函数形式（返回新列表，不改全局日志）；
    否则写入模块级审计日志并返回该条目。``seq`` 单调递增保证可排序。
    """

    entry = {"seq": len(history) if history is not None else len(_AUDIT_LOG),
             "time": time.time(), "event": event}
    entry.update(fields)
    if history is not None:
        return history + [entry]
    _AUDIT_LOG.append(entry)
    return entry


def audit_log() -> List[Dict[str, Any]]:
    return list(_AUDIT_LOG)


def clear_audit() -> None:
    _AUDIT_LOG.clear()


def fallback(tile: Any, error: Optional[str] = None,
             bins: int = BINS,
             value_range: Sequence[float] = (VLO, VHI)) -> Dict[str, Any]:
    """退化块（无有效像素/常数块/LUT 构造失败）的安全兜底。

    返回原图副本与恒等 LUT，即该块不做直方图匹配，避免错误映射在
    迭代中扩散。
    """

    t = np.asarray(tile)
    _, centers = _edges_centers(bins, value_range)
    stats = None
    try:
        stats = tile_stats(t, bins=bins, value_range=value_range)
        degenerate = bool(stats["constant"] or stats["count"] == 0)
    except Exception:  # pragma: no cover - 防御性
        degenerate = True
    reason = error or ("degenerate_tile" if degenerate else "identity_passthrough")
    return {
        "tile": t.copy(),
        "lut": centers.copy(),
        "reason": reason,
        "degenerate": degenerate,
        "ok": True,
    }


# ---------------------------------------------------------------------------
# 顶层流水线
# ---------------------------------------------------------------------------

def _plan_grid(h: int, w: int, tile_size: int, overlap: int) -> List[tuple]:
    stride = tile_size - overlap
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    boxes = []
    for y in ys:
        for x in xs:
            boxes.append((y, x, min(tile_size, h - y), min(tile_size, w - x)))
    # 末行/末列贴边，保证整图被覆盖
    if h - tile_size > 0 and (h - tile_size) % stride != 0:
        for x in xs:
            boxes.append((h - tile_size, x, tile_size, min(tile_size, w - x)))
    if w - tile_size > 0 and (w - tile_size) % stride != 0:
        for y in ys:
            boxes.append((y, w - tile_size, min(tile_size, h - y), tile_size))
    if h > tile_size and w > tile_size and (h - tile_size) % stride != 0 \
            and (w - tile_size) % stride != 0:
        boxes.append((h - tile_size, w - tile_size, tile_size, tile_size))
    # 去重并保持行主序
    seen, unique = set(), []
    for b in boxes:
        if b not in seen and b[2] > 0 and b[3] > 0:
            seen.add(b)
            unique.append(b)
    return sorted(unique)


def match_image(
    image: np.ndarray,
    ref: np.ndarray,
    mask: Optional[np.ndarray] = None,
    tile_size: int = 64,
    overlap: int = 16,
    iterations: int = 1,
    bins: int = BINS,
    value_range: Optional[Sequence[float]] = None,
    gain_lo: float = DEFAULT_GAIN_LO,
    gain_hi: float = DEFAULT_GAIN_HI,
    seam_threshold: float = DEFAULT_SEAM_THRESHOLD,
) -> Dict[str, Any]:
    """分块直方图匹配完整流水线，集中落实六个修复点。

    流程：校验 → 网格规划 → 每轮每块（掩码排除统计 → 同坐标参考块 →
    LUT/退化兜底 → 应用 → 增益限幅）→ 确定顺序加权拼合 → 边界检查与
    指标 → 审计。返回结果图、检查报告与指标。
    """

    image = np.asarray(image, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if value_range is None:
        value_range = (0.0, 1.0) if max(np.nanmax(image), np.nanmax(ref)) <= 1.0 \
            else (VLO, 256.0)

    report = validation(image, mask, ref, tile_size, overlap, value_range)
    if not report["valid"]:
        raise ValueError("入参校验失败: " + "; ".join(report["errors"]))
    if overlap >= tile_size:
        raise ValueError("overlap 必须小于 tile_size")

    h, w = image.shape[:2]
    boxes = _plan_grid(h, w, tile_size, overlap)
    current = image.copy()
    audit("pipeline_start", tiles=len(boxes), tile_size=tile_size,
          overlap=overlap, iterations=iterations)

    for it in range(max(1, int(iterations))):
        tiles = []
        for idx, (y0, x0, bh, bw) in enumerate(boxes):
            block = current[y0:y0 + bh, x0:x0 + bw]
            bmask = None if mask is None else np.asarray(mask, dtype=bool)[
                y0:y0 + bh, x0:x0 + bw]
            # 修复点 4：参考块与目标块同一坐标映射
            ref_block = ref_map(ref, y0, x0, bh, bw)
            ref_mask = bmask
            stats = tile_stats(block, mask=bmask, bins=bins, value_range=value_range)
            if stats["constant"] or stats["count"] == 0:
                fb = fallback(block, bins=bins, value_range=value_range)
                matched = block.copy()
                lut = fb["lut"]
            else:
                # 修复点 2：两侧统计都排除掩码外像素
                lut = match_lut(block, ref_block, bmask, ref_mask,
                                bins=bins, value_range=value_range)
                matched = apply_lut(block, lut, value_range=value_range, mask=bmask)

            # 修复点 5：每轮增益独立限幅；多轮迭代时启用基于像素余量的
            # 高光保护，阻止累积增益把高光压平（单轮时 LUT 本身即为最终映射）
            base = np.where(np.abs(block) > EPS, block, EPS)
            protect = block if int(iterations) > 1 else None
            gains = gain_limit(matched / base, gain_lo, gain_hi,
                               base=protect, value_range=value_range)
            if bmask is not None:
                gains = np.where(bmask, gains, 1.0)
            out_block = block * gains
            if bmask is not None:
                out_block[~bmask] = block[~bmask]

            tiles.append({
                "id": idx, "y0": y0, "x0": x0,
                "tile": out_block, "mask": bmask,
            })
        # 修复点 1+3：确定顺序 + 重叠区羽化加权融合；base 保证掩码外像素保留
        current = compose(tiles, shape=(h, w), overlap=overlap, feather=True,
                          base=current)
        boundary = boundary_check(
            current, tile_size=tile_size, overlap=overlap, mask=mask,
            threshold=seam_threshold,
        )
        audit("iteration_done", iteration=it,
              gain_range=(float(np.min(gains)), float(np.max(gains))),
              max_jump=boundary["max_jump"], passed=boundary["pass"])

    # 修复点 6：迭代结束后的边界一致性检查
    lo, hi = float(value_range[0]), float(value_range[1])
    # LUT bin 中心可能略微超出合法动态范围（如 255.5），统一钳回，防止高光溢出
    current = np.clip(current, lo, min(hi, 255.0) if hi <= 256.0 else hi)
    boundary = boundary_check(
        current, tile_size=tile_size, overlap=overlap, mask=mask,
        threshold=seam_threshold,
    )
    quality = metrics(current, ref=ref, tile_size=tile_size, overlap=overlap,
                      mask=mask, value_range=value_range)
    audit("pipeline_end", boundary_pass=boundary["pass"],
          grid_score=quality["grid_score"])
    return {
        "image": current,
        "boundary": boundary,
        "metrics": quality,
        "audit_id": len(audit_log()),
    }
