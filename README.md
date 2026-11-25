# pairwise-tile-histogram-match

## 领域

游戏、图形与多媒体

## 主题

分块直方图匹配

## 说明

大图按块独立做直方图匹配时，块边界亮度台阶、掩码泄漏、异步乱序覆盖、
参考块错位与增益累积会产生网格伪影。本模块在
`imaging/tile_histogram_match.py` 中实现完整流水线，修复点与入口函数：

| 缺陷 | 修复 | 函数 |
| --- | --- | --- |
| 块边界亮度台阶拼成网格 | 重叠区羽化加权融合 | `overlap_blend` / `compose` / `feather_weights` |
| 掩码外像素进入直方图 | 统计与映射排除掩码像素 | `mask_exclude` / `tile_stats` |
| 异步完成顺序导致后到块覆盖重叠区 | 确定行主序 + 加权累加（交换律） | `merge_order` / `compose` / `determinism` |
| 参考裁剪与目标块尺寸不一致导致映射错位 | 同一坐标裁剪，越界边缘填充/重采样对齐 | `ref_map` |
| 多轮迭代累积增益压平高光 | 增益全局限幅 + 高光余量保护 | `gain_limit` |
| 迭代后无一致性保障 | 拼缝跳变检查与网格伪影指标 | `boundary_check` / `metrics` |

配套机制：`validation`（入参校验）、`audit`/`audit_log`/`clear_audit`
（审计轨迹）、`fallback`（常数块/空块恒等兜底）、`tile_stats`/
`match_lut`/`apply_lut`（统计与查找表）。

顶层入口 `match_image(image, ref, mask, tile_size, overlap, iterations)`
一次跑通全部修复，返回 `{"image", "boundary", "metrics", "audit_id"}`。
掩码约定 `True` 为有效像素；灰度图与 `H×W×C` 多通道图均支持。

## 测试

```bash
python -m pytest -q
```

- `tests/test_tile_histogram_match.py`：15 个入口函数的契约测试
- `tests/test_fixes.py`：六个修复点与端到端流水线回归测试
