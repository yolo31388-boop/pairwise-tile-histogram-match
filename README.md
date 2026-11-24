# pairwise-tile-histogram-match

## 领域

游戏、图形与多媒体

## 主题

分块直方图匹配

## 说明

重叠区加权、掩码排除与增益限幅。

初始状态下 `imaging/tile_histogram_match.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
