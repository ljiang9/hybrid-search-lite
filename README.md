# hybrid-search-lite

零第三方依赖的**关键词 + 向量混合检索**：BM25 关键词分与 TF-IDF 余弦向量分归一化后加权融合。

## 功能简介

- **关键词路**：从零实现的简化 BM25（词频饱和 + 文档长度归一化 + IDF）；
- **向量路**：TF-IDF 稀疏向量 + 余弦相似度；
- 两路分数各自按 max 归一化到 [0,1]，按 `final = alpha·向量 + (1-alpha)·BM25` 融合；
- `alpha=0` 纯关键词，`alpha=1` 纯向量，默认各占一半。

## 快速开始

```bash
# 默认 0.5/0.5 融合
python3 cli.py "向量和关键词怎么混合检索"

# 纯 BM25 关键词
python3 cli.py "BM25 检索" --alpha 0

# 纯向量
python3 cli.py "怎么结合更准" --alpha 1 --k 2
```

代码调用：

```python
from hybrid import HybridIndex
idx = HybridIndex()
idx.add("一些文档……")
results = idx.search("查询", k=5, alpha=0.5)
```

## 无 API key 如何运行

本项目**完全不需要 API key**，BM25 与向量融合全部本地完成。

## 目录说明

```
hybrid-search-lite/
├── hybrid.py             # 核心库：HybridIndex（BM25 + TF-IDF 融合）
├── cli.py                # 命令行入口
├── tests/test_hybrid.py  # unittest 测试
└── README.md
```

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT License，Copyright (c) 2026 ljiang9
