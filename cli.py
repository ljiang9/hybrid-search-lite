#!/usr/bin/env python3
"""cli.py — hybrid-search-lite 命令行入口。

用法：python3 cli.py "查询" [--k 3] [--alpha 0.5]
alpha=0 纯 BM25，alpha=1 纯向量。
"""
from __future__ import annotations

import sys

from hybrid import SAMPLE_DOCS, HybridIndex


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    query = argv[0] if argv else "向量和关键词怎么混合检索"
    k = 3
    alpha = 0.5
    if "--k" in argv:
        k = int(argv[argv.index("--k") + 1])
    if "--alpha" in argv:
        alpha = float(argv[argv.index("--alpha") + 1])

    idx = HybridIndex()
    for d in SAMPLE_DOCS:
        idx.add(d)

    print(f"查询：{query}（alpha={alpha}, top-{k}）")
    for rank, (doc_id, score) in enumerate(idx.search(query, k=k, alpha=alpha), 1):
        print(f"  {rank}. [score={score:.4f}] {SAMPLE_DOCS[doc_id]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
