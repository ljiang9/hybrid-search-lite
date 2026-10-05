"""hybrid.py — 关键词 + 向量混合检索（零第三方依赖）。

两路召回并融合：
- 关键词路：简化版 BM25（词频饱和 + 文档长度归一化 + IDF）；
- 向量路：TF-IDF 稀疏向量 + 余弦相似度。

两路分数分别按 max 归一化到 [0,1] 后，按权重 alpha 线性融合：
    final = alpha * vec_norm + (1 - alpha) * bm25_norm
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.docs: List[str] = []
        self.tf: List[Counter] = []
        self.df: Counter = Counter()
        self.dl: List[int] = []
        self.k1 = k1
        self.b = b

    def add(self, text: str) -> int:
        tokens = tokenize(text)
        counts = Counter(tokens)
        self.docs.append(text)
        self.tf.append(counts)
        self.dl.append(len(tokens))
        for term in counts:
            self.df[term] += 1
        return len(self.docs) - 1

    @property
    def avgdl(self) -> float:
        return (sum(self.dl) / len(self.dl)) if self.dl else 0.0

    # ---------- BM25 ----------
    def _bm25_scores(self, q_tokens: List[str]) -> Dict[int, float]:
        N = len(self.docs)
        avgdl = self.avgdl or 1.0
        scores: Dict[int, float] = {i: 0.0 for i in range(N)}
        for term in set(q_tokens):
            n = self.df.get(term, 0)
            if n == 0:
                continue
            idf = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                norm = 1 - self.b + self.b * (self.dl[i] / avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * norm)
        return scores

    # ---------- TF-IDF 向量 ----------
    def _vec(self, counts: Counter) -> Dict[str, float]:
        N = max(len(self.docs), 1)
        if not counts:
            return {}
        mx = max(counts.values())
        return {t: (c / mx) * (math.log((1 + N) / (1 + self.df.get(t, 0))) + 1.0)
                for t, c in counts.items()}

    @staticmethod
    def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def _vec_scores(self, q_tokens: List[str]) -> Dict[int, float]:
        qv = self._vec(Counter(q_tokens))
        if not qv:
            return {}
        return {i: self._cos(qv, self._vec(c)) for i, c in enumerate(self.tf)}

    @staticmethod
    def _norm_max(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        mx = max(scores.values())
        if mx <= 0:
            return scores
        return {i: v / mx for i, v in scores.items()}

    def search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[int, float]]:
        """alpha=0 纯 BM25，alpha=1 纯向量，默认 0.5 各占一半。"""
        q = tokenize(query)
        if not q or not self.docs:
            return []
        bm25 = self._norm_max(self._bm25_scores(q))
        vec = self._norm_max(self._vec_scores(q))
        fused = {}
        for i in range(len(self.docs)):
            fused[i] = alpha * vec.get(i, 0.0) + (1 - alpha) * bm25.get(i, 0.0)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]


SAMPLE_DOCS: Sequence[str] = [
    "余弦相似度衡量两个向量夹角的余弦，常用于向量检索排序。",
    "BM25 是经典的关键词检索排序函数，考虑词频饱和与文档长度归一化。",
    "今天天气很好，适合去公园散步，还吃了冰淇淋。",
    "混合检索把关键词命中和向量相似度加权融合，往往比单路更准。",
]
