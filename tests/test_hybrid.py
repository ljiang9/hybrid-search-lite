import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid import HybridIndex, SAMPLE_DOCS, tokenize  # noqa: E402


def build():
    idx = HybridIndex()
    for d in SAMPLE_DOCS:
        idx.add(d)
    return idx


class TestHybrid(unittest.TestCase):
    def setUp(self):
        self.idx = build()

    def test_count(self):
        self.assertEqual(len(self.idx.docs), len(SAMPLE_DOCS))

    def test_search_ordered(self):
        hits = self.idx.search("向量 余弦", k=3)
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_exact_keyword_match_bm25(self):
        # 纯关键词路（alpha=0）应把含 "BM25" 的文档排到最前
        hits = self.idx.search("BM25 关键词 检索", k=1, alpha=0.0)
        self.assertEqual(hits[0][0], 1)  # SAMPLE_DOCS[1]

    def test_hybrid_finds_concept(self):
        hits = self.idx.search("关键词和向量怎么结合更准", k=2)
        ids = [i for i, _ in hits]
        self.assertIn(3, ids)  # SAMPLE_DOCS[3] 讲混合检索

    def test_extremes(self):
        # alpha=0 与 alpha=1 都能跑通且结果数正确
        self.assertEqual(len(self.idx.search("天气 散步", k=2, alpha=0.0)), 2)
        self.assertEqual(len(self.idx.search("天气 散步", k=2, alpha=1.0)), 2)

    def test_empty_query(self):
        self.assertEqual(self.idx.search(""), [])

    def test_empty_index(self):
        self.assertEqual(HybridIndex().search("x"), [])

    def test_tokenize(self):
        self.assertIn("bm25", tokenize("BM25"))


if __name__ == "__main__":
    unittest.main()
