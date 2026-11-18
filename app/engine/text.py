"""A small TF-IDF implementation for content-based recommendations.

No scikit-learn dependency: the whole thing is about forty lines and keeping
it here means the content-based path is inspectable rather than a black box.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that the
    this to was were will with you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [
        t
        for t in _TOKEN_RE.findall(text.lower())
        if len(t) > 1 and t not in _STOPWORDS
    ]


def build_tfidf(documents: Sequence[str]) -> np.ndarray:
    """Return an L2-normalised ``(n_documents, n_terms)`` TF-IDF matrix."""
    tokenised = [tokenize(d) for d in documents]
    vocab: dict[str, int] = {}
    for tokens in tokenised:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)

    n_docs = len(documents)
    if n_docs == 0 or not vocab:
        return np.zeros((n_docs, max(len(vocab), 1)), dtype=np.float64)

    df = np.zeros(len(vocab), dtype=np.float64)
    for tokens in tokenised:
        for t in set(tokens):
            df[vocab[t]] += 1.0

    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

    matrix = np.zeros((n_docs, len(vocab)), dtype=np.float64)
    for row, tokens in enumerate(tokenised):
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        for term, count in counts.items():
            matrix[row, vocab[term]] = (count / total) * idf[vocab[term]]

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of ``a`` and every row of ``b``."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    return a @ b.T


def idf_weight(term: str, n_docs: int, doc_freq: int) -> float:
    """Exposed for tests: the same weighting used inside :func:`build_tfidf`."""
    return math.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0
