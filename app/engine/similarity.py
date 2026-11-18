"""Similarity functions for collaborative filtering.

Two things here are not obvious and both matter more than the choice of
similarity metric itself:

1. **Adjusted cosine.** Users rate on different scales. One person's 3 is
   another person's 5. Subtracting each user's mean before comparing items
   removes that bias. Plain cosine on raw ratings mostly measures "who rates
   generously", not "who likes the same things".

2. **Shrinkage.** Cosine similarity between two items that share a single
   co-rater is noise wearing a number's clothing. Multiplying by
   ``n / (n + lambda)`` pulls low-support pairs toward zero, which is exactly
   what you want: trust the pairs you have evidence for.
"""

from __future__ import annotations

import numpy as np


def _l2_normalise_columns(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_item_similarity(
    dense: np.ndarray, shrinkage_lambda: float = 0.0
) -> np.ndarray:
    """Item-item cosine similarity on raw ratings."""
    normed = _l2_normalise_columns(dense)
    sim = normed.T @ normed
    np.fill_diagonal(sim, 0.0)
    if shrinkage_lambda > 0.0:
        sim = sim * _shrinkage_factor(dense, shrinkage_lambda)
    return sim


def adjusted_cosine_item_similarity(
    dense: np.ndarray, shrinkage_lambda: float = 0.0
) -> np.ndarray:
    """Item-item similarity after removing each user's mean rating."""
    mask = dense > 0.0
    counts = mask.sum(axis=1, keepdims=True)
    counts[counts == 0] = 1
    means = np.where(mask, dense, 0.0).sum(axis=1, keepdims=True) / counts
    centred = np.where(mask, dense - means, 0.0)

    normed = _l2_normalise_columns(centred)
    sim = normed.T @ normed
    np.fill_diagonal(sim, 0.0)
    if shrinkage_lambda > 0.0:
        sim = sim * _shrinkage_factor(dense, shrinkage_lambda)
    return sim


def _shrinkage_factor(dense: np.ndarray, lam: float) -> np.ndarray:
    """``n / (n + lambda)`` where n is the number of co-raters for a pair."""
    present = (dense > 0.0).astype(np.float64)
    co_counts = present.T @ present
    return co_counts / (co_counts + lam)


def user_user_similarity(
    dense: np.ndarray, shrinkage_lambda: float = 0.0
) -> np.ndarray:
    """User-user cosine similarity on mean-centred ratings."""
    mask = dense > 0.0
    counts = mask.sum(axis=1, keepdims=True)
    counts[counts == 0] = 1
    means = np.where(mask, dense, 0.0).sum(axis=1, keepdims=True) / counts
    centred = np.where(mask, dense - means, 0.0)

    norms = np.linalg.norm(centred, axis=1)
    norms[norms == 0.0] = 1.0
    normed = centred / norms[:, None]
    sim = normed @ normed.T
    np.fill_diagonal(sim, 0.0)

    if shrinkage_lambda > 0.0:
        present = mask.astype(np.float64)
        co_counts = present @ present.T
        sim = sim * (co_counts / (co_counts + shrinkage_lambda))
    return sim


def top_k_neighbours(sim: np.ndarray, k: int) -> np.ndarray:
    """Zero out all but the ``k`` strongest neighbours in each row."""
    if k <= 0 or k >= sim.shape[1]:
        return sim
    out = np.zeros_like(sim)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    out[rows, idx] = sim[rows, idx]
    return out
