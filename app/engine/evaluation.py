"""Offline evaluation.

Every metric here is computed on a temporal holdout, and every one is reported
next to a baseline. A precision@10 of 0.31 means nothing on its own -- it means
something when the popularity baseline scores 0.12 on the same split.

Coverage and popularity bias are included because they are the metrics that
catch the failure mode offline accuracy hides: a model that recommends the same
twenty popular items to everyone scores well on precision and is useless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .matrix import RatingMatrix
from .recommender import Recommender


@dataclass(frozen=True, slots=True)
class EvalResult:
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    coverage: float
    popularity_bias: float
    n_users_evaluated: int
    k: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "coverage": round(self.coverage, 4),
            "popularity_bias": round(self.popularity_bias, 4),
            "n_users_evaluated": self.n_users_evaluated,
            "k": self.k,
        }


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def evaluate(
    recommender: Recommender,
    test: dict[int, list[int]],
    k: int = 10,
) -> EvalResult:
    """Score a fitted recommender against a held-out test set."""
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    recommended_items: set[int] = set()
    recommendation_counts: dict[int, int] = {}

    for user_id, held_out in test.items():
        if not held_out:
            continue
        recs = recommender.recommend(user_id, k=k, exclude_rated=True)
        if not recs:
            continue

        rec_ids = [r.item_id for r in recs]
        for item_id in rec_ids:
            recommended_items.add(item_id)
            recommendation_counts[item_id] = recommendation_counts.get(item_id, 0) + 1

        held_set = set(held_out)
        hits = [1.0 if item_id in held_set else 0.0 for item_id in rec_ids]

        precisions.append(sum(hits) / len(rec_ids))
        recalls.append(sum(hits) / len(held_set))

        ideal = [1.0] * min(len(held_set), len(rec_ids))
        idcg = _dcg(ideal)
        ndcgs.append(_dcg(hits) / idcg if idcg > 0 else 0.0)

    n_eval = len(precisions)
    total_items = recommender.matrix.n_items or 1
    coverage = len(recommended_items) / total_items

    # Popularity bias: share of all recommendation slots taken by the top 10%
    # most-recommended items. 1.0 means everything went to a tiny head.
    bias = 0.0
    if recommendation_counts:
        counts = sorted(recommendation_counts.values(), reverse=True)
        head = max(1, len(counts) // 10)
        total_slots = sum(counts)
        bias = sum(counts[:head]) / total_slots if total_slots else 0.0

    return EvalResult(
        precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
        recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        coverage=coverage,
        popularity_bias=bias,
        n_users_evaluated=n_eval,
        k=k,
    )


def popularity_baseline(
    train: RatingMatrix, test: dict[int, list[int]], k: int = 10
) -> EvalResult:
    """Recommend the globally most-rated items to everyone.

    This is the baseline every recommender must beat. If a model cannot beat
    it, the model is not adding information -- it is a slower popularity list.
    """
    counts: dict[int, int] = {}
    for r in train.ratings:
        counts[r.item_id] = counts.get(r.item_id, 0) + 1
    ranked = [item for item, _ in sorted(counts.items(), key=lambda x: -x[1])]

    rated_by_user = train.user_rated_items()
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    recommended: set[int] = set()
    rec_counts: dict[int, int] = {}

    for user_id, held_out in test.items():
        if not held_out:
            continue
        seen = rated_by_user.get(user_id, set())
        rec_ids = [i for i in ranked if i not in seen][:k]
        if not rec_ids:
            continue
        for item_id in rec_ids:
            recommended.add(item_id)
            rec_counts[item_id] = rec_counts.get(item_id, 0) + 1

        held_set = set(held_out)
        hits = [1.0 if i in held_set else 0.0 for i in rec_ids]
        precisions.append(sum(hits) / len(rec_ids))
        recalls.append(sum(hits) / len(held_set))
        ideal = [1.0] * min(len(held_set), len(rec_ids))
        idcg = _dcg(ideal)
        ndcgs.append(_dcg(hits) / idcg if idcg > 0 else 0.0)

    total_items = train.n_items or 1
    bias = 0.0
    if rec_counts:
        c = sorted(rec_counts.values(), reverse=True)
        head = max(1, len(c) // 10)
        bias = sum(c[:head]) / sum(c)

    return EvalResult(
        precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
        recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        coverage=len(recommended) / total_items,
        popularity_bias=bias,
        n_users_evaluated=len(precisions),
        k=k,
    )
