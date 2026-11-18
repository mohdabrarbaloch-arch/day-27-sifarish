"""Builds a fitted Recommender from the database, with a cache.

Refitting on every request would be correct and unusably slow. Refitting once
and never again would serve stale recommendations forever. The middle ground:
cache the fitted model and invalidate it whenever a rating is written.

This is the honest simple version. At real scale you would fit offline in a
batch job and load the matrices from disk; the interface here is the same.
"""

from __future__ import annotations

import threading

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .engine import Rating, RatingMatrix, Recommender
from .models import Item
from .models import Rating as RatingModel

settings = get_settings()


class EngineCache:
    """Thread-safe holder for the fitted model.

    A lock is not paranoia here: FastAPI runs sync endpoints in a thread pool,
    so two requests can genuinely race on a refit.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recommender: Recommender | None = None
        self._version = 0
        self._fitted_version = -1

    def invalidate(self) -> None:
        with self._lock:
            self._version += 1

    def get(self, db: Session) -> Recommender:
        with self._lock:
            if self._recommender is not None and self._fitted_version == self._version:
                return self._recommender

            recommender = _fit_from_db(db)
            self._recommender = recommender
            self._fitted_version = self._version
            return recommender

    def reset(self) -> None:
        with self._lock:
            self._recommender = None
            self._fitted_version = -1
            self._version = 0


engine_cache = EngineCache()


def _fit_from_db(db: Session) -> Recommender:
    items = list(db.scalars(select(Item).order_by(Item.id)))
    item_texts = {item.id: item.searchable_text for item in items}
    item_titles = {item.id: item.title for item in items}

    rows = db.execute(
        select(RatingModel.user_id, RatingModel.item_id, RatingModel.value, RatingModel.id)
    ).all()

    # The rating row id is a monotonic proxy for time. Using it instead of
    # created_at keeps the split deterministic even when many rows share a
    # timestamp, which they do in seeded data.
    ratings = [
        Rating(user_id=uid, item_id=iid, value=float(value), timestamp=rid)
        for uid, iid, value, rid in rows
    ]

    matrix = RatingMatrix(
        ratings,
        user_ids=sorted({r.user_id for r in ratings}) or [0],
        item_ids=[item.id for item in items] or [0],
    )

    return Recommender(
        matrix=matrix,
        item_texts=item_texts,
        item_titles=item_titles,
        shrinkage_lambda=settings.shrinkage_lambda,
        neighbour_k=settings.neighbour_k,
    )
