"""Rating matrix construction and honest train/test splitting.

The most common mistake in recommender demos is a random train/test split. It
leaks the future into the past: a user's December rating lands in the training
set while their October rating is being predicted. Offline metrics look great
and the model disappoints in production.

Everything here splits by time instead.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Rating:
    """One explicit rating event."""

    user_id: int
    item_id: int
    value: float
    timestamp: int


class RatingMatrix:
    """A users x items rating matrix with stable index maps.

    Missing ratings are 0.0 in the dense form. That is safe here because every
    rating is >= 1, so 0 unambiguously means "not rated".
    """

    __slots__ = (
        "_dense",
        "item_ids",
        "item_index",
        "n_items",
        "n_users",
        "ratings",
        "user_ids",
        "user_index",
    )

    def __init__(
        self,
        ratings: Iterable[Rating],
        user_ids: Sequence[int] | None = None,
        item_ids: Sequence[int] | None = None,
    ) -> None:
        self.ratings: list[Rating] = list(ratings)
        if user_ids is None:
            user_ids = sorted({r.user_id for r in self.ratings})
        if item_ids is None:
            item_ids = sorted({r.item_id for r in self.ratings})
        self.user_ids = list(user_ids)
        self.item_ids = list(item_ids)
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {i: j for j, i in enumerate(self.item_ids)}
        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_ids)
        self._dense: np.ndarray | None = None

    def to_dense(self) -> np.ndarray:
        """Materialise the matrix. Cached; callers must not mutate the result."""
        if self._dense is None:
            m = np.zeros((self.n_users, self.n_items), dtype=np.float64)
            for r in self.ratings:
                ui = self.user_index.get(r.user_id)
                ii = self.item_index.get(r.item_id)
                if ui is not None and ii is not None:
                    m[ui, ii] = r.value
            self._dense = m
        return self._dense

    @property
    def density(self) -> float:
        if self.n_users == 0 or self.n_items == 0:
            return 0.0
        return len(self.ratings) / (self.n_users * self.n_items)

    def split_leave_last_out(
        self, n: int = 1
    ) -> tuple[RatingMatrix, dict[int, list[int]]]:
        """Hold out each user's most recent ``n`` ratings as the test set.

        Users with ``<= n`` ratings keep everything in train and contribute no
        test items. They are excluded from evaluation rather than scored on a
        set the model has already seen -- that would inflate every metric.
        """
        by_user: dict[int, list[Rating]] = {}
        for r in self.ratings:
            by_user.setdefault(r.user_id, []).append(r)

        train: list[Rating] = []
        test: dict[int, list[int]] = {}
        for uid, rs in by_user.items():
            ordered = sorted(rs, key=lambda r: (r.timestamp, r.item_id))
            if len(ordered) <= n:
                train.extend(ordered)
                continue
            train.extend(ordered[:-n])
            test[uid] = [r.item_id for r in ordered[-n:]]

        return RatingMatrix(train, self.user_ids, self.item_ids), test

    def split_temporal(
        self, fraction: float = 0.8
    ) -> tuple[RatingMatrix, dict[int, list[int]]]:
        """Global time cut: earliest ``fraction`` of events train, rest test.

        Unlike leave-last-out this can leave a user with no test items, which is
        fine -- they simply do not contribute to the average.
        """
        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction must be strictly between 0 and 1")
        ordered = sorted(self.ratings, key=lambda r: (r.timestamp, r.item_id))
        cut = int(len(ordered) * fraction)
        train = ordered[:cut]
        test: dict[int, list[int]] = {}
        for r in ordered[cut:]:
            test.setdefault(r.user_id, []).append(r.item_id)
        return RatingMatrix(train, self.user_ids, self.item_ids), test

    def user_rated_items(self) -> dict[int, set[int]]:
        """user_id -> set of item_ids they have rated. Used for exclusion."""
        out: dict[int, set[int]] = {}
        for r in self.ratings:
            out.setdefault(r.user_id, set()).add(r.item_id)
        return out
