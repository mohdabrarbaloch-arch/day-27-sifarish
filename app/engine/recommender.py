"""The hybrid recommender.

Three signals, blended with weights that depend on how much evidence we have
about the user:

* **Item-item CF** -- "people who liked what you liked also liked this". Strong
  when the user has history, useless when they don't.
* **User-user CF** -- "people like you liked this". Adds diversity that
  item-item misses, but noisier on small data.
* **Content-based** -- TF-IDF over item text. The only signal that works for a
  brand-new user, and the only one that can explain itself in words.

The blend is not a fixed 0.5/0.3/0.2. It is a function of the user's rating
count, because a fixed blend is the single most common reason a demo
recommender falls apart on a new user: 60% of the score comes from a
collaborative signal that has nothing to say.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .matrix import RatingMatrix
from .similarity import (
    adjusted_cosine_item_similarity,
    top_k_neighbours,
    user_user_similarity,
)
from .text import build_tfidf, cosine_rows


@dataclass(frozen=True, slots=True)
class Recommendation:
    item_id: int
    score: float
    reason: str
    source: str


@dataclass(frozen=True, slots=True)
class BlendWeights:
    item_cf: float
    user_cf: float
    content: float

    def normalised(self) -> BlendWeights:
        total = self.item_cf + self.user_cf + self.content
        if total <= 0.0:
            return BlendWeights(0.0, 0.0, 1.0)
        return BlendWeights(
            self.item_cf / total, self.user_cf / total, self.content / total
        )


def blend_for_history(n_ratings: int, rating_std: float | None = None) -> BlendWeights:
    """Weights as a function of how much we know about the user.

    The curve is deliberately steep at the start. With zero ratings the
    collaborative signals are pure noise, so content gets everything. By ten
    ratings the user has enough overlap for item-item to lead.

    ``rating_std`` is the second half of the story, and it is the part most
    implementations miss. Collaborative filtering works on *deviations* from a
    user's own mean. A user who rates everything 5.0 has a standard deviation
    of zero, so every deviation is zero, so item-item CF returns a flat vector
    of noise -- no matter how many ratings they have. Counting ratings alone
    would hand that user a 55% item-CF weight and a feed that looks random.

    When the spread is low we shift weight to content, which does not care
    about the user's scale at all.
    """
    if n_ratings <= 0:
        return BlendWeights(0.0, 0.0, 1.0)

    base: BlendWeights
    if n_ratings < 3:
        base = BlendWeights(0.15, 0.05, 0.80)
    elif n_ratings < 10:
        base = BlendWeights(0.45, 0.15, 0.40)
    else:
        base = BlendWeights(0.55, 0.25, 0.20)

    if rating_std is None:
        return base

    # Below ~0.5 the user is effectively rating on a single level and the
    # collaborative signal is unusable. Scale the CF share down smoothly
    # rather than switching at a cliff.
    confidence = min(rating_std / 0.5, 1.0)
    if confidence >= 1.0:
        return base

    cf_share = (base.item_cf + base.user_cf) * confidence
    content_share = 1.0 - cf_share
    if base.item_cf + base.user_cf <= 0.0:
        return BlendWeights(0.0, 0.0, 1.0)
    ratio = base.item_cf / (base.item_cf + base.user_cf)
    return BlendWeights(
        cf_share * ratio, cf_share * (1.0 - ratio), content_share
    )


@dataclass
class Recommender:
    """Fitted state for one catalogue snapshot.

    Build it once per data refresh, then serve many requests. All the matrix
    work happens in :meth:`fit`; :meth:`recommend` is a few dot products.
    """

    matrix: RatingMatrix
    item_texts: dict[int, str]
    item_titles: dict[int, str] = field(default_factory=dict)
    shrinkage_lambda: float = 5.0
    neighbour_k: int = 40
    similarity_floor: float = 0.05

    _dense: np.ndarray = field(init=False, repr=False)
    _item_sim: np.ndarray = field(init=False, repr=False)
    _user_sim: np.ndarray = field(init=False, repr=False)
    _tfidf: np.ndarray = field(init=False, repr=False)
    _item_row: dict[int, int] = field(init=False, repr=False)
    _user_row: dict[int, int] = field(init=False, repr=False)
    _rated: dict[int, set[int]] = field(init=False, repr=False)
    _popularity: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.fit()

    def fit(self) -> None:
        self._dense = self.matrix.to_dense()
        self._item_row = dict(self.matrix.item_index)
        self._user_row = dict(self.matrix.user_index)
        self._rated = self.matrix.user_rated_items()

        self._item_sim = adjusted_cosine_item_similarity(
            self._dense, self.shrinkage_lambda
        )
        self._user_sim = top_k_neighbours(
            user_user_similarity(self._dense, self.shrinkage_lambda),
            self.neighbour_k,
        )

        docs = [self.item_texts.get(i, "") for i in self.matrix.item_ids]
        self._tfidf = build_tfidf(docs)

        # Popularity is a Bayesian-smoothed mean, not a raw count. A raw count
        # makes one 5-star rating look better than fifty 4-star ratings.
        mask = self._dense > 0.0
        counts = mask.sum(axis=0)
        sums = np.where(mask, self._dense, 0.0).sum(axis=0)
        prior_mean = float(sums.sum() / counts.sum()) if counts.sum() else 3.0
        prior_weight = 10.0
        self._popularity = (sums + prior_weight * prior_mean) / (
            counts + prior_weight
        )

    # -- individual signals -------------------------------------------------

    def _item_cf_scores(self, user_id: int) -> np.ndarray:
        row = self._user_row.get(user_id)
        if row is None:
            return np.zeros(self.matrix.n_items)
        user_ratings = self._dense[row]
        rated_mask = user_ratings > 0.0
        if not rated_mask.any():
            return np.zeros(self.matrix.n_items)

        # Weight each neighbour by how far above the user's own mean it sits.
        user_mean = float(user_ratings[rated_mask].mean())
        centred = np.where(rated_mask, user_ratings - user_mean, 0.0)

        # Only trust neighbours with a similarity worth trusting.
        #
        # This floor is the single most important line in the function. Without
        # it, an item that has a similarity of 0.02 to everything the user
        # rated still gets a positive numerator, and dividing by a denominator
        # built from those same tiny numbers inflates it into a top-ranked
        # recommendation. That is exactly what happened: a Romance title with
        # ~0.02 similarity to every Thriller the demo account had rated was
        # scoring 0.97 and landing at the top of a Thriller fan's feed.
        #
        # Zeroing sub-threshold similarities means "no evidence" produces a
        # zero score instead of a loud one.
        sims = self._item_sim.copy()
        sims[np.abs(sims) < self.similarity_floor] = 0.0

        scores = sims.T @ centred
        denom = np.abs(sims).T @ rated_mask.astype(np.float64)
        denom[denom == 0.0] = 1.0
        result = scores / denom

        # Where no neighbour cleared the floor, there is no collaborative
        # opinion at all. Return zeros so the blend falls through to content.
        result[denom <= 1e-12] = 0.0
        return result

    def _user_cf_scores(self, user_id: int) -> np.ndarray:
        row = self._user_row.get(user_id)
        if row is None:
            return np.zeros(self.matrix.n_items)
        neighbours = self._user_sim[row]
        if not np.any(neighbours):
            return np.zeros(self.matrix.n_items)

        # Same floor as item-CF, for the same reason: a neighbour with a
        # near-zero similarity contributes noise that the normalisation then
        # amplifies.
        neighbours = neighbours.copy()
        neighbours[np.abs(neighbours) < self.similarity_floor] = 0.0
        if not np.any(neighbours):
            return np.zeros(self.matrix.n_items)

        mask = self._dense > 0.0
        counts = mask.sum(axis=1, keepdims=True)
        counts[counts == 0] = 1
        means = np.where(mask, self._dense, 0.0).sum(axis=1, keepdims=True) / counts
        centred = np.where(mask, self._dense - means, 0.0)

        scores = neighbours @ centred
        denom = np.abs(neighbours) @ mask.astype(np.float64)
        denom[denom == 0.0] = 1.0
        result = scores / denom
        result[denom <= 1e-12] = 0.0
        return result

    def _content_scores(self, user_id: int) -> np.ndarray:
        """Average TF-IDF vector of what the user rated highly, vs every item."""
        row = self._user_row.get(user_id)
        if row is None:
            return np.zeros(self.matrix.n_items)
        user_ratings = self._dense[row]
        rated_mask = user_ratings > 0.0
        if not rated_mask.any():
            return np.zeros(self.matrix.n_items)

        # Ratings below the user's own mean contribute nothing; above it, they
        # weight the profile. This stops a user who rates everything 5 from
        # having a profile that is just "the whole catalogue".
        user_mean = float(user_ratings[rated_mask].mean())
        weights = np.where(rated_mask, np.maximum(user_ratings - user_mean, 0.0), 0.0)
        if weights.sum() == 0.0:
            weights = rated_mask.astype(np.float64)

        profile = weights @ self._tfidf
        norm = np.linalg.norm(profile)
        if norm == 0.0:
            return np.zeros(self.matrix.n_items)
        profile = profile / norm
        return cosine_rows(profile[None, :], self._tfidf)[0]

    # -- public API ---------------------------------------------------------

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_rated: bool = True,
        weights: BlendWeights | None = None,
    ) -> list[Recommendation]:
        if k <= 0 or self.matrix.n_items == 0:
            return []

        rated = self._rated.get(user_id, set())
        w = (weights or blend_for_history(len(rated), self.rating_std(user_id))).normalised()

        item_cf = self._item_cf_scores(user_id)
        user_cf = self._user_cf_scores(user_id)
        content = self._content_scores(user_id)

        # Each signal is min-max normalised before blending. Without this the
        # signal with the largest raw scale silently wins regardless of weight.
        blended = (
            w.item_cf * _minmax(item_cf)
            + w.user_cf * _minmax(user_cf)
            + w.content * _minmax(content)
        )

        # A small popularity prior breaks ties sensibly and keeps a brand-new
        # user from seeing the same arbitrary five items every time.
        blended = blended + 0.02 * _minmax(self._popularity)

        if exclude_rated:
            for item_id in rated:
                idx = self._item_row.get(item_id)
                if idx is not None:
                    blended[idx] = -np.inf

        source = self._dominant_source(w)

        order = np.argsort(-blended)[:k]
        out: list[Recommendation] = []
        for idx in order:
            score = float(blended[idx])
            if not np.isfinite(score):
                continue
            item_id = self.matrix.item_ids[idx]
            out.append(
                Recommendation(
                    item_id=item_id,
                    score=round(score, 6),
                    reason=self._explain(user_id, item_id, source),
                    source=source,
                )
            )
        return out

    def rating_std(self, user_id: int) -> float:
        """Standard deviation of a user's ratings, or 0.0 if unknown.

        This is the number that decides whether collaborative filtering has
        anything to work with. See :func:`blend_for_history`.
        """
        row = self._user_row.get(user_id)
        if row is None:
            return 0.0
        values = self._dense[row]
        values = values[values > 0.0]
        if values.size < 2:
            return 0.0
        return float(np.std(values))

    def similar_items(self, item_id: int, k: int = 10) -> list[tuple[int, float]]:
        idx = self._item_row.get(item_id)
        if idx is None:
            return []
        sims = self._item_sim[idx].copy()
        order = np.argsort(-sims)[:k]
        return [
            (self.matrix.item_ids[j], round(float(sims[j]), 6))
            for j in order
            if sims[j] > 0.0
        ]

    def _dominant_source(self, w: BlendWeights) -> str:
        if w.content >= max(w.item_cf, w.user_cf):
            return "content"
        if w.item_cf >= w.user_cf:
            return "item_cf"
        return "user_cf"

    def _explain(self, user_id: int, item_id: int, source: str) -> str:
        """A sentence built from the actual numbers, not a template.

        ``source`` is the dominant signal that produced this score, and the
        explanation must agree with it. An earlier version recomputed the
        branch independently and could label an item ``item_cf`` while the
        sentence talked about content -- the card contradicted itself, and a
        reader who noticed would be right not to trust the rest of it.
        """
        rated = self._rated.get(user_id, set())
        if not rated:
            return "New here, so this is based on what the item is about."

        if source == "content":
            return (
                f"Matches the topics in the {len(rated)} item(s) you rated "
                "highest."
            )

        idx = self._item_row.get(item_id)
        if idx is None:
            return "Recommended from the catalogue."

        # Find the strongest co-rated item to name in the explanation.
        row = self._user_row.get(user_id)
        if row is None:
            return "Recommended from the catalogue."
        sims = self._item_sim[idx]
        best_j, best_sim = -1, 0.0
        for rated_item in rated:
            j = self._item_row.get(rated_item)
            if j is None or j == idx:
                continue
            if sims[j] > best_sim:
                best_j, best_sim = j, float(sims[j])

        # Only name a neighbour if the similarity is meaningful. An earlier
        # version tested ``> 0.0`` and then formatted to two decimals, so a
        # similarity of 0.004 produced the sentence "similarity 0.00" -- a
        # claim the reader can see is false. The threshold now matches the
        # precision of the number being printed.
        #
        # 0.05 rather than 0.01: on a sparse catalogue almost every pair clears
        # 0.01, so naming a 0.02 match as the reason is technically true and
        # practically meaningless. Below 0.05 we say the honest thing instead.
        if best_j >= 0 and best_sim >= 0.05:
            neighbour = self.item_titles.get(
                self.matrix.item_ids[best_j],
                f"item {self.matrix.item_ids[best_j]}",
            )
            return (
                f"People who rated \u201c{neighbour}\u201d highly also rated this "
                f"highly (similarity {best_sim:.2f})."
            )

        # CF led on the strength of many weak overlaps rather than one strong
        # one. Say that, instead of switching to a content sentence the source
        # field would contradict.
        return (
            "No single strong match to what you have rated, so this comes from "
            "many weak overlaps across your history."
        )


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scale a signal to roughly [0, 1] *without* promoting noise.

    The obvious implementation -- ``(x - min) / (max - min)`` -- is wrong here,
    and it took a failing demo to see why. Min-max always maps the largest
    value to 1.0, no matter how small that value actually is. A signal whose
    strongest entry is 0.001 gets stretched to exactly the same range as one
    whose strongest entry is 0.9. Noise becomes indistinguishable from
    evidence, and the blend weights stop meaning anything.

    This version divides by the largest magnitude (preserving sign and relative
    size) and then multiplies by a confidence term that shrinks the whole
    signal when its peak is small. A weak signal stays weak.
    """
    if values.size == 0:
        return values
    # A constant signal carries no information: every item scores the same, so
    # there is nothing to rank on. Returning zeros lets the blend fall through
    # to a signal that does discriminate, instead of handing this one a flat
    # 0.95 that competes with real evidence.
    if float(np.ptp(values)) < 1e-12:
        return np.zeros_like(values)
    peak = float(np.max(np.abs(values)))
    if peak < 1e-12:
        return np.zeros_like(values)

    scaled = values / peak
    # peak / (peak + 0.1) is ~0.5 at peak=0.1 and ~0.9 at peak=0.9, so a
    # signal has to be genuinely strong before it earns full weight.
    confidence = peak / (peak + 0.1)
    return scaled * confidence
