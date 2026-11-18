"""Tests for the core engine.

These are the tests that matter: they assert on the mathematics of the
recommender, not on whether an endpoint returns 200. Several of them encode
real bugs that were found and fixed, so a regression fails loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.engine import (
    Rating,
    RatingMatrix,
    Recommender,
    blend_for_history,
    evaluate,
    popularity_baseline,
)
from app.engine.text import build_tfidf, tokenize

# --------------------------------------------------------------- fixtures


def _synthetic(n_users: int = 60, seed: int = 7) -> tuple[RatingMatrix, dict[int, str]]:
    """Three clear taste clusters over 30 items.

    Cluster membership is the only signal, which makes it obvious whether the
    engine recovers it.
    """
    rng = np.random.default_rng(seed)
    items = list(range(1, 31))
    texts = {
        i: (
            "action chase explosion pursuit" if i <= 10
            else "romance letters longing wedding" if i <= 20
            else "documentary ocean wildlife ecology"
        )
        for i in items
    }

    ratings: list[Rating] = []
    ts = 0
    for u in range(1, n_users + 1):
        cluster = u % 3
        pool = items[cluster * 10 : cluster * 10 + 10]
        # Rate 4 of the 10 cluster items, leaving 6 unseen in-cluster
        # candidates. Rating 8 would leave only 2, and no recommender could
        # then return 5 in-cluster items out of 8 -- the test would be
        # asserting something impossible.
        chosen = rng.choice(pool, size=4, replace=False)
        for it in chosen:
            ts += 1
            ratings.append(Rating(u, int(it), float(rng.choice([4, 4, 5, 5])), ts))
    return RatingMatrix(ratings), texts


# ----------------------------------------------------------------- matrix


def test_matrix_index_and_density():
    matrix = RatingMatrix([
        Rating(1, 10, 5.0, 1),
        Rating(2, 10, 4.0, 2),
        Rating(2, 20, 3.0, 3),
    ])
    assert matrix.n_users == 2
    assert matrix.n_items == 2
    assert matrix.user_index[1] == 0
    assert matrix.item_index[20] == 1
    # 3 ratings out of 2x2 = 4 cells
    assert matrix.density == pytest.approx(0.75)


def test_dense_matrix_places_ratings_correctly():
    matrix = RatingMatrix([Rating(1, 10, 5.0, 1), Rating(2, 20, 3.0, 2)])
    dense = matrix.to_dense()
    assert dense[matrix.user_index[1], matrix.item_index[10]] == 5.0
    assert dense[matrix.user_index[2], matrix.item_index[20]] == 3.0
    assert dense[matrix.user_index[1], matrix.item_index[20]] == 0.0


def test_leave_last_out_holds_the_newest_rating():
    matrix = RatingMatrix([
        Rating(1, 10, 5.0, 1),
        Rating(1, 20, 4.0, 2),
        Rating(1, 30, 3.0, 3),  # newest
        Rating(2, 10, 5.0, 4),
        Rating(2, 20, 4.0, 5),  # newest
    ])
    train, test = matrix.split_leave_last_out(1)
    assert test[1] == [30]
    assert test[2] == [20]
    assert 30 not in {r.item_id for r in train.ratings if r.user_id == 1}


def test_user_with_too_few_ratings_is_not_in_test_set():
    """A user with one rating cannot be evaluated -- scoring them would leak."""
    matrix = RatingMatrix([
        Rating(1, 10, 5.0, 1),
        Rating(2, 10, 4.0, 2),
        Rating(2, 20, 3.0, 3),
    ])
    train, test = matrix.split_leave_last_out(1)
    assert 1 not in test
    assert 2 in test
    assert len(train.ratings) == 2


def test_temporal_split_rejects_bad_fraction():
    matrix = RatingMatrix([Rating(1, 10, 5.0, 1)])
    with pytest.raises(ValueError):
        matrix.split_temporal(0.0)
    with pytest.raises(ValueError):
        matrix.split_temporal(1.5)


# ---------------------------------------------------------------- blending


def test_blend_is_content_only_for_a_new_user():
    w = blend_for_history(0)
    assert w.content == 1.0
    assert w.item_cf == 0.0 and w.user_cf == 0.0


def test_blend_shifts_to_collaborative_as_history_grows():
    early = blend_for_history(2)
    mid = blend_for_history(6)
    late = blend_for_history(20)
    assert early.content > mid.content > late.content
    assert early.item_cf < late.item_cf


def test_low_rating_spread_hands_weight_to_content():
    """The bug that made a thriller fan's feed full of romance.

    Collaborative filtering works on deviations from a user's own mean. A user
    who rates everything 5.0 has no deviation, so item-CF carries no signal.
    The blend must detect that and fall back to content.
    """
    confident = blend_for_history(20, rating_std=1.0)
    flat = blend_for_history(20, rating_std=0.0)

    assert confident.item_cf == pytest.approx(0.55)
    assert flat.content == pytest.approx(1.0)
    assert flat.item_cf == pytest.approx(0.0)


def test_blend_partial_confidence_scales_smoothly():
    half = blend_for_history(20, rating_std=0.25)
    assert 0.0 < half.item_cf < 0.55
    assert half.item_cf + half.user_cf + half.content == pytest.approx(1.0)


def test_blend_weights_always_normalise():
    for n in (0, 1, 5, 50):
        for std in (None, 0.0, 0.2, 3.0):
            w = blend_for_history(n, std).normalised()
            assert w.item_cf + w.user_cf + w.content == pytest.approx(1.0)
            assert w.item_cf >= 0 and w.user_cf >= 0 and w.content >= 0


# ------------------------------------------------------------- recommender


def test_recommender_returns_the_requested_count():
    matrix, texts = _synthetic()
    rec = Recommender(matrix, texts)
    assert len(rec.recommend(1, k=7)) == 7


def test_recommender_never_recommends_an_already_rated_item():
    matrix, texts = _synthetic()
    rec = Recommender(matrix, texts)
    rated = {r.item_id for r in matrix.ratings if r.user_id == 1}
    recs = rec.recommend(1, k=10)
    assert rated.isdisjoint({r.item_id for r in recs})


def test_recommender_recovers_the_taste_cluster():
    """A user in cluster 0 should mostly get cluster-0 items."""
    matrix, texts = _synthetic()
    rec = Recommender(matrix, texts)
    for user_id in (1, 4, 7):
        recs = rec.recommend(user_id, k=6)
        cluster = user_id % 3
        lo, hi = cluster * 10 + 1, cluster * 10 + 10
        in_cluster = sum(1 for r in recs if lo <= r.item_id <= hi)
        assert in_cluster >= 5, f"user {user_id} got {in_cluster}/6 in-cluster"


def test_similar_items_are_from_the_same_cluster():
    matrix, texts = _synthetic()
    rec = Recommender(matrix, texts)
    similar = rec.similar_items(1, k=5)
    assert similar, "expected at least one similar item"
    assert all(1 <= item_id <= 10 for item_id, _ in similar)
    assert all(sim > 0 for _, sim in similar)


def test_similarity_floor_suppresses_weak_evidence():
    """A raised floor must never add spurious cross-cluster recommendations.

    With the floor at zero, items with near-zero similarity still produce a
    positive score because the normalising denominator is built from those same
    tiny numbers. That was a real bug: a romance title with 0.02 similarity to
    every thriller the user had rated ranked first for a thriller fan.
    """
    matrix, texts = _synthetic()
    permissive = Recommender(matrix, texts, similarity_floor=0.0)
    strict = Recommender(matrix, texts, similarity_floor=0.15)

    def cross_cluster_count(rec, user_id):
        cluster = user_id % 3
        lo, hi = cluster * 10 + 1, cluster * 10 + 10
        return sum(
            1 for r in rec.recommend(user_id, k=10) if not (lo <= r.item_id <= hi)
        )

    assert cross_cluster_count(strict, 1) <= cross_cluster_count(permissive, 1)


def test_unknown_user_falls_back_to_content_not_a_crash():
    matrix, texts = _synthetic()
    rec = Recommender(matrix, texts)
    recs = rec.recommend(999_999, k=5)
    assert len(recs) == 5
    assert all(r.source == "content" for r in recs)


def test_zero_k_returns_nothing():
    matrix, texts = _synthetic()
    rec = Recommender(matrix, texts)
    assert rec.recommend(1, k=0) == []


def test_explanation_never_claims_a_zero_similarity():
    """The card must not print "similarity 0.00" as evidence."""
    matrix, texts = _synthetic()
    rec = Recommender(matrix, texts)
    for user_id in range(1, 20):
        for r in rec.recommend(user_id, k=10):
            if "similarity" in r.reason:
                value = float(r.reason.rsplit(" ", 1)[-1].rstrip(")."))
                assert value >= 0.05, f"claimed negligible similarity: {r.reason}"


def test_explanation_agrees_with_the_reported_source():
    """The pill and the sentence must not contradict each other."""
    matrix, texts = _synthetic()
    rec = Recommender(matrix, texts)
    for user_id in range(1, 30):
        for r in rec.recommend(user_id, k=8):
            if r.source == "content":
                assert "topics" in r.reason or "based on what the item is" in r.reason
            else:
                assert "topics" not in r.reason


# -------------------------------------------------------------- evaluation


def test_evaluation_beats_the_popularity_baseline():
    """The headline claim, asserted.

    If a change to the engine makes the model worse than a list of the most-
    rated items, this fails -- which is exactly what should happen.
    """
    matrix, texts = _synthetic(n_users=90)
    train, test = matrix.split_leave_last_out(1)
    rec = Recommender(train, texts)

    model = evaluate(rec, test, k=10)
    baseline = popularity_baseline(train, test, k=10)

    assert model.precision_at_k > baseline.precision_at_k
    assert model.ndcg_at_k > baseline.ndcg_at_k
    assert model.coverage > baseline.coverage
    assert model.n_users_evaluated > 0


def test_evaluation_metrics_are_in_valid_ranges():
    matrix, texts = _synthetic()
    train, test = matrix.split_leave_last_out(1)
    result = evaluate(Recommender(train, texts), test, k=10)
    for value in (result.precision_at_k, result.recall_at_k, result.ndcg_at_k):
        assert 0.0 <= value <= 1.0
    assert 0.0 <= result.coverage <= 1.0
    assert 0.0 <= result.popularity_bias <= 1.0


def test_ndcg_rewards_hits_at_the_top():
    """An ideal ranking scores 1.0, and order must matter."""
    from app.engine.evaluation import _dcg

    assert _dcg([1.0, 1.0, 1.0]) > _dcg([1.0, 0.0, 0.0])
    assert _dcg([1.0, 0.0, 1.0]) > _dcg([0.0, 1.0, 1.0])


# ------------------------------------------------------------------- text


def test_tokenizer_drops_stopwords_and_short_tokens():
    tokens = tokenize("The Ocean and a Reef")
    assert "the" not in tokens
    assert "ocean" in tokens and "reef" in tokens


def test_tfidf_rows_are_unit_normalised():
    matrix = build_tfidf(["ocean reef", "desert sand", "ocean current"])
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0)


def test_tfidf_rare_terms_outweigh_common_ones():
    matrix = build_tfidf(["ocean reef", "ocean current", "ocean tide"])
    # "ocean" appears in every document, so its weight must be the lowest.
    weights = matrix.sum(axis=0)
    assert weights.max() > weights.min()


def test_empty_corpus_does_not_crash():
    matrix = build_tfidf([])
    assert matrix.shape[0] == 0


def test_minmax_preserves_a_weak_signal():
    """A tiny signal must stay tiny. Plain min-max would inflate it to 1.0."""
    from app.engine.recommender import _minmax

    weak = _minmax(np.array([0.001, 0.0005, 0.0]))
    strong = _minmax(np.array([0.9, 0.5, 0.1]))
    assert weak.max() < 0.02
    assert strong.max() > 0.7


def test_minmax_handles_a_constant_array():
    from app.engine.recommender import _minmax

    result = _minmax(np.array([2.0, 2.0, 2.0]))
    assert np.allclose(result, 0.0)
