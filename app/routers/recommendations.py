"""Recommendation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..deps import get_current_user, get_optional_user
from ..engine import blend_for_history
from ..models import Item, Rating, User
from ..schemas import ItemOut, RecommendationOut, RecommendationResponse
from ..service import engine_cache

settings = get_settings()
router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


def _blend_dict(weights) -> dict[str, float]:
    """Blend weights rounded so the three shares sum to exactly 1.0.

    Rounding each weight independently to three places produced 0.42 + 0.14 +
    0.43 = 0.99, and the UI printed "42% / 14% / 43%" -- a visible arithmetic
    error on a page whose whole point is that the numbers are honest. Largest-
    remainder rounding fixes the total without distorting any share by more
    than one thousandth.
    """
    raw = {
        "item_cf": weights.item_cf,
        "user_cf": weights.user_cf,
        "content": weights.content,
    }
    scaled = {k: v * 1000 for k, v in raw.items()}
    floors = {k: int(v) for k, v in scaled.items()}
    remainder = 1000 - sum(floors.values())
    # Hand the leftover thousandths to the largest fractional parts.
    order = sorted(scaled, key=lambda k: scaled[k] - floors[k], reverse=True)
    for key in order[:remainder]:
        floors[key] += 1
    return {k: v / 1000 for k, v in floors.items()}


def _item_map(db: Session, ids: list[int]) -> dict[int, ItemOut]:
    """Load items with their aggregate rating stats.

    The catalogue endpoint always attached an average and a count; this one did
    not, so every recommendation card claimed "no ratings yet" for items with
    fifteen ratings -- the card contradicted the catalogue one click away.
    """
    if not ids:
        return {}
    items = list(db.scalars(select(Item).where(Item.id.in_(ids))))

    rows = db.execute(
        select(
            Rating.item_id,
            func.avg(Rating.value).label("avg"),
            func.count(Rating.id).label("cnt"),
        )
        .where(Rating.item_id.in_(ids))
        .group_by(Rating.item_id)
    ).all()
    stats = {row.item_id: (float(row.avg), int(row.cnt)) for row in rows}

    out: dict[int, ItemOut] = {}
    for item in items:
        avg, cnt = stats.get(item.id, (None, 0))
        payload = ItemOut.model_validate(item)
        payload.average_rating = round(avg, 2) if avg is not None else None
        payload.rating_count = cnt
        out[item.id] = payload
    return out


@router.get("/me", response_model=RecommendationResponse)
def my_recommendations(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    k: int = Query(default=0, ge=0, le=50),
) -> RecommendationResponse:
    """Personalised feed for the signed-in user."""
    count = k or settings.default_recommendation_count
    recommender = engine_cache.get(db)
    recs = recommender.recommend(user.id, k=count)

    history_size = len(recommender.matrix.user_rated_items().get(user.id, set()))
    weights = blend_for_history(history_size, recommender.rating_std(user.id)).normalised()

    items = _item_map(db, [r.item_id for r in recs])
    payload = [
        RecommendationOut(
            item=items[r.item_id], score=r.score, reason=r.reason, source=r.source
        )
        for r in recs
        if r.item_id in items
    ]

    return RecommendationResponse(
        user_id=user.id,
        count=len(payload),
        blend=_blend_dict(weights),
        history_size=history_size,
        recommendations=payload,
    )


@router.get("/for/{user_id}", response_model=RecommendationResponse)
def recommendations_for_user(
    user_id: int,
    db: Session = Depends(get_db),
    viewer: User | None = Depends(get_optional_user),
    k: int = Query(default=0, ge=0, le=50),
) -> RecommendationResponse:
    """Recommendations for any user.

    Only the user themselves or an admin may read another person's feed -- a
    recommendation list is a fairly direct window into someone's taste.
    """
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    if viewer is None or (viewer.id != user_id and not viewer.is_admin):
        raise HTTPException(
            status_code=404, detail="User not found"
        )  # 404, not 403: do not confirm the account exists

    count = k or settings.default_recommendation_count
    recommender = engine_cache.get(db)
    recs = recommender.recommend(user_id, k=count)

    history_size = len(recommender.matrix.user_rated_items().get(user_id, set()))
    weights = blend_for_history(history_size, recommender.rating_std(user_id)).normalised()

    items = _item_map(db, [r.item_id for r in recs])
    payload = [
        RecommendationOut(
            item=items[r.item_id], score=r.score, reason=r.reason, source=r.source
        )
        for r in recs
        if r.item_id in items
    ]

    return RecommendationResponse(
        user_id=user_id,
        count=len(payload),
        blend=_blend_dict(weights),
        history_size=history_size,
        recommendations=payload,
    )


@router.get("/cold-start", response_model=RecommendationResponse)
def cold_start(
    db: Session = Depends(get_db),
    k: int = Query(default=0, ge=0, le=50),
) -> RecommendationResponse:
    """What a brand-new visitor sees: pure content + popularity, no CF."""
    count = k or settings.default_recommendation_count
    recommender = engine_cache.get(db)
    recs = recommender.recommend(-1, k=count)

    items = _item_map(db, [r.item_id for r in recs])
    payload = [
        RecommendationOut(
            item=items[r.item_id], score=r.score, reason=r.reason, source=r.source
        )
        for r in recs
        if r.item_id in items
    ]

    return RecommendationResponse(
        user_id=0,
        count=len(payload),
        blend={"item_cf": 0.0, "user_cf": 0.0, "content": 1.0},
        history_size=0,
        recommendations=payload,
    )
