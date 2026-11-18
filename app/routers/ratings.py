"""Rating endpoints. Writing a rating invalidates the fitted model."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models import Item, Rating, User
from ..schemas import MessageOut, RatingCreate, RatingOut
from ..service import engine_cache

router = APIRouter(prefix="/api/ratings", tags=["ratings"])


@router.get("", response_model=list[RatingOut])
def my_ratings(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> list[RatingOut]:
    rows = db.scalars(
        select(Rating).where(Rating.user_id == user.id).order_by(Rating.updated_at.desc())
    )
    return [RatingOut.model_validate(r) for r in rows]


@router.put("", response_model=RatingOut)
def upsert_rating(
    payload: RatingCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> RatingOut:
    """Create or update. Idempotent by design -- a user has one opinion per item."""
    if db.get(Item, payload.item_id) is None:
        raise HTTPException(status_code=404, detail="Item not found")

    existing = db.scalar(
        select(Rating).where(
            Rating.user_id == user.id, Rating.item_id == payload.item_id
        )
    )

    if existing is None:
        rating = Rating(
            user_id=user.id, item_id=payload.item_id, value=payload.value
        )
        db.add(rating)
    else:
        existing.value = payload.value
        rating = existing

    db.commit()
    db.refresh(rating)
    engine_cache.invalidate()
    return RatingOut.model_validate(rating)


@router.delete("/{item_id}", response_model=MessageOut)
def delete_rating(
    item_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> MessageOut:
    rating = db.scalar(
        select(Rating).where(Rating.user_id == user.id, Rating.item_id == item_id)
    )
    if rating is None:
        raise HTTPException(status_code=404, detail="Rating not found")
    db.delete(rating)
    db.commit()
    engine_cache.invalidate()
    return MessageOut(detail="Rating removed")
