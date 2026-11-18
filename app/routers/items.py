"""Catalogue endpoints: browse, search, create, and item similarity."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import require_admin
from ..models import Item, Rating, User
from ..schemas import ItemCreate, ItemOut, SimilarItemOut
from ..service import engine_cache

router = APIRouter(prefix="/api/items", tags=["items"])


def _with_stats(db: Session, items: list[Item]) -> list[ItemOut]:
    """Attach average rating and count in one query, not N."""
    if not items:
        return []
    ids = [i.id for i in items]
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

    out: list[ItemOut] = []
    for item in items:
        avg, cnt = stats.get(item.id, (None, 0))
        payload = ItemOut.model_validate(item)
        payload.average_rating = round(avg, 2) if avg is not None else None
        payload.rating_count = cnt
        out.append(payload)
    return out


@router.get("", response_model=list[ItemOut])
def list_items(
    db: Session = Depends(get_db),
    q: str | None = Query(default=None, max_length=120),
    category: str | None = Query(default=None, max_length=80),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[ItemOut]:
    stmt = select(Item)
    if q:
        needle = f"%{q.strip().lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(Item.title).like(needle),
                func.lower(Item.description).like(needle),
                func.lower(Item.tags).like(needle),
            )
        )
    if category:
        stmt = stmt.where(func.lower(Item.category) == category.strip().lower())

    items = list(db.scalars(stmt.order_by(Item.id).limit(limit).offset(offset)))
    return _with_stats(db, items)


@router.get("/categories", response_model=list[str])
def list_categories(db: Session = Depends(get_db)) -> list[str]:
    rows = db.scalars(select(Item.category).distinct().order_by(Item.category))
    return list(rows)


@router.get("/{item_id}", response_model=ItemOut)
def get_item(item_id: int, db: Session = Depends(get_db)) -> ItemOut:
    item = db.get(Item, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return _with_stats(db, [item])[0]


@router.get("/{item_id}/similar", response_model=list[SimilarItemOut])
def similar_items(
    item_id: int,
    db: Session = Depends(get_db),
    k: int = Query(default=8, ge=1, le=50),
) -> list[SimilarItemOut]:
    if db.get(Item, item_id) is None:
        raise HTTPException(status_code=404, detail="Item not found")

    recommender = engine_cache.get(db)
    pairs = recommender.similar_items(item_id, k=k)
    if not pairs:
        return []

    ids = [pid for pid, _ in pairs]
    items = {i.id: i for i in db.scalars(select(Item).where(Item.id.in_(ids)))}
    enriched = _with_stats(db, [items[pid] for pid, _ in pairs if pid in items])
    by_id = {e.id: e for e in enriched}

    return [
        SimilarItemOut(item=by_id[pid], similarity=sim)
        for pid, sim in pairs
        if pid in by_id
    ]


@router.post("", response_model=ItemOut, status_code=status.HTTP_201_CREATED)
def create_item(
    payload: ItemCreate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> ItemOut:
    item = Item(
        title=payload.title.strip(),
        category=payload.category.strip(),
        description=payload.description.strip(),
        tags=payload.tags.strip(),
        year=payload.year,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    engine_cache.invalidate()
    return _with_stats(db, [item])[0]


@router.delete(
    "/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_item(
    item_id: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> Response:
    item = db.get(Item, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    db.delete(item)
    db.commit()
    engine_cache.invalidate()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
