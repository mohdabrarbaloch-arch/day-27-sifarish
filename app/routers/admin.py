"""Admin endpoints: stats, evaluation, and demo seeding."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import require_admin
from ..engine import evaluate, popularity_baseline
from ..models import Item, Rating, User
from ..schemas import AdminStatsOut, EvaluationOut
from ..service import engine_cache

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats", response_model=AdminStatsOut)
def stats(
    db: Session = Depends(get_db), _admin: User = Depends(require_admin)
) -> AdminStatsOut:
    n_users = db.scalar(select(func.count(User.id))) or 0
    n_items = db.scalar(select(func.count(Item.id))) or 0
    n_ratings = db.scalar(select(func.count(Rating.id))) or 0

    density = n_ratings / (n_users * n_items) if n_users and n_items else 0.0

    recommender = engine_cache.get(db)
    # Coverage of the catalogue by the current model, measured over all users
    # that have any history at all.
    recommended: set[int] = set()
    for user_id in recommender.matrix.user_ids:
        for rec in recommender.recommend(user_id, k=10):
            recommended.add(rec.item_id)
    coverage = len(recommended) / n_items if n_items else 0.0

    cat_rows = db.execute(
        select(Item.category, func.count(Item.id).label("cnt"))
        .group_by(Item.category)
        .order_by(func.count(Item.id).desc())
        .limit(8)
    ).all()

    top_rows = db.execute(
        select(
            Item.id,
            Item.title,
            func.count(Rating.id).label("cnt"),
            func.avg(Rating.value).label("avg"),
        )
        .join(Rating, Rating.item_id == Item.id)
        .group_by(Item.id, Item.title)
        .order_by(func.count(Rating.id).desc())
        .limit(8)
    ).all()

    return AdminStatsOut(
        users=n_users,
        items=n_items,
        ratings=n_ratings,
        density=round(density, 4),
        catalogue_coverage=round(coverage, 4),
        top_categories=[
            {"category": row.category, "count": int(row.cnt)} for row in cat_rows
        ],
        most_rated=[
            {
                "id": int(row.id),
                "title": row.title,
                "rating_count": int(row.cnt),
                "average": round(float(row.avg), 2),
            }
            for row in top_rows
        ],
    )


@router.get("/evaluate", response_model=EvaluationOut)
def run_evaluation(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
    k: int = Query(default=10, ge=1, le=50),
) -> EvaluationOut:
    """Score the live model against a temporal holdout and the popularity baseline."""
    recommender = engine_cache.get(db)
    train, test = recommender.matrix.split_leave_last_out(1)

    from ..engine import Recommender  # local import keeps the module graph flat

    fitted = Recommender(
        matrix=train,
        item_texts=recommender.item_texts,
        shrinkage_lambda=recommender.shrinkage_lambda,
        neighbour_k=recommender.neighbour_k,
    )

    model = evaluate(fitted, test, k=k)
    baseline = popularity_baseline(train, test, k=k)

    def _lift(m: float, b: float) -> float:
        return round((m / b - 1.0) * 100.0, 1) if b > 0 else 0.0

    return EvaluationOut(
        model=model.as_dict(),
        baseline=baseline.as_dict(),
        lift={
            "precision_pct": _lift(model.precision_at_k, baseline.precision_at_k),
            "recall_pct": _lift(model.recall_at_k, baseline.recall_at_k),
            "ndcg_pct": _lift(model.ndcg_at_k, baseline.ndcg_at_k),
            "coverage_pct": _lift(model.coverage, baseline.coverage),
        },
        split="leave-last-out (temporal)",
        k=k,
    )
