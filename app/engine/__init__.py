"""Recommendation engine package."""

from .evaluation import EvalResult, evaluate, popularity_baseline
from .matrix import Rating, RatingMatrix
from .recommender import BlendWeights, Recommendation, Recommender, blend_for_history

__all__ = [
    "BlendWeights",
    "EvalResult",
    "Rating",
    "RatingMatrix",
    "Recommendation",
    "Recommender",
    "blend_for_history",
    "evaluate",
    "popularity_baseline",
]
