"""Request and response schemas.

Every input is bounded. An unbounded string field is a denial-of-service vector
and a storage bill waiting to happen.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class UserCreate(BaseModel):
    email: EmailStr
    display_name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("display_name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("display_name cannot be blank")
        return cleaned

    @field_validator("password")
    @classmethod
    def _password_strength(cls, value: str) -> str:
        if value.isdigit() or value.isalpha():
            raise ValueError("password must mix letters and numbers")
        return value


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    display_name: str
    is_admin: bool
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    category: str
    description: str
    tags: str
    year: int | None
    average_rating: float | None = None
    rating_count: int = 0


class ItemCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=4000)
    tags: str = Field(default="", max_length=400)
    year: int | None = Field(default=None, ge=1800, le=2100)


class RatingCreate(BaseModel):
    item_id: int = Field(gt=0)
    value: float = Field(ge=1.0, le=5.0)


class RatingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    item_id: int
    value: float
    created_at: datetime
    updated_at: datetime


class RecommendationOut(BaseModel):
    item: ItemOut
    score: float
    reason: str
    source: str


class RecommendationResponse(BaseModel):
    user_id: int
    count: int
    blend: dict[str, float]
    history_size: int
    recommendations: list[RecommendationOut]


class SimilarItemOut(BaseModel):
    item: ItemOut
    similarity: float


class EvaluationOut(BaseModel):
    model: dict[str, float | int]
    baseline: dict[str, float | int]
    lift: dict[str, float]
    split: str
    k: int


class AdminStatsOut(BaseModel):
    users: int
    items: int
    ratings: int
    density: float
    catalogue_coverage: float
    top_categories: list[dict[str, str | int]]
    most_rated: list[dict[str, str | int | float]]


class MessageOut(BaseModel):
    detail: str
