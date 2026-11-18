"""Application settings, loaded from the environment only.

No secret has a usable default. If ``SECRET_KEY`` is missing the app refuses to
start rather than quietly signing tokens with a placeholder -- a default secret
is how a demo becomes a vulnerability.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Sifarish"
    environment: str = "development"
    debug: bool = False

    secret_key: str = Field(..., min_length=32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    database_url: str = "sqlite:///./sifarish.db"

    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    rate_limit_per_minute: int = 120
    login_rate_limit_per_minute: int = 10

    # Engine tuning, exposed so it can be changed without a code edit.
    shrinkage_lambda: float = 5.0
    neighbour_k: int = 40
    default_recommendation_count: int = 12

    @field_validator("secret_key")
    @classmethod
    def _reject_placeholder(cls, value: str) -> str:
        banned = {
            "changeme",
            "secret",
            "your-secret-key",
            "change-me-in-production",
        }
        if value.strip().lower() in banned:
            raise ValueError(
                "SECRET_KEY is a placeholder. Generate one with: "
                "python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
