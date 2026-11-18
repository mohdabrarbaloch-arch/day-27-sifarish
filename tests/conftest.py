"""Shared pytest fixtures.

Each test gets its own in-memory database so nothing leaks between tests, and
``SECRET_KEY`` is set before the app package is imported.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-for-validation")
os.environ.setdefault("ENVIRONMENT", "development")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(scope="session")
def db_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture()
def client(db_path, monkeypatch):
    """A TestClient backed by a fresh SQLite file."""
    from app import database
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    get_settings.cache_clear()

    test_engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    monkeypatch.setattr(database, "engine", test_engine)
    monkeypatch.setattr(
        database,
        "SessionLocal",
        sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False),
    )

    from app import deps

    # The rate limiters are module-level singletons keyed by client IP, and
    # every TestClient request comes from the same "testclient" host. Without
    # this reset the login limiter (10/min) trips partway through the suite and
    # later tests fail with 429 instead of the status they are asserting on.
    deps.general_limiter.reset()
    deps.login_limiter.reset()

    from app.main import app

    database.Base.metadata.drop_all(bind=test_engine)
    database.Base.metadata.create_all(bind=test_engine)

    with TestClient(app) as test_client:
        yield test_client

    database.Base.metadata.drop_all(bind=test_engine)


@pytest.fixture()
def admin_token(client) -> str:
    from app.database import SessionLocal
    from app.models import User
    from app.security import hash_password

    db = SessionLocal()
    db.add(
        User(
            email="admin@test.pk",
            display_name="Admin",
            password_hash=hash_password("admin12345"),
            is_admin=True,
        )
    )
    db.commit()
    db.close()

    res = client.post(
        "/api/auth/login",
        json={"email": "admin@test.pk", "password": "admin12345"},
    )
    return res.json()["access_token"]


@pytest.fixture()
def user_token(client) -> str:
    res = client.post(
        "/api/auth/register",
        json={
            "email": "user@test.pk",
            "display_name": "Test User",
            "password": "user12345",
        },
    )
    return res.json()["access_token"]
