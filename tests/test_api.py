"""Tests for the HTTP API: auth, ownership, validation and status codes."""

from __future__ import annotations

# ------------------------------------------------------------------- meta


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_security_headers_present(client):
    res = client.get("/api/health")
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert res.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in res.headers["Content-Security-Policy"]


# ------------------------------------------------------------------- auth


def test_register_returns_201_and_a_token(client):
    res = client.post(
        "/api/auth/register",
        json={
            "email": "new@test.pk",
            "display_name": "New Person",
            "password": "newpass123",
        },
    )
    assert res.status_code == 201
    body = res.json()
    assert body["access_token"]
    assert body["user"]["email"] == "new@test.pk"
    assert "password" not in body["user"]
    assert "password_hash" not in body["user"]


def test_duplicate_email_is_409(client):
    payload = {
        "email": "dup@test.pk",
        "display_name": "Dup",
        "password": "duppass123",
    }
    assert client.post("/api/auth/register", json=payload).status_code == 201
    assert client.post("/api/auth/register", json=payload).status_code == 409


def test_weak_password_is_rejected(client):
    res = client.post(
        "/api/auth/register",
        json={"email": "w@test.pk", "display_name": "W", "password": "12345678"},
    )
    assert res.status_code == 422


def test_invalid_email_is_rejected(client):
    res = client.post(
        "/api/auth/register",
        json={"email": "nope", "display_name": "N", "password": "abc12345"},
    )
    assert res.status_code == 422


def test_login_succeeds_with_correct_password(client, user_token):
    res = client.post(
        "/api/auth/login",
        json={"email": "user@test.pk", "password": "user12345"},
    )
    assert res.status_code == 200
    assert res.json()["access_token"]


def test_login_failure_does_not_reveal_whether_the_account_exists(client, user_token):
    wrong_password = client.post(
        "/api/auth/login",
        json={"email": "user@test.pk", "password": "wrongpass1"},
    ).json()
    no_such_user = client.post(
        "/api/auth/login",
        json={"email": "ghost@test.pk", "password": "wrongpass1"},
    ).json()
    assert wrong_password["detail"] == no_such_user["detail"]


def test_me_requires_a_token(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_rejects_a_garbage_token(client):
    res = client.get("/api/auth/me", headers={"Authorization": "Bearer nonsense"})
    assert res.status_code == 401


def test_me_returns_the_signed_in_user(client, user_token):
    res = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {user_token}"}
    )
    assert res.status_code == 200
    assert res.json()["email"] == "user@test.pk"


# ------------------------------------------------------------------ items


def test_unknown_item_is_404(client):
    assert client.get("/api/items/424242").status_code == 404


def test_only_admins_can_create_items(client, user_token, admin_token):
    payload = {"title": "New Title", "category": "Drama"}

    assert (
        client.post(
            "/api/items", json=payload, headers={"Authorization": f"Bearer {user_token}"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/items", json=payload, headers={"Authorization": f"Bearer {admin_token}"}
        ).status_code
        == 201
    )


def test_item_search_and_category_filter(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    client.post(
        "/api/items",
        json={"title": "Ocean Reef", "category": "Documentary"},
        headers=headers,
    )
    client.post(
        "/api/items",
        json={"title": "Desert Wind", "category": "Drama"},
        headers=headers,
    )

    found = client.get("/api/items?q=ocean").json()
    assert any("Ocean" in i["title"] for i in found)

    docs = client.get("/api/items?category=Documentary").json()
    assert all(i["category"] == "Documentary" for i in docs)


# ---------------------------------------------------------------- ratings


def test_rating_requires_authentication(client):
    res = client.put("/api/ratings", json={"item_id": 1, "value": 4})
    assert res.status_code == 401


def test_rating_out_of_range_is_422(client, user_token, admin_token):
    created = client.post(
        "/api/items",
        json={"title": "Rated Thing", "category": "Drama"},
        headers={"Authorization": f"Bearer {admin_token}"},
    ).json()
    for bad in (0, 6, -1, 99):
        res = client.put(
            "/api/ratings",
            json={"item_id": created["id"], "value": bad},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert res.status_code == 422, f"value {bad} should be rejected"


def test_rating_an_unknown_item_is_404(client, user_token):
    res = client.put(
        "/api/ratings",
        json={"item_id": 999_999, "value": 4},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert res.status_code == 404


def test_rating_twice_updates_rather_than_duplicates(client, user_token, admin_token):
    item = client.post(
        "/api/items",
        json={"title": "Repeat", "category": "Drama"},
        headers={"Authorization": f"Bearer {admin_token}"},
    ).json()
    headers = {"Authorization": f"Bearer {user_token}"}

    first = client.put(
        "/api/ratings", json={"item_id": item["id"], "value": 3}, headers=headers
    ).json()
    second = client.put(
        "/api/ratings", json={"item_id": item["id"], "value": 5}, headers=headers
    ).json()

    assert first["id"] == second["id"]
    assert second["value"] == 5.0
    assert len(client.get("/api/ratings", headers=headers).json()) == 1


# -------------------------------------------------- recommendations & admin


def test_recommendations_require_auth(client):
    assert client.get("/api/recommendations/me").status_code == 401


def test_cold_start_works_without_a_token(client):
    res = client.get("/api/recommendations/cold-start")
    assert res.status_code == 200
    assert res.json()["blend"] == {"item_cf": 0.0, "user_cf": 0.0, "content": 1.0}


def test_a_user_cannot_read_another_users_recommendations(client, user_token, admin_token):
    """A recommendation list is a window into someone's taste, so this is 404.

    404 rather than 403 on purpose: a 403 confirms the account exists.
    """
    admin_me = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {admin_token}"}
    ).json()
    res = client.get(
        f"/api/recommendations/for/{admin_me['id']}",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert res.status_code == 404


def test_admin_stats_requires_admin(client, user_token, admin_token):
    assert client.get("/api/admin/stats").status_code == 401
    assert (
        client.get(
            "/api/admin/stats", headers={"Authorization": f"Bearer {user_token}"}
        ).status_code
        == 403
    )
    res = client.get(
        "/api/admin/stats", headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert res.status_code == 200
    assert "density" in res.json()


def test_evaluation_endpoint_returns_both_sides(client, admin_token):
    res = client.get(
        "/api/admin/evaluate?k=10",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 200
    body = res.json()
    assert "model" in body and "baseline" in body and "lift" in body
    assert body["k"] == 10
    assert "temporal" in body["split"]


def test_docs_are_disabled_in_production(client, monkeypatch):
    """Swagger must not be reachable on a production deployment."""
    import importlib

    monkeypatch.setenv("ENVIRONMENT", "production")
    from app.config import get_settings

    get_settings.cache_clear()
    import app.main as main_module

    importlib.reload(main_module)
    assert main_module.app.docs_url is None
    assert main_module.app.openapi_url is None

    get_settings.cache_clear()
    monkeypatch.setenv("ENVIRONMENT", "development")
    get_settings.cache_clear()
    importlib.reload(main_module)


def test_blend_percentages_sum_to_one_hundred(client, user_token, admin_token):
    """The three shares must add up to exactly 100%, not 99%.

    Rounding each weight independently produced 42 + 14 + 43 = 99 on screen.
    """
    from app.engine import blend_for_history
    from app.routers.recommendations import _blend_dict

    for n in range(0, 30):
        for std in (None, 0.0, 0.1, 0.3, 1.0):
            weights = blend_for_history(n, std).normalised()
            blend = _blend_dict(weights)
            total = sum(blend.values())
            assert abs(total - 1.0) < 1e-9, f"n={n} std={std} summed to {total}"
            assert all(v >= 0 for v in blend.values())
