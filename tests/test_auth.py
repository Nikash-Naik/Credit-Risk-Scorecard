"""Unit tests for auth utilities and integration tests for /auth endpoints."""
import pytest
from jose import jwt

from app.auth.utils import create_access_token, hash_password, verify_password
from app.config import ALGORITHM, SECRET_KEY


@pytest.mark.unit
def test_password_roundtrip():
    hashed = hash_password("mypassword")
    assert verify_password("mypassword", hashed)
    assert not verify_password("wrong", hashed)


@pytest.mark.unit
def test_password_truncates_at_72_bytes():
    """bcrypt hard-limits inputs at 72 bytes; longer passwords must not crash."""
    long_password = "a" * 200
    hashed = hash_password(long_password)
    assert verify_password(long_password, hashed)


@pytest.mark.unit
def test_access_token_is_valid_jwt():
    token = create_access_token("user@example.com")
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"] == "user@example.com"
    assert "exp" in payload


@pytest.mark.integration
def test_register_creates_user(client):
    resp = client.post("/auth/register", json={
        "email": "alice@example.com", "password": "secret123",
    })
    assert resp.status_code == 201


@pytest.mark.integration
def test_register_rejects_duplicate_email(client):
    creds = {"email": "bob@example.com", "password": "secret123"}
    client.post("/auth/register", json=creds)
    resp = client.post("/auth/register", json=creds)
    assert resp.status_code == 400
    assert "already registered" in resp.json()["detail"].lower()


@pytest.mark.integration
def test_login_returns_bearer_token(client):
    creds = {"email": "carol@example.com", "password": "secret123"}
    client.post("/auth/register", json=creds)
    resp = client.post("/auth/login", json=creds)
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


@pytest.mark.integration
def test_login_rejects_wrong_password(client):
    client.post("/auth/register", json={
        "email": "dave@example.com", "password": "right",
    })
    resp = client.post("/auth/login", json={
        "email": "dave@example.com", "password": "wrong",
    })
    assert resp.status_code == 401


@pytest.mark.integration
def test_login_rejects_unknown_user(client):
    resp = client.post("/auth/login", json={
        "email": "ghost@example.com", "password": "anything",
    })
    assert resp.status_code == 401
