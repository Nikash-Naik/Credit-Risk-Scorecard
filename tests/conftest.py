"""
Shared pytest fixtures.

The fixtures here let the test suite run with no external dependencies:
  - synthetic_loans     small synthetic DataFrame matching RAW_INPUT_FEATURES
  - trained_artifacts   real production pipeline trained on the synthetic data,
                        with config paths monkeypatched so the API loads it
  - client              FastAPI TestClient backed by an in-memory SQLite DB
  - auth_headers        Bearer-token headers for an already-registered user
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def synthetic_loans() -> pd.DataFrame:
    """100 synthetic rows covering all RAW_INPUT_FEATURES columns."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "loan_amnt": rng.uniform(1_000, 40_000, n),
        "int_rate": rng.uniform(5.0, 25.0, n),
        "annual_inc": rng.uniform(20_000, 150_000, n),
        "fico_range_low": rng.uniform(640, 800, n),
        "fico_range_high": rng.uniform(645, 805, n),
        "dti": rng.uniform(0.0, 35.0, n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
        "purpose": rng.choice(
            ["debt_consolidation", "credit_card", "home_improvement"], n
        ),
        "term": rng.choice([" 36 months", " 60 months"], n),
        "grade": rng.choice(["A", "B", "C", "D", "E"], n),
    })


@pytest.fixture
def trained_artifacts(synthetic_loans, tmp_path, monkeypatch):
    """Train a tiny pipeline on synthetic data, save it, patch config paths."""
    from app import config
    from app.monitor import router as monitor_router
    from app.predict import router as predict_router
    from train_model import PSI_N_BINS, build_pipeline

    # Synthetic targets: high-rate loans default more often (mild signal).
    rng = np.random.default_rng(0)
    base_prob = (synthetic_loans["int_rate"] - 5) / 25
    targets = (rng.uniform(0, 1, len(synthetic_loans)) < base_prob).astype(int)
    # Guarantee both classes are present.
    targets[0] = 1
    targets[1] = 0

    model = build_pipeline()
    model.fit(synthetic_loans, targets)

    model_path = tmp_path / "credit_risk_model.pkl"
    psi_path = tmp_path / "psi_reference.npy"

    joblib.dump(model, model_path)

    probs = model.predict_proba(synthetic_loans)[:, 1]
    bins = np.linspace(0, 1, PSI_N_BINS + 1)
    counts, _ = np.histogram(probs, bins=bins)
    proportions = counts / counts.sum()
    proportions = np.where(proportions == 0, 1e-6, proportions)
    np.save(psi_path, proportions)

    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    monkeypatch.setattr(config, "PSI_REFERENCE_PATH", psi_path)
    # The router modules captured the original paths at import time.
    monkeypatch.setattr(predict_router, "MODEL_PATH", model_path)
    monkeypatch.setattr(monitor_router, "MODEL_PATH", model_path)
    monkeypatch.setattr(monitor_router, "PSI_REFERENCE_PATH", psi_path)
    # Reset lazy-load caches so each test gets a fresh load.
    monkeypatch.setattr(predict_router, "_model", None)
    monkeypatch.setattr(monitor_router, "_model", None)
    monkeypatch.setattr(monitor_router, "_psi_reference", None)

    return model


@pytest.fixture
def client(monkeypatch):
    """FastAPI TestClient with an in-memory SQLite database."""
    from app import db as db_module
    from app.main import app

    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    db_module.Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)

    def override_get_db():
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[db_module.get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(client) -> dict[str, str]:
    """Register + login a test user; return the Bearer header dict."""
    creds = {"email": "tester@example.com", "password": "testpass123"}
    client.post("/auth/register", json=creds)
    token = client.post("/auth/login", json=creds).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_loan() -> dict:
    """A single valid /predict request body."""
    return {
        "loan_amnt": 15000,
        "int_rate": 13.5,
        "annual_inc": 65000,
        "fico_range_low": 680,
        "fico_range_high": 684,
        "dti": 18.5,
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
        "term": " 36 months",
        "grade": "C",
    }
