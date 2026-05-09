"""Integration tests for the /predict endpoint."""
import pytest


@pytest.mark.integration
def test_predict_requires_auth(client, trained_artifacts, sample_loan):
    resp = client.post("/predict", json=sample_loan)
    # HTTPBearer returns 403 in some FastAPI versions, 401 in others -- both mean auth required.
    assert resp.status_code in {401, 403}


@pytest.mark.integration
def test_predict_returns_probability_and_decision(
    client, auth_headers, trained_artifacts, sample_loan,
):
    resp = client.post("/predict", json=sample_loan, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["decision"] in {"Approve", "Decline"}


@pytest.mark.integration
def test_predict_decision_matches_threshold(
    client, auth_headers, trained_artifacts, sample_loan,
):
    """The decision must agree with the configured DECISION_THRESHOLD."""
    from app.config import DECISION_THRESHOLD

    body = client.post("/predict", json=sample_loan, headers=auth_headers).json()
    expected = "Decline" if body["default_probability"] >= DECISION_THRESHOLD else "Approve"
    assert body["decision"] == expected


@pytest.mark.integration
def test_predict_rejects_missing_field(client, auth_headers, trained_artifacts, sample_loan):
    incomplete = {k: v for k, v in sample_loan.items() if k != "grade"}
    resp = client.post("/predict", json=incomplete, headers=auth_headers)
    assert resp.status_code == 422


@pytest.mark.integration
def test_predict_rejects_invalid_token(client, trained_artifacts, sample_loan):
    resp = client.post(
        "/predict",
        json=sample_loan,
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert resp.status_code == 401
