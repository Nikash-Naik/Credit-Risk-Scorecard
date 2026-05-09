"""Tests for PSI calculation and the /monitor/psi endpoint."""
import numpy as np
import pytest

from app.monitor.router import _compute_psi, _interpret


@pytest.mark.unit
def test_psi_zero_for_identical_distributions():
    dist = np.array([0.1, 0.2, 0.3, 0.4])
    assert _compute_psi(dist, dist) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_psi_positive_for_shifted_distribution():
    expected = np.array([0.7, 0.2, 0.1])
    actual = np.array([0.1, 0.2, 0.7])
    assert _compute_psi(expected, actual) > 0.2


@pytest.mark.unit
def test_psi_handles_zero_bins_without_log_error():
    expected = np.array([0.0, 0.5, 0.5])
    actual = np.array([0.5, 0.0, 0.5])
    # Must not raise; returns a finite positive number.
    psi = _compute_psi(expected, actual)
    assert np.isfinite(psi) and psi > 0


@pytest.mark.unit
@pytest.mark.parametrize("psi,expected", [
    (0.05, "No change"),
    (0.099, "No change"),
    (0.10, "Slight change"),
    (0.19, "Slight change"),
    (0.20, "Significant shift"),
    (0.50, "Significant shift"),
])
def test_interpret_thresholds(psi, expected):
    assert _interpret(psi) == expected


@pytest.mark.integration
def test_monitor_requires_auth(client, trained_artifacts, sample_loan):
    resp = client.post("/monitor/psi", json={"loans": [sample_loan]})
    # HTTPBearer returns 403 in some FastAPI versions, 401 in others.
    assert resp.status_code in {401, 403}


@pytest.mark.integration
def test_monitor_returns_psi_response(
    client, auth_headers, trained_artifacts, sample_loan,
):
    resp = client.post(
        "/monitor/psi",
        json={"loans": [sample_loan, sample_loan, sample_loan]},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_observations"] == 3
    assert body["interpretation"] in {"No change", "Slight change", "Significant shift"}
    assert isinstance(body["bin_proportions"], list)
    assert len(body["bin_proportions"]) == 10  # PSI_N_BINS
    # Proportions sum to ~1 (allowing for the 1e-6 zero-bin floor).
    assert sum(body["bin_proportions"]) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.integration
def test_monitor_rejects_empty_loans(client, auth_headers, trained_artifacts):
    resp = client.post("/monitor/psi", json={"loans": []}, headers=auth_headers)
    assert resp.status_code == 422
