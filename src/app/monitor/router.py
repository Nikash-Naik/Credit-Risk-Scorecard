"""
PSI (Population Stability Index) monitoring endpoint.

POST /monitor/psi
  Accepts a batch of loan records (same schema as /predict).
  Scores them with the production model, bins the score distribution into
  PSI_N_BINS equal-width buckets, then compares against the reference
  distribution saved by train_model.py.

PSI interpretation (standard industry thresholds):
  < 0.10  No significant change
  0.10-0.20  Slight change -- worth monitoring
  > 0.20  Significant shift -- investigate and consider retraining
"""
import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.auth.utils import get_current_user
from app.config import MODEL_PATH, PSI_N_BINS, PSI_REFERENCE_PATH
from app.db import User
from app.monitor.schemas import PSIRequest, PSIResponse

router = APIRouter(prefix="/monitor", tags=["monitor"])

_model = None
_psi_reference: np.ndarray | None = None


def _load_artifacts():
    global _model, _psi_reference
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Run: python src/train_model.py",
            )
        _model = joblib.load(MODEL_PATH)
    if _psi_reference is None:
        if not PSI_REFERENCE_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"PSI reference not found at {PSI_REFERENCE_PATH}. "
                    "Run: python src/train_model.py"
                ),
            )
        _psi_reference = np.load(PSI_REFERENCE_PATH)
    return _model, _psi_reference


def _compute_psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI = sum((actual - expected) * ln(actual / expected))"""
    actual = np.where(actual == 0, 1e-6, actual)
    expected = np.where(expected == 0, 1e-6, expected)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _interpret(psi: float) -> str:
    if psi < 0.10:
        return "No change"
    if psi < 0.20:
        return "Slight change"
    return "Significant shift"


@router.post("/psi", response_model=PSIResponse)
def psi_monitor(body: PSIRequest, _: User = Depends(get_current_user)):
    if not body.loans:
        raise HTTPException(status_code=422, detail="loans list must not be empty")

    model, psi_reference = _load_artifacts()

    df = pd.DataFrame([loan.model_dump() for loan in body.loans])
    probs = model.predict_proba(df)[:, 1]

    bins = np.linspace(0, 1, PSI_N_BINS + 1)
    counts, _ = np.histogram(probs, bins=bins)
    actual = counts / counts.sum()
    actual = np.where(actual == 0, 1e-6, actual)

    psi_value = _compute_psi(psi_reference, actual)

    return PSIResponse(
        psi=round(psi_value, 4),
        n_observations=len(body.loans),
        interpretation=_interpret(psi_value),
        bin_proportions=[round(float(p), 4) for p in actual],
    )
