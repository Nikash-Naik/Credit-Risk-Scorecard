import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.auth.utils import get_current_user
from app.config import DECISION_THRESHOLD, MODEL_PATH
from app.db import User
from app.predict.schemas import PredictInput, PredictOutput

router = APIRouter(prefix="/predict", tags=["predict"])

_model = None


def _load_model():
    """Lazily load the serialized sklearn pipeline on first request."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model not found at {MODEL_PATH}. "
                    "Run: python src/train_model.py"
                ),
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@router.post("", response_model=PredictOutput)
def predict(body: PredictInput, _: User = Depends(get_current_user)):
    model = _load_model()
    df = pd.DataFrame([body.model_dump()])
    prob = float(model.predict_proba(df)[0][1])
    return PredictOutput(
        default_probability=round(prob, 4),
        decision="Decline" if prob >= DECISION_THRESHOLD else "Approve",
    )
