from typing import List

from pydantic import BaseModel

from app.predict.schemas import PredictInput


class PSIRequest(BaseModel):
    loans: List[PredictInput]


class PSIResponse(BaseModel):
    psi: float
    n_observations: int
    interpretation: str           # "No change" | "Slight change" | "Significant shift"
    bin_proportions: List[float]  # current score distribution across PSI_N_BINS bins
