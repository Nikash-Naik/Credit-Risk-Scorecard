from pydantic import BaseModel


class PredictInput(BaseModel):
    loan_amnt: float
    int_rate: float
    annual_inc: float
    fico_range_low: float
    fico_range_high: float
    dti: float
    home_ownership: str   # RENT, OWN, MORTGAGE
    purpose: str          # debt_consolidation, credit_card, home_improvement, etc.
    term: str             # " 36 months" or " 60 months"
    grade: str            # A, B, C, D, E, F, G -- assigned by lender at origination


class PredictOutput(BaseModel):
    default_probability: float
    decision: str         # "Approve" or "Decline"
