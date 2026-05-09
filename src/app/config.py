import os
import pathlib
import secrets

# --- Auth ---
SECRET_KEY = os.environ.get("JWT_SECRET", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Database ---
DATABASE_URL = "sqlite:///./users.db"

# --- Model ---
# Threshold matches the analysis recommendation in notebooks/credit_default_analysis.ipynb
DECISION_THRESHOLD = 0.25

MODEL_PATH = pathlib.Path(__file__).resolve().parents[2] / "models" / "credit_risk_model.pkl"
PSI_REFERENCE_PATH = pathlib.Path(__file__).resolve().parents[2] / "models" / "psi_reference.npy"

# Must match PSI_N_BINS in src/train_model.py — both define the histogram bin count.
PSI_N_BINS = 10
