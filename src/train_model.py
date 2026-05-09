"""
Train and serialize the production logistic regression model.

Enhancements over the original:
  - Feature engineering: grade (categorical) + two interaction features
    (dti × int_rate, fico_range_low × loan_amnt) computed by the pipeline.
  - Class imbalance: class_weight="balanced" so the minority default class is
    up-weighted during training without resampling.
  - PSI reference: after training, the in-sample score distribution is binned
    and saved alongside the model so the /monitor/psi endpoint can compare
    future batches against it.

Usage:
    python src/train_model.py
"""
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import InteractionFeatures

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "accepted_2007_to_2018Q4.csv"
MODEL_PATH = ROOT / "models" / "credit_risk_model.pkl"
PSI_REFERENCE_PATH = ROOT / "models" / "psi_reference.npy"

# Raw columns fed to the pipeline from the CSV / API request.
RAW_INPUT_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "fico_range_low",
    "fico_range_high",
    "dti",
    "home_ownership",
    "purpose",
    "term",
    "grade",
]

# After InteractionFeatures runs, the ColumnTransformer sees these columns.
NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "fico_range_low",
    "fico_range_high",
    "dti",
    "dti_x_rate",   # dti × int_rate: high-debt + high-rate compound risk
    "fico_x_amnt",  # fico_range_low × loan_amnt: credit quality vs loan size
]
CATEGORICAL_FEATURES = ["home_ownership", "purpose", "term", "grade"]

BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Late (16-30 days)",
    "Does not meet the credit policy. Status:Charged Off",
}
GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

PSI_N_BINS = 10
RANDOM_STATE = 42



def load_data(path: pathlib.Path = DATA_PATH, n: int | None = None) -> pd.DataFrame:
    print(f"Loading data from {path.name} ...")
    df = pd.read_csv(path, nrows=n, low_memory=False)
    df = df[df["loan_status"].isin(BAD_STATUSES | GOOD_STATUSES)].copy()
    df["target"] = df["loan_status"].isin(BAD_STATUSES).astype(int)
    df["issue_year"] = pd.to_datetime(
        df["issue_d"], format="%b-%Y", errors="coerce"
    ).dt.year
    return df


def build_pipeline() -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("interactions", InteractionFeatures()),
        ("preprocessor", preprocessor),
        # class_weight="balanced" up-weights the minority default class by
        # n_samples / (n_classes * n_samples_per_class) — no resampling needed.
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])


def _save_psi_reference(model: Pipeline, X_train: pd.DataFrame) -> None:
    """Bin in-sample predicted probabilities and persist as the PSI baseline."""
    probs = model.predict_proba(X_train)[:, 1]
    bins = np.linspace(0, 1, PSI_N_BINS + 1)
    counts, _ = np.histogram(probs, bins=bins)
    proportions = counts / counts.sum()
    # Replace empty bins with a small value to avoid log(0) in future PSI math.
    proportions = np.where(proportions == 0, 1e-6, proportions)
    np.save(PSI_REFERENCE_PATH, proportions)
    print(f"PSI reference ({PSI_N_BINS} bins) saved → {PSI_REFERENCE_PATH}")


def main() -> None:
    df = load_data()
    print(f"Labeled rows: {len(df):,}  |  default rate: {df['target'].mean():.1%}")

    train = df[df["issue_year"] <= 2015]
    X_train = train[RAW_INPUT_FEATURES]
    y_train = train["target"]
    print(f"Training on {len(X_train):,} rows (issue years ≤ 2015)")

    model = build_pipeline()
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    _save_psi_reference(model, X_train)


if __name__ == "__main__":
    main()
