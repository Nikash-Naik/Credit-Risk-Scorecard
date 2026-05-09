"""
XGBoost hyperparameter tuning with a time-based train / validate split.

  Train    : loans issued 2007–2015
  Validate : loans issued 2016  (held-out vintage, never touches 2017-2018 test)

Runs a randomised search over a broad hyperparameter grid, prints each trial's
validation AUC, then saves the best pipeline to models/credit_risk_model_xgb_tuned.pkl.

Usage:
    python src/tune_model.py                        # 200k rows, 20 trials
    python src/tune_model.py --sample 500000        # larger sample
    python src/tune_model.py --n-iter 40            # more trials
"""
import argparse
import pathlib
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Ensure train_model.py is importable when running from the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_INPUT_FEATURES,
    InteractionFeatures,
    load_data,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
TUNED_MODEL_PATH = ROOT / "models" / "credit_risk_model_xgb_tuned.pkl"

PARAM_GRID = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2],
}


def _build_xgb_pipeline(params: dict, scale_pos_weight: float) -> Pipeline:
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
    clf = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([
        ("interactions", InteractionFeatures()),
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])


def main(n_sample: int | None, n_iter: int) -> None:
    df = load_data(n=n_sample)

    train = df[df["issue_year"] <= 2015]
    val = df[df["issue_year"] == 2016]

    if len(val) == 0:
        print(
            "ERROR: No 2016 validation rows in this sample.\n"
            "Try a larger --sample value (e.g. --sample 500000)."
        )
        sys.exit(1)

    X_train, y_train = train[RAW_INPUT_FEATURES], train["target"]
    X_val, y_val = val[RAW_INPUT_FEATURES], val["target"]

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos

    print(f"Train : {len(X_train):>7,}  default rate: {y_train.mean():.1%}")
    print(f"Val   : {len(X_val):>7,}  default rate: {y_val.mean():.1%}")
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")
    print(f"\nRunning {n_iter} random trials ...\n")

    best_auc = 0.0
    best_params: dict = {}
    best_pipeline: Pipeline | None = None

    sampler = ParameterSampler(PARAM_GRID, n_iter=n_iter, random_state=42)

    for i, params in enumerate(sampler, 1):
        pipeline = _build_xgb_pipeline(params, scale_pos_weight)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)

        val_probs = pipeline.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)

        marker = " <- best" if auc > best_auc else ""
        print(f"[{i:2d}/{n_iter}]  AUC={auc:.4f}{marker}  {params}")

        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_pipeline = pipeline

    print(f"\n{'='*60}")
    print(f"Best val AUC : {best_auc:.4f}")
    print(f"Best params  : {best_params}")

    TUNED_MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, TUNED_MODEL_PATH)
    print(f"Tuned model saved -> {TUNED_MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost hyperparameter tuning")
    parser.add_argument(
        "--sample",
        type=int,
        default=200_000,
        metavar="N",
        help="Rows to load from the CSV (default: 200 000). "
             "Use a larger value for better year coverage.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        metavar="N",
        help="Number of random hyperparameter trials (default: 20).",
    )
    args = parser.parse_args()
    main(n_sample=args.sample, n_iter=args.n_iter)
