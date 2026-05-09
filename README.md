# Credit Default Risk Scorecard

A binary classification project that predicts loan default risk using the Lending Club dataset (2.26M loans issued 2007-2018). Compares a logistic regression baseline against an XGBoost challenger, with full leakage control, calibration diagnostics, SHAP explainability, business threshold analysis, and a deployment-ready FastAPI scoring service with JWT authentication and PSI drift monitoring.

## Key findings

- **Logistic regression and XGBoost performed nearly identically** once data leakage was removed (AUC ~0.71 for both). Recommended deployment is logistic regression for simplicity and interpretability.
- **Removing leaky features changed the result dramatically.** With post-origination features (settlement amounts, hardship flags) included, XGBoost appeared to outperform LR (AUC 0.76 vs 0.71). After cleanup, the gap disappeared -- a strong demonstration of why leakage control is the highest-leverage step in credit modeling.
- **Top default risk drivers** (via SHAP): interest rate, loan term length, debt-to-income ratio, FICO score, and loan grade.
- **Calibration was unnecessary.** XGBoost's raw probabilities were already well-calibrated (Brier 0.169 vs 0.172 for Platt-scaled version).
- **Honest test-set evaluation** on 2017-2018 vintages showed AUC degradation to 0.69 -- a realistic out-of-time number that random splits would have hidden.

## Final model performance (validation set, 2016 vintage)

| Metric            | Logistic Regression | XGBoost |
|-------------------|--------------------:|--------:|
| AUC               | 0.711               | 0.708   |
| Gini              | 0.421               | 0.417   |
| KS                | 0.304               | 0.301   |
| Lift @ top decile | 2.15                | 2.11    |
| Brier score       | 0.169               | 0.172   |

## Methodology

1. **Target definition** -- Charged Off, Default, and Late statuses -> bad (1); Fully Paid -> good (0); Current and Grace Period -> dropped (outcome unknown).
2. **Leakage control** -- Dropped 30+ post-origination features including payment history, recoveries, settlements, and hardship plans.
3. **Time-based split** -- Train on loans issued 2007-2015, validate on 2016, test on 2017-2018. Avoids the trap of "predicting the past" that random splits create.
4. **Preprocessing pipeline** -- Median imputation + standardization for numeric features; mode imputation + one-hot encoding for categoricals. Wrapped in sklearn `ColumnTransformer` for reproducibility.
5. **Feature engineering** -- Loan `grade` added as a categorical feature; two interaction terms computed inside the pipeline (`dti x int_rate`, `fico_range_low x loan_amnt`).
6. **Class imbalance** -- 21.5% / 78.5% default-vs-good split handled with `class_weight="balanced"` on logistic regression and `scale_pos_weight` on XGBoost (no resampling).
7. **Models** -- Logistic regression baseline; XGBoost challenger with randomised hyperparameter search.
8. **Calibration** -- Platt scaling via `CalibratedClassifierCV` with 5-fold CV.
9. **Explainability** -- SHAP TreeExplainer for global feature importance (beeswarm) and individual decision waterfalls; logistic regression coefficients for direct linear interpretation.
10. **Threshold analysis** -- Net-value backtest combining avoided loss (TP) against forgone revenue (FP) at every threshold from 0.05 to 0.95.

## Production scripts

| Script                          | Purpose                                                                      |
|---------------------------------|------------------------------------------------------------------------------|
| `src/train_model.py`            | Trains the production logistic regression and saves model + PSI reference.   |
| `src/tune_model.py`             | Randomised XGBoost hyperparameter search using a 2007-2015 / 2016 split.     |
| `src/backtest_threshold.py`     | Business cost-benefit analysis: net value vs threshold on the 2017-2018 set. |

## Deployment -- FastAPI scoring service

The repo includes a production-style FastAPI service that serves the trained model behind JWT authentication.

| Endpoint              | Auth     | Purpose                                                            |
|-----------------------|----------|--------------------------------------------------------------------|
| `POST /auth/register` | none     | Create a user account.                                             |
| `POST /auth/login`    | none     | Exchange email/password for a Bearer token.                        |
| `POST /predict`       | Bearer   | Score a single loan, return default probability + approve/decline. |
| `POST /monitor/psi`   | Bearer   | Compute Population Stability Index for a batch of recent loans.    |

Run it locally:

```bash
python src/app/main.py
# or
uvicorn src.app.main:app --reload
```

Then visit `http://localhost:8000/docs` for the interactive Swagger UI.

The service uses:
- **bcrypt** for password hashing
- **JWT** (HS256) for stateless authentication
- **HTTPBearer** security scheme so Swagger shows a single token field on Authorize
- **SQLAlchemy** with SQLite for user storage
- **Pydantic** for request / response validation

### PSI monitoring

`POST /monitor/psi` accepts a batch of recent loans (same schema as `/predict`), scores them, and compares the resulting score distribution to the in-sample reference saved at training time. Standard interpretation:

| PSI         | Interpretation       | Action                                  |
|-------------|----------------------|-----------------------------------------|
| < 0.10      | No change            | Continue serving.                       |
| 0.10 - 0.20 | Slight change        | Monitor closely.                        |
| > 0.20      | Significant shift    | Investigate; consider retraining.       |

## Tech stack

- Python 3.13
- pandas, numpy
- scikit-learn 1.8
- XGBoost
- SHAP
- matplotlib, seaborn
- FastAPI, SQLAlchemy, Pydantic, python-jose, passlib, bcrypt
- pytest (test suite)

## Project structure

```
credit-risk-scorecard/
|-- README.md
|-- requirements.txt
|-- requirements-api.txt
|-- requirements-dev.txt
|-- pytest.ini
|-- .gitignore
|-- notebooks/
|   `-- credit_default_analysis.ipynb
|-- src/
|   |-- features.py              # InteractionFeatures transformer
|   |-- train_model.py
|   |-- tune_model.py
|   |-- backtest_threshold.py
|   `-- app/
|       |-- main.py
|       |-- config.py
|       |-- db.py
|       |-- auth/
|       |   |-- router.py
|       |   |-- schemas.py
|       |   `-- utils.py
|       |-- predict/
|       |   |-- router.py
|       |   `-- schemas.py
|       `-- monitor/
|           |-- router.py
|           `-- schemas.py
|-- tests/
|   |-- conftest.py
|   |-- test_features.py
|   |-- test_auth.py
|   |-- test_predict.py
|   `-- test_monitor.py
|-- models/                      # gitignored
`-- data/                        # gitignored
```

## How to run

1. Clone the repo.
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and place `accepted_2007_to_2018Q4.csv` in the `data/` folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-api.txt    # FastAPI service
   pip install -r requirements-dev.txt    # tests
   ```
4. Open the notebook to reproduce the analysis:
   ```bash
   jupyter notebook notebooks/credit_default_analysis.ipynb
   ```
5. Train and save a deployment-ready model:
   ```bash
   python src/train_model.py
   ```
6. (Optional) Tune XGBoost hyperparameters:
   ```bash
   python src/tune_model.py --sample 200000 --n-iter 20
   ```
7. (Optional) Run the business threshold backtest:
   ```bash
   python src/backtest_threshold.py --lgd 0.7 --margin 0.06
   ```
8. Serve the model via API:
   ```bash
   python src/app/main.py
   ```
9. Run the test suite:
   ```bash
   pytest
   pytest --cov=src --cov-report=term-missing   # with coverage
   ```

## What I learned

- **Leakage detection is the single most important skill in applied ML.** SHAP caught two leaky features I missed during manual review -- a reminder that explainability tools double as data quality tools.
- **Simpler models often win.** A 110-feature logistic regression matched a 300-tree XGBoost on real out-of-time data. Complexity should be earned, not assumed.
- **Time-based splits change everything.** Random splits would have inflated test scores by ~5 AUC points by leaking future information.
- **Calibration is not always an improvement.** Running diagnostics before applying a technique matters more than applying it by default.
- **A model is not done when it is trained.** Wrapping it in an authenticated API with drift monitoring closes the loop between analysis and deployment.

## Next steps

- Containerise the API with Docker.
- Add automated CI (lint + tests on every push).
- Engineer additional behavioural features on the borrower level when repayment history becomes available.
- Backtest economic value monthly to detect drift in the optimal threshold.

## Dataset

[Lending Club Loan Data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) -- 2.26M accepted loans, 151 features, 2007-2018.
