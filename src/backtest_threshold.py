"""
Business threshold backtest: expected net profit at each decision threshold.

For each threshold t in [0.05, 0.95] the model declines loans where
predicted default probability >= t.  We then compute:

  avoided_loss     = sum(loan_amnt[TP]) * lgd
                     Bad loans correctly declined -- loss avoided.

  forgone_revenue  = sum(loan_amnt[FP]) * margin
                     Good loans incorrectly declined -- interest forgone.

  net_value        = avoided_loss - forgone_revenue

The optimal threshold maximises net_value given the supplied LGD and margin
assumptions.  Change --lgd and --margin to reflect your portfolio economics.

Defaults
  --lgd    0.70  (70 cents lost per dollar of defaulted principal, typical unsecured)
  --margin 0.06  (6% annualised interest margin on performing loans)

Usage:
    python src/backtest_threshold.py                   # full dataset
    python src/backtest_threshold.py --sample 300000   # faster run
    python src/backtest_threshold.py --lgd 0.8 --margin 0.05
"""
import argparse
import pathlib
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # headless -- no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_model import RAW_INPUT_FEATURES, load_data

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_risk_model.pkl"
OUTPUT_DIR = ROOT / "models"


def _backtest(
    probs: np.ndarray,
    y_true: np.ndarray,
    loan_amnt: np.ndarray,
    lgd: float,
    margin: float,
) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    n_bad = int((y_true == 1).sum())
    rows = []

    for t in thresholds:
        declined = probs >= t

        tp = declined & (y_true == 1)  # bad loans correctly declined
        fp = declined & (y_true == 0)  # good loans incorrectly declined

        avoided_loss = float(loan_amnt[tp].sum() * lgd)
        forgone_rev = float(loan_amnt[fp].sum() * margin)
        net = avoided_loss - forgone_rev

        rows.append({
            "threshold": t,
            "approval_rate": round(float((~declined).mean()), 4),
            "precision": round(float(tp.sum() / declined.sum()) if declined.sum() else 0.0, 4),
            "recall": round(float(tp.sum() / n_bad) if n_bad else 0.0, 4),
            "avoided_loss_usd": round(avoided_loss, 2),
            "forgone_revenue_usd": round(forgone_rev, 2),
            "net_value_usd": round(net, 2),
        })

    return pd.DataFrame(rows)


def _plot(results: pd.DataFrame, out_path: pathlib.Path) -> None:
    opt_t = results.loc[results["net_value_usd"].idxmax(), "threshold"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Business Threshold Backtest (2017-2018 out-of-time test set)", y=1.02)

    ax = axes[0]
    ax.plot(results["threshold"], results["net_value_usd"] / 1e6, color="steelblue")
    ax.axvline(opt_t, color="crimson", linestyle="--", label=f"Optimal t={opt_t:.2f}")
    ax.set_title("Net Value vs Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Net Value ($M)")
    ax.legend()

    ax = axes[1]
    ax.plot(results["threshold"], results["precision"], label="Precision (of declined)")
    ax.plot(results["threshold"], results["recall"], label="Recall (defaults caught)")
    ax.axvline(opt_t, color="crimson", linestyle="--", alpha=0.5)
    ax.set_title("Precision & Recall vs Threshold")
    ax.set_xlabel("Threshold")
    ax.legend()

    ax = axes[2]
    ax.plot(results["threshold"], results["approval_rate"], color="seagreen")
    ax.axvline(opt_t, color="crimson", linestyle="--", alpha=0.5)
    ax.set_title("Approval Rate vs Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Approval Rate")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {out_path}")


def main(n_sample: int | None, lgd: float, margin: float) -> None:
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}. Run train_model.py first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    df = load_data(n=n_sample)
    test = df[df["issue_year"].isin([2017, 2018])]

    if len(test) == 0:
        print("No 2017-2018 rows found in this sample; using the full labeled set.")
        test = df

    X_test = test[RAW_INPUT_FEATURES]
    y_test = test["target"].to_numpy()
    loan_amnt = test["loan_amnt"].to_numpy()

    probs = model.predict_proba(X_test)[:, 1]

    print(f"Test rows   : {len(X_test):,}")
    print(f"Default rate: {y_test.mean():.1%}")
    print(f"Assumptions : LGD={lgd:.0%}  margin={margin:.0%}")
    print()

    results = _backtest(probs, y_test, loan_amnt, lgd, margin)

    best = results.loc[results["net_value_usd"].idxmax()]
    print(f"Optimal threshold : {best['threshold']:.2f}")
    print(f"  Approval rate   : {best['approval_rate']:.1%}")
    print(f"  Precision       : {best['precision']:.3f}  (share of declined that are true defaults)")
    print(f"  Recall          : {best['recall']:.3f}  (share of defaults caught)")
    print(f"  Avoided loss    : ${best['avoided_loss_usd']:>12,.0f}")
    print(f"  Forgone revenue : ${best['forgone_revenue_usd']:>12,.0f}")
    print(f"  Net value       : ${best['net_value_usd']:>12,.0f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "threshold_backtest.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nFull results saved -> {csv_path}")

    _plot(results, OUTPUT_DIR / "threshold_backtest.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business threshold backtest")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Rows to load (default: all).",
    )
    parser.add_argument(
        "--lgd",
        type=float,
        default=0.70,
        help="Loss Given Default as a fraction of principal (default: 0.70).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.06,
        help="Revenue margin on performing loans as a fraction (default: 0.06).",
    )
    args = parser.parse_args()
    main(n_sample=args.sample, lgd=args.lgd, margin=args.margin)
