"""Unit tests for the InteractionFeatures transformer."""
import pandas as pd
import pytest

from features import InteractionFeatures


@pytest.mark.unit
def test_adds_dti_x_rate_column():
    transformer = InteractionFeatures()
    df = pd.DataFrame({
        "dti": [10.0, 20.0],
        "int_rate": [5.0, 12.0],
        "fico_range_low": [700, 720],
        "loan_amnt": [10_000, 25_000],
    })

    out = transformer.transform(df)

    assert list(out["dti_x_rate"]) == [50.0, 240.0]


@pytest.mark.unit
def test_adds_fico_x_amnt_column():
    transformer = InteractionFeatures()
    df = pd.DataFrame({
        "dti": [10.0],
        "int_rate": [5.0],
        "fico_range_low": [700],
        "loan_amnt": [10_000],
    })

    out = transformer.transform(df)

    assert out["fico_x_amnt"].iloc[0] == 7_000_000


@pytest.mark.unit
def test_does_not_mutate_input():
    """transform() must return a new DataFrame, never mutate the caller's copy."""
    transformer = InteractionFeatures()
    df = pd.DataFrame({
        "dti": [10.0],
        "int_rate": [5.0],
        "fico_range_low": [700],
        "loan_amnt": [10_000],
    })

    transformer.transform(df)

    assert "dti_x_rate" not in df.columns
    assert "fico_x_amnt" not in df.columns


@pytest.mark.unit
def test_fit_returns_self():
    transformer = InteractionFeatures()
    assert transformer.fit(pd.DataFrame()) is transformer
