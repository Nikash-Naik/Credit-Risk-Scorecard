import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Adds multiplicative interaction columns before the ColumnTransformer.

    Defined in its own module so joblib pickles a stable import path
    (features.InteractionFeatures) rather than __main__.InteractionFeatures.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["dti_x_rate"] = X["dti"] * X["int_rate"]
        X["fico_x_amnt"] = X["fico_range_low"] * X["loan_amnt"]
        return X
