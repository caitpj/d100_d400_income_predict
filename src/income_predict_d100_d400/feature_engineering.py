from typing import Any

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_is_fitted


class SignedLogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a log transformation that handles negative values and zeros.
    Formula: sign(x) * log(1 + |x|)
    """

    def fit(self, X, y=None):
        """
        Validates input data. This transformer is stateless (learns nothing).
        """
        check_array(X)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Apply the signed log transformation.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        return np.sign(X) * np.log1p(np.abs(X))


class LGBMClassifierWithEarlyStopping(LGBMClassifier):
    """
    LGBMClassifier wrapper that automatically creates a validation set
    from the training data to enable early stopping.
    """

    def __init__(
        self,
        early_stopping_round: int = 10,
        test_size: float = 0.1,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.early_stopping_round = early_stopping_round
        self.test_size = test_size
        self.random_state = random_state
        super().__init__(**kwargs)

    def fit(
        self, X: pl.DataFrame, y: Any, **kwargs: Any
    ) -> "LGBMClassifierWithEarlyStopping":
        """
        Fit the model with automatic internal validation split.

        Parameters:
            X: Feature matrix (Polars DataFrame).
            y: Target vector.
            **kwargs: Additional arguments passed to fit (e.g. sample_weight).

        Returns:
            self: The fitted model.
        """
        # Convert to pandas to bypass sklearn/lightgbm compatibility issues
        X_pd = X.to_pandas()

        X_train, X_val, y_train, y_val = train_test_split(
            X_pd,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        super().fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            **kwargs,
        )

        return self
