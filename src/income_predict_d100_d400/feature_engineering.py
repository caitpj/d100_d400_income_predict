from typing import Any, List, Optional

import polars as pl
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted


class SimpleStandardScaler(BaseEstimator, TransformerMixin):
    """
    A simple StandardScaler that supports Polars DataFrames natively.
    """

    def __init__(self) -> None:
        self.means_: Optional[List[float]] = None
        self.stds_: Optional[List[float]] = None
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X: pl.DataFrame, y: Any = None) -> "SimpleStandardScaler":
        """
        Compute the mean and std to be used for later scaling.

        Parameters:
            X: The input data (Polars DataFrame).
            y: Ignored, exists for compatibility.

        Returns:
            self: The fitted scaler.
        """
        # Convert to pandas for simple calculation (method is on the Polars object)
        X_pd = X.to_pandas()

        self.feature_names_in_ = X_pd.columns.tolist()
        self.means_ = X_pd.mean().tolist()
        self.stds_ = X_pd.std().tolist()

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Perform standardization by centering and scaling.

        Parameters:
            X: The input data to transform (Polars DataFrame).

        Returns:
            The transformed data as a Polars DataFrame.
        """
        check_is_fitted(self, ["means_", "stds_"])

        assert self.feature_names_in_ is not None
        assert self.means_ is not None
        assert self.stds_ is not None

        # Convert to pandas for simple math broadcasting
        X_pd = X.to_pandas()
        X_copy = X_pd.copy()

        for i, col in enumerate(self.feature_names_in_):
            X_copy[col] = (X_copy[col] - self.means_[i]) / self.stds_[i]

        return pl.from_pandas(X_copy)

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get output feature names for transformation.

        Parameters:
            input_features: Ignored, exists for compatibility.

        Returns:
            List of feature names.
        """
        check_is_fitted(self, ["feature_names_in_"])
        assert self.feature_names_in_ is not None
        return self.feature_names_in_


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
