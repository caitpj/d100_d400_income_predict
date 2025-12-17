from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_array


class SignedLogTransformer(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    Applies a log transformation that handles negative values and zeros.
    Formula: sign(x) * log(1 + |x|)
    """

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None
    ) -> "SignedLogTransformer":
        """
        Validate input data and capture feature names for transformation.

        Parameters:
            X: Input data of shape (n_samples, n_features). Can be a numpy array
                or pandas DataFrame. If a DataFrame, feature names are captured.
            y: Ignored. Present for API compatibility with sklearn transformers.

        Returns:
            The fitted transformer instance.
        """
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)

        X = check_array(X, accept_sparse=False, ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Apply the signed log transformation to the input data.

        Transforms each value using the formula: sign(x) * log(1 + |x|).
        This handles negative values and zeros gracefully.

        Parameters:
            X: Input data of shape (n_samples, n_features) to transform.

        Returns:
            Transformed array of the same shape as input, with the signed log
            transformation applied element-wise.
        """
        X = check_array(X, accept_sparse=False, ensure_2d=True)
        return np.sign(X) * np.log1p(np.abs(X))
