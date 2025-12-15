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
        Validates input data and captures feature names.
        """
        # Capture feature names if input is a DataFrame (needed for OneToOneFeatureMixin)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)

        # Use public check_array instead of internal _validate_data
        X = check_array(X, accept_sparse=False, ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Apply the signed log transformation.
        """
        # Use public check_array instead of internal _validate_data
        X = check_array(X, accept_sparse=False, ensure_2d=True)
        return np.sign(X) * np.log1p(np.abs(X))
