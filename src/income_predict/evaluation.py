import numpy as np
import pandas as pd


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray = None,
) -> pd.DataFrame:
    """
    Compute various performance metrics for model predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    sample_weight : np.ndarray, optional
        Sample weights (e.g., exposure). If None, uniform weights are used.

    Returns
    -------
    pd.DataFrame
        DataFrame with metric names as index and their values
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    else:
        sample_weight = np.asarray(sample_weight)

    total_weight = np.sum(sample_weight)

    # 1. Bias: deviation from actual exposure-adjusted mean
    weighted_true_mean = np.sum(y_true * sample_weight) / total_weight
    weighted_pred_mean = np.sum(y_pred * sample_weight) / total_weight
    bias = weighted_pred_mean - weighted_true_mean

    # 2. Deviance (for binary classification, using Bernoulli deviance)
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    log_loss_per_sample = -(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )
    deviance = 2 * np.sum(sample_weight * log_loss_per_sample)

    # 3. Mean Absolute Error (MAE)
    absolute_errors = np.abs(y_true - y_pred)
    mae = np.sum(sample_weight * absolute_errors) / total_weight

    # 4. Root Mean Squared Error (RMSE)
    squared_errors = (y_true - y_pred) ** 2
    mse = np.sum(sample_weight * squared_errors) / total_weight
    rmse = np.sqrt(mse)

    metrics = {
        "Bias": bias,
        "Deviance": deviance,
        "MAE": mae,
        "RMSE": rmse,
    }

    return pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
