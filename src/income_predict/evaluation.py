import numpy as np
import pandas as pd
from sklearn.metrics import auc, log_loss


def evaluate_predictions(
    df,
    outcome_column,
    *,
    preds_column=None,
    model=None,
    sample_weight_column=None,
):
    """Evaluate predictions against actual outcomes for binary classification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame used for evaluation
    outcome_column : str
        Name of outcome column (binary: 0/1 or True/False)
    preds_column : str, optional
        Name of predictions column (probabilities), by default None
    model :
        Fitted model with predict_proba method, by default None
    sample_weight_column : str, optional
        Name of sample weight column, by default None

    Returns
    -------
    evals
        DataFrame containing metrics
    """

    evals = {}

    assert (
        preds_column or model
    ), """
    Please either provide the column name of the pre-computed
    predictions or a model to predict from.
    """

    if preds_column is None:
        preds = model.predict_proba(df)[:, 1]
    else:
        preds = df[preds_column]

    if sample_weight_column:
        weights = df[sample_weight_column]
    else:
        weights = np.ones(len(df))

    actuals = df[outcome_column].astype(float)

    evals["mean_preds"] = np.average(preds, weights=weights)
    evals["mean_outcome"] = np.average(actuals, weights=weights)
    evals["bias"] = (evals["mean_preds"] - evals["mean_outcome"]) / evals[
        "mean_outcome"
    ]

    evals["mse"] = np.average((preds - actuals) ** 2, weights=weights)
    evals["rmse"] = np.sqrt(evals["mse"])
    evals["mae"] = np.average(np.abs(preds - actuals), weights=weights)

    # Bernoulli deviance (log loss) for binary classification
    evals["deviance"] = log_loss(actuals, preds, sample_weight=weights)

    ordered_samples, cum_actuals = lorenz_curve(actuals, preds, weights)
    evals["gini"] = 1 - 2 * auc(ordered_samples, cum_actuals)

    return pd.DataFrame(evals, index=[0]).T


def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount
