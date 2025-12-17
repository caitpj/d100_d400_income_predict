from typing import Any, Optional

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

from income_predict_d100_d400.plotting import (
    plot_calibration_curve,
    plot_confusion_matrices,
    plot_partial_dependence,
    plot_roc_curve,
)


def evaluate_predictions(
    df: pl.DataFrame,
    outcome_column: str,
    *,
    preds_column: Optional[str] = None,
    model: Optional[Any] = None,
    sample_weight_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Evaluate model predictions against actual outcomes.

    Calculates various metrics including bias, MSE, RMSE, MAE, deviance, and Gini
    coefficient. Predictions can be provided either as a column in the DataFrame
    or generated from a model.

    Parameters:
        df: The DataFrame containing actual outcomes and optionally predictions.
        outcome_column: Name of the column containing actual outcome values.
        preds_column: Name of the column containing pre-computed predictions.
            Mutually exclusive with `model`.
        model: A fitted model with a `predict_proba` method to generate predictions.
            Mutually exclusive with `preds_column`.
        sample_weight_column: Name of the column containing sample weights.
            If None, uniform weights are used.

    Returns:
        A DataFrame with metrics as rows, containing columns 'metric' and 'value'.
        Metrics include: mean_preds, mean_outcome, bias, mse, rmse, mae, deviance, gini.

    Raises:
        ValueError: If neither `preds_column` nor `model` is provided.
    """

    evals = {}

    if preds_column is not None:
        preds = df[preds_column].to_numpy()
    elif model is not None:
        preds = model.predict_proba(df)[:, 1]
    else:
        raise ValueError(
            "provide column name of the pre-computed predictions or model to predict from."
        )

    if sample_weight_column:
        weights = df[sample_weight_column].to_numpy()
    else:
        weights = np.ones(len(df))

    actuals = df[outcome_column].cast(pl.Float64).to_numpy()

    evals["mean_preds"] = np.average(preds, weights=weights)
    evals["mean_outcome"] = np.average(actuals, weights=weights)
    evals["bias"] = (evals["mean_preds"] - evals["mean_outcome"]) / evals[
        "mean_outcome"
    ]

    evals["mse"] = np.average((preds - actuals) ** 2, weights=weights)
    evals["rmse"] = np.sqrt(evals["mse"])
    evals["mae"] = np.average(np.abs(preds - actuals), weights=weights)

    evals["deviance"] = log_loss(actuals, preds, sample_weight=weights)

    auc_score = roc_auc_score(actuals, preds, sample_weight=weights)
    evals["gini"] = 2 * auc_score - 1

    return pl.DataFrame(evals).transpose(
        include_header=True, header_name="metric", column_names=["value"]
    )


def get_feature_importance(
    importances: np.ndarray, feature_names: np.ndarray
) -> pl.DataFrame:
    """
    Create a sorted DataFrame of feature importances.

    Parameters:
        importances: Array of importance values for each feature.
        feature_names: Array of feature names corresponding to the importances.

    Returns:
        A DataFrame with columns 'feature' and 'importance', sorted by importance
        in descending order.
    """

    return pl.DataFrame({"feature": feature_names, "importance": importances}).sort(
        "importance", descending=True
    )


def run_evaluation(
    test_df: pl.DataFrame,
    target: str,
    glm_model: Any,
    lgbm_model: Any,
    train_features: pl.DataFrame,
) -> None:
    """
    Run full evaluation pipeline for GLM and LGBM models.

    Generates predictions on the test set, computes evaluation metrics,
    and produces visualizations including confusion matrices, calibration curves,
    ROC curves, feature importance rankings, and partial dependence plots.

    Parameters:
        test_df: The test DataFrame containing features, target, and unique_id.
        target: Name of the target column.
        glm_model: A fitted GLM sklearn Pipeline with 'preprocessor' and 'classifier' steps.
        lgbm_model: A fitted LGBM sklearn Pipeline with 'preprocessor' and 'classifier' steps.
        train_features: Training features DataFrame used for partial dependence plots.

    Returns:
        None. Prints evaluation metrics and saves plots to the plots directory.
    """

    test_X = test_df.drop([target, "unique_id"])
    test_y = test_df[target]

    test_eval_df = test_X.with_columns(
        test_y.alias(target),
        pl.Series("glm_preds", glm_model.predict_proba(test_X)[:, 1]),
        pl.Series("lgbm_preds", lgbm_model.predict_proba(test_X)[:, 1]),
    )

    glm_eval = evaluate_predictions(test_eval_df, target, preds_column="glm_preds")
    print("\nTuned GLM Evaluation Metrics:")
    print(glm_eval)

    lgbm_eval = evaluate_predictions(test_eval_df, target, preds_column="lgbm_preds")
    print("\nTuned LGBM Evaluation Metrics:")
    print(lgbm_eval)

    plot_confusion_matrices(
        test_eval_df[target].cast(pl.Int32).to_numpy(),
        test_eval_df["glm_preds"].to_numpy(),
        test_eval_df["lgbm_preds"].to_numpy(),
    )

    plot_calibration_curve(
        test_eval_df[target].cast(pl.Int32).to_numpy(),
        test_eval_df["glm_preds"].to_numpy(),
        test_eval_df["lgbm_preds"].to_numpy(),
    )

    plot_roc_curve(
        test_eval_df[target].cast(pl.Int32).to_numpy(),
        test_eval_df["glm_preds"].to_numpy(),
        test_eval_df["lgbm_preds"].to_numpy(),
    )

    glm_preprocessor = glm_model.named_steps["preprocessor"]
    glm_transformed_names = glm_preprocessor.get_feature_names_out()

    glm_clf = glm_model.named_steps["classifier"]
    glm_importance = get_feature_importance(
        np.abs(glm_clf.coef_).flatten(), glm_transformed_names
    )
    print("\nTuned GLM Feature Importance (Top 5):")
    print(glm_importance.head(5))

    lgbm_preprocessor = lgbm_model.named_steps["preprocessor"]
    lgbm_transformed_names = lgbm_preprocessor.get_feature_names_out()

    lgbm_clf = lgbm_model.named_steps["classifier"]
    lgbm_importance = get_feature_importance(
        lgbm_clf.feature_importances_, lgbm_transformed_names
    )
    print("\nTuned LGBM Feature Importance (Top 5):")
    print(lgbm_importance.head(5))

    top_features = lgbm_importance.head(5)["feature"].to_list()

    original_features = []
    for feat in top_features:
        if feat.startswith("cat__"):
            original = feat.replace("cat__", "").rsplit("_", 1)[0]
        elif feat.startswith("num__"):
            original = feat.replace("num__", "")
        else:
            original = feat
        if original not in original_features and original in train_features.columns:
            original_features.append(original)

    pdp_sample_size = min(1000, len(train_features))
    pdp_data = train_features.sample(n=pdp_sample_size, seed=42)

    plot_partial_dependence(glm_model, lgbm_model, pdp_data, original_features[:5])
