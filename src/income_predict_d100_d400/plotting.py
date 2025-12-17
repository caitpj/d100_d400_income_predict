from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import partial_dependence
from sklearn.metrics import auc, confusion_matrix, roc_curve

from income_predict_d100_d400.robust_paths import PLOTS_DIR


def _save_plot(name: Optional[str] = None) -> None:
    """
    Save current figure to file instead of displaying.

    Parameters:
        name: Name for the file.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = PLOTS_DIR / f"{name}.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {filepath}")


def plot_roc_curve(
    y_true: np.ndarray, glm_preds: np.ndarray, lgbm_preds: np.ndarray
) -> None:
    """
    Plots ROC curves for GLM and LGBM models.
    Saves the plot to file.

    Parameters:
        y_true: Array of true labels.
        glm_preds: Array of GLM prediction probabilities.
        lgbm_preds: Array of LGBM prediction probabilities.
    """
    plt.figure(figsize=(8, 6))

    # Plot diagonal line (random guess)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")

    # GLM
    fpr_glm, tpr_glm, _ = roc_curve(y_true, glm_preds)
    roc_auc_glm = auc(fpr_glm, tpr_glm)
    plt.plot(
        fpr_glm,
        tpr_glm,
        label=f"GLM (AUC = {roc_auc_glm:.3f})",
        color="orange",
        linewidth=2,
    )

    # LGBM
    fpr_lgbm, tpr_lgbm, _ = roc_curve(y_true, lgbm_preds)
    roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)
    plt.plot(
        fpr_lgbm,
        tpr_lgbm,
        label=f"LGBM (AUC = {roc_auc_lgbm:.3f})",
        color="blue",
        linewidth=2,
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_plot(name="roc_curve_plot")


def plot_calibration_curve(
    y_true: np.ndarray, glm_preds: np.ndarray, lgbm_preds: np.ndarray
) -> None:
    """
    Plots calibration curves (reliability diagrams) for GLM and LGBM models.
    Saves the plot to file.

    Parameters:
        y_true: Array of true labels.
        glm_preds: Array of GLM prediction probabilities.
        lgbm_preds: Array of LGBM prediction probabilities.
    """
    plt.figure(figsize=(8, 6))

    # Plot perfectly calibrated line
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")

    # Calculate and plot GLM calibration
    prob_true_glm, prob_pred_glm = calibration_curve(y_true, glm_preds, n_bins=10)
    plt.plot(
        prob_pred_glm,
        prob_true_glm,
        marker="s",
        linestyle="-",
        label="GLM",
        color="orange",
    )

    # Calculate and plot LGBM calibration
    prob_true_lgbm, prob_pred_lgbm = calibration_curve(y_true, lgbm_preds, n_bins=10)
    plt.plot(
        prob_pred_lgbm,
        prob_true_lgbm,
        marker="s",
        linestyle="-",
        label="LGBM",
        color="blue",
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_plot(name="calibration_plot")


def plot_partial_dependence(
    glm_model: Any,
    lgbm_model: Any,
    X: Union[pl.DataFrame, pd.DataFrame],
    top_features: List[str],
) -> None:
    """
    Plot partial dependence for top features with both GLM and LGBM models.
    Saves a separate plot for each feature.

    Parameters:
        glm_model: The fitted GLM model.
        lgbm_model: The fitted LGBM model.
        X: The dataset used for computing partial dependence.
        top_features: List of feature names to plot.
    """
    # Convert to pandas to avoid sklearn partial_dependence indexing issues with Polars
    if hasattr(X, "to_pandas"):
        X_plot = X.to_pandas()
    else:
        X_plot = X.copy()

    int_cols = X_plot.select_dtypes(include=["int", "integer"]).columns
    X_plot[int_cols] = X_plot[int_cols].astype(float)

    for feature in top_features:
        # Create a new figure for each feature
        _, ax = plt.subplots(figsize=(6, 4))

        is_categorical = (
            X_plot[feature].dtype == "object"
            or str(X_plot[feature].dtype) == "category"
        )

        lgbm_pd = partial_dependence(
            lgbm_model,
            X_plot,
            features=[feature],
            kind="average",
            categorical_features=[feature] if is_categorical else None,
        )
        lgbm_grid = lgbm_pd["grid_values"][0]
        lgbm_avg = lgbm_pd["average"][0]

        glm_pd = partial_dependence(
            glm_model,
            X_plot,
            features=[feature],
            kind="average",
            categorical_features=[feature] if is_categorical else None,
        )
        glm_grid = glm_pd["grid_values"][0]
        glm_avg = glm_pd["average"][0]

        if is_categorical:
            # For categorical features, bar plot with offset
            x_positions = np.arange(len(lgbm_grid))
            width = 0.35

            ax.bar(
                x_positions - width / 2,
                lgbm_avg,
                width,
                color="blue",
                label="LGBM",
                alpha=0.8,
            )
            ax.bar(
                x_positions + width / 2,
                glm_avg,
                width,
                color="orange",
                label="GLM",
                alpha=0.8,
            )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(lgbm_grid, rotation=45, ha="right")
        else:
            # For numeric features, line plot
            ax.plot(lgbm_grid, lgbm_avg, color="blue", label="LGBM", linewidth=2)
            ax.plot(glm_grid, glm_avg, color="orange", label="GLM", linewidth=2)

        ax.set_xlabel(feature)
        ax.set_ylabel("Partial Dependence")
        ax.legend(loc="best")
        ax.set_title(f"Partial Dependence: {feature}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        _save_plot(name=f"partial_dependence_{feature}")


def plot_confusion_matrices(
    y_true: np.ndarray, glm_preds: np.ndarray, lgbm_preds: np.ndarray
) -> None:
    """
    Plots confusion matrix heatmaps for GLM and LGBM side by side.

    Parameters:
        y_true: Array of true labels.
        glm_preds: Array of GLM prediction probabilities.
        lgbm_preds: Array of LGBM prediction probabilities.
    """
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, preds, title in zip(
        axes,
        [glm_preds, lgbm_preds],
        ["Tuned GLM", "Tuned LGBM"],
    ):
        y_pred = (preds >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        cm_pct = cm / cm.sum() * 100

        annotations = np.array(
            [
                [f"{count}\n({pct:.1f}%)" for count, pct in zip(row_counts, row_pcts)]
                for row_counts, row_pcts in zip(cm, cm_pct)
            ]
        )

        sns.heatmap(
            cm,
            annot=annotations,
            fmt="",
            cmap="Blues",
            xticklabels=["<=50K", ">50K"],
            yticklabels=["<=50K", ">50K"],
            ax=ax,
            cbar=False,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)

    plt.suptitle("Confusion Matrices: Predicted vs Actual", y=1.02)
    plt.tight_layout()
    _save_plot(name="classification_plot")
