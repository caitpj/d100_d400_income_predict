import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def model_comparrison():
    """Plot comparison of GLM and LGBM evaluation metrics."""
    metrics = ["Mean Prediction", "Bias", "MSE", "RMSE", "MAE", "Deviance", "Gini"]

    glm_values = [0.246782, 0.055333, 0.106909, 0.326970, 0.216439, 0.335232, 0.602056]
    lgbm_values = [0.237601, 0.016069, 0.086969, 0.294905, 0.174007, 0.274437, 0.654425]
    mean_outcome = 0.233843

    # Define what's better for each metric: "lower", "higher", or "closer"
    metric_direction = {
        "Mean Prediction": "closer",
        "Bias": "lower",
        "MSE": "lower",
        "RMSE": "lower",
        "MAE": "lower",
        "Deviance": "lower",
        "Gini": "higher",
    }

    # Determine winner for each metric
    winners = []
    for i, metric in enumerate(metrics):
        direction = metric_direction[metric]
        if direction == "lower":
            winners.append("LGBM" if lgbm_values[i] < glm_values[i] else "GLM")
        elif direction == "higher":
            winners.append("LGBM" if lgbm_values[i] > glm_values[i] else "GLM")
        else:  # closer to mean_outcome
            glm_diff = abs(glm_values[i] - mean_outcome)
            lgbm_diff = abs(lgbm_values[i] - mean_outcome)
            winners.append("LGBM" if lgbm_diff < glm_diff else "GLM")

    x = np.arange(len(metrics))
    width = 0.35

    _, ax = plt.subplots(figsize=(14, 7))

    # Add horizontal line for mean_outcome spanning the mean_preds bars
    ax.hlines(
        y=mean_outcome,
        xmin=x[0] - width,
        xmax=x[0] + width,
        colors="black",
        linestyles="dashed",
        linewidth=2,
    )

    # Add winner stars above the winning bar
    for i, winner in enumerate(winners):
        if winner == "GLM":
            ax.annotate(
                "★",
                xy=(x[i] - width / 2, glm_values[i]),
                ha="center",
                va="bottom",
                fontsize=16,
                color="gold",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
            )
        else:
            ax.annotate(
                "★",
                xy=(x[i] + width / 2, lgbm_values[i]),
                ha="center",
                va="bottom",
                fontsize=16,
                color="gold",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
            )

    # Add direction indicators below metric names
    direction_symbols = {
        "lower": "lower is better",
        "higher": "higher is better",
        "closer": "closer to outcome is better",
    }

    # Create x-axis labels with direction
    metric_labels = [
        f"{m}\n({direction_symbols[metric_direction[m]]})" for m in metrics
    ]

    ax.set_title(
        "Tuned LGBM Beats Tuned GLM In All Evaluation Metrics",
        fontsize=18,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)

    # Custom legend
    star_marker = plt.Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor="gold",
        markeredgecolor="black",
        markeredgewidth=0.5,
        markersize=14,
        label="Winner",
    )
    legend_elements = [
        Patch(facecolor="#5B9BD5", edgecolor="black", label="GLM"),
        Patch(facecolor="#ED7D31", edgecolor="black", label="LGBM"),
        star_marker,
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle="dashed",
            linewidth=2,
            label="mean_outcome",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
