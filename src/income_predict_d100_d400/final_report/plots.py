"""Visualisations for the final project report."""

from typing import Dict, List, Optional, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch


def confusion_matrix() -> None:
    """Plot confusion matrices for tuned GLM and tuned LGBM models."""
    # Hardcoded confusion matrix values for tuned models
    glm_tuned_cm = np.array([[7021, 495], [1025, 1269]])
    lgbm_tuned_cm = np.array([[7064, 452], [796, 1498]])

    # Calculate percentages
    glm_total = glm_tuned_cm.sum()
    lgbm_total = lgbm_tuned_cm.sum()
    glm_pct = glm_tuned_cm / glm_total * 100
    lgbm_pct = lgbm_tuned_cm / lgbm_total * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["<=50K", ">50K"]

    for idx, (ax, cm, pct, title) in enumerate(
        zip(
            axes,
            [glm_tuned_cm, lgbm_tuned_cm],
            [glm_pct, lgbm_pct],
            ["Tuned GLM", "Tuned LGBM"],
        )
    ):
        # Create heatmap
        ax.imshow(cm, cmap="Blues", aspect="auto")

        # Add text annotations
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                percent = pct[i, j]
                # Use white text for dark cells, dark text for light cells
                text_color = "white" if cm[i, j] > cm.max() / 2 else "darkblue"
                ax.text(
                    j,
                    i,
                    f"{count}\n({percent:.1f}%)",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                )

        # Add star to LGBM cells (idx=1)
        if idx == 1:
            for i in range(2):
                for j in range(2):
                    ax.annotate(
                        "★",
                        xy=(j, i - 0.3),
                        ha="center",
                        va="center",
                        fontsize=14,
                        color="gold",
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
                    )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(title, fontsize=13)

    # Add legend for star
    star_marker = plt.Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor="gold",
        markeredgecolor="black",
        markeredgewidth=0.5,
        markersize=14,
        label="Best",
    )
    fig.legend(handles=[star_marker], loc="upper right", fontsize=9)

    plt.suptitle("Confusion Matrices: Predicted vs Actual", y=1.02, fontsize=14)
    plt.tight_layout()


def correlation_compare(df: pl.DataFrame) -> None:
    """
    Plots strip plots for unique_id and age against the target to show pattern contrast.
    """
    target = "high_income"
    # Only plot these two specific columns
    features = ["unique_id", "age"]

    # Check if columns exist
    for col in features + [target]:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in DataFrame!")
            return

    # Convert to pandas for seaborn compatibility
    df_pd = df.select(features + [target]).to_pandas()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: unique_id (no pattern)
    sns.stripplot(
        data=df_pd,
        x=target,
        y="unique_id",
        jitter=0.45,
        alpha=0.05,
        legend=False,
        ax=ax1,
    )
    ax1.set_title("Unique ID (No Pattern)", fontsize=13, fontweight="bold")
    ax1.set_xlabel(target.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax1.set_ylabel("Unique ID", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Right plot: age (clear pattern)
    sns.stripplot(
        data=df_pd, x=target, y="age", jitter=0.45, alpha=0.05, legend=False, ax=ax2
    )
    ax2.set_title("Age (Clear Pattern)", fontsize=13, fontweight="bold")
    ax2.set_xlabel(target.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax2.set_ylabel("Age", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Overall title
    fig.suptitle(
        "Pattern Contrast: No Relationship vs. Clear Separation",
        fontsize=15,
        fontweight="bold",
        y=1.00,
    )

    plt.tight_layout()
    plt.show()


def display_dataset(df: pl.DataFrame) -> None:
    """
    Creates a pretty table displaying dataset information including shape,
    data types, and unique values.

    Parameters:
    -----------
    df : polars.DataFrame
        The DataFrame to analyze

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Gather information
    info_data = {
        "Column": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Unique Values": [df[col].n_unique() for col in df.columns],
    }

    info_df = pd.DataFrame(info_data)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, max(6, len(df.columns) * 0.35 + 1)))
    ax.axis("tight")
    ax.axis("off")

    # Create the table - position it lower to avoid title overlap
    table = ax.table(
        cellText=info_df.values,
        colLabels=info_df.columns,
        cellLoc="left",
        loc="upper center",
        colWidths=[0.4, 0.3, 0.3],
        bbox=[0, 0, 1, 0.95],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(info_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor("#3498db")
        cell.set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(info_df) + 1):
        for j in range(len(info_df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#ecf0f1")
            else:
                cell.set_facecolor("white")

    # Add title with dataset shape
    title_text = f"Dataset Information: {df.height:,} rows, {df.width} columns"
    plt.title(title_text, fontsize=14, fontweight="bold", pad=30, y=0.98)

    plt.tight_layout()
    plt.show()


def distribution_variety(
    df: pl.DataFrame,
) -> Optional[Tuple[Figure, Tuple[Axes, Axes]]]:
    """
    Visualizes the distribution comparison between two columns:
    1. 'age' - showing smooth distribution
    2. 'hours-per-week' - showing sharp peaks

    Parameters:
    -----------
    df : polars.DataFrame
        The DataFrame containing 'age' and 'hours-per-week' columns

    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    # Check if both columns exist
    if "age" not in df.columns or "hours-per-week" not in df.columns:
        print("Error: Required columns not found in DataFrame!")
        return None

    # Get the data for both columns, removing null values
    # Convert to pandas Series to reuse the complex plotting logic below
    age_data = df["age"].cast(pl.Float64, strict=False).drop_nulls().to_pandas()
    hours_data = (
        df["hours-per-week"].cast(pl.Float64, strict=False).drop_nulls().to_pandas()
    )

    if len(age_data) == 0 or len(hours_data) == 0:
        print("No valid numeric data found in required columns!")
        return None

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # LEFT PLOT: Age - Smooth distribution
    age_counts = age_data.value_counts().sort_index()

    # Fill in missing integer values with 0 to ensure one bar per integer
    age_min = int(age_data.min())
    age_max = int(age_data.max())
    all_ages = pd.Series(0, index=range(age_min, age_max + 1))
    all_ages.update(age_counts)
    age_counts = all_ages

    # Find bars around median that total 46.7% of count
    age_median = age_data.median()
    target_pct = 0.467
    target_count = len(age_data) * target_pct

    # Sort by distance from median and accumulate counts
    cumulative = 0
    highlighted_ages: set = set()

    # Expand outward from median until we reach target
    ages_sorted = sorted(age_counts.index, key=lambda x: abs(x - age_median))
    for age in ages_sorted:
        if cumulative >= target_count:
            break
        highlighted_ages.add(age)
        cumulative += age_counts[age]

    # Color bars based on whether they're in the highlighted set
    colors = []
    for age in age_counts.index:
        if age in highlighted_ages:
            colors.append("#e74c3c")  # Red for highlighted bars
        else:
            colors.append("#95a5a6")  # Gray for others

    ax1.bar(
        age_counts.index,
        age_counts.values,
        color=colors,
        alpha=0.8,
        width=0.8,
        edgecolor="none",
    )

    ax1.set_xlabel("Age", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Age Distribution (Relativley Smoothly Distrubuted)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # RIGHT PLOT: Hours per week - Sharp peaks
    hours_counts = hours_data.value_counts().sort_index()

    # Fill in missing integer values with 0 to ensure one bar per integer
    hours_min = int(hours_data.min())
    hours_max = int(hours_data.max())
    all_hours = pd.Series(0, index=range(hours_min, hours_max + 1))
    all_hours.update(hours_counts)
    hours_counts = all_hours

    # Color bars - only the highest frequency bar is red, rest are gray
    max_count = hours_counts.max()
    colors = []
    for count in hours_counts.values:
        if count == max_count:  # Only the peak
            colors.append("#e74c3c")  # Red for the highest peak
        else:
            colors.append("#95a5a6")  # Gray for all others

    ax2.bar(
        hours_counts.index,
        hours_counts.values,
        color=colors,
        alpha=0.8,
        width=0.8,
        edgecolor="none",
    )

    ax2.set_xlabel("Hours per Week", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Hours per Week Distribution (Highly concentrated)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Add single shared legend for both plots
    legend_elements = [
        Patch(
            facecolor="#e74c3c", alpha=0.8, label="Represents 46.7% around the median"
        )
    ]

    # Position legend centered between the two plots
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
        shadow=True,
        fontsize=15,
        ncol=1,
    )

    # Overall title
    fig.suptitle(
        "Distribution Diversity: Spread vs. Concentrated Data",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.show()

    return None


# GLM feature ranks (1 = most important)
GLM_RANKS: Dict[str, int] = {
    "cat__relationship_Own-child": 1,
    "num__capital_net": 2,
    "cat__relationship_Unmarried": 3,
    "cat__relationship_Not-in-family": 4,
    "cat__relationship_Married": 5,
    "cat__occupation_Farming-fishing": 6,
    "cat__occupation_Other-service": 7,
    "cat__work_class_Self-emp-not-inc": 8,
    "cat__relationship_Other-relative": 9,
    "num__education": 10,
    "cat__occupation_Handlers-cleaners": 11,
    "cat__work_class_State-gov": 12,
    "cat__occupation_None": 13,
    "cat__work_class_None": 14,
    "cat__native_country_Mexico": 15,
    "cat__work_class_Private": 16,
    "cat__work_class_Local-gov": 17,
    "cat__occupation_Exec-managerial": 18,
    "cat__occupation_Machine-op-inspct": 19,
    "cat__native_country_United-States": 20,
    "num__age": 21,
    "cat__native_country_None": 22,
    "cat__occupation_Transport-moving": 23,
    "num__hours_per_week": 24,
    "cat__native_country_Columbia": 25,
    "cat__occupation_Prof-specialty": 26,
    "cat__native_country_South": 27,
    "cat__occupation_Priv-house-serv": 28,
    "cat__occupation_Adm-clerical": 29,
    "cat__work_class_Self-emp-inc": 30,
    "cat__occupation_Tech-support": 31,
    "cat__native_country_Puerto-Rico": 32,
    "cat__native_country_Vietnam": 33,
    "cat__occupation_Craft-repair": 34,
    "cat__native_country_China": 35,
    "cat__native_country_Dominican-Republic": 36,
    "cat__native_country_Poland": 37,
    "cat__native_country_India": 38,
    "cat__native_country_Germany": 39,
    "cat__native_country_El-Salvador": 40,
    "cat__native_country_Peru": 41,
    "cat__work_class_Federal-gov": 42,
    "cat__native_country_Greece": 43,
    "cat__native_country_Nicaragua": 44,
    "cat__native_country_Philippines": 45,
    "cat__native_country_Cuba": 46,
    "cat__native_country_Scotland": 47,
    "cat__native_country_Laos": 48,
    "cat__native_country_Ecuador": 49,
    "cat__occupation_Protective-serv": 50,
    "cat__native_country_Guatemala": 51,
    "cat__work_class_Without-pay": 52,
    "cat__native_country_Trinadad&Tobago": 53,
    "cat__native_country_Jamaica": 54,
    "cat__native_country_Thailand": 55,
    "cat__native_country_Cambodia": 56,
    "cat__native_country_Iran": 57,
    "cat__native_country_Outlying-US(Guam-USVI-etc)": 58,
    "num__is_female": 59,
    "num__is_white": 60,
    "cat__native_country_Haiti": 61,
    "cat__native_country_Italy": 62,
    "cat__native_country_Ireland": 63,
    "cat__native_country_Taiwan": 64,
    "cat__native_country_Hungary": 65,
    "cat__native_country_Portugal": 66,
    "cat__occupation_Sales": 67,
    "cat__native_country_Japan": 68,
    "cat__native_country_Canada": 69,
    "cat__native_country_Hong": 70,
    "cat__native_country_Honduras": 71,
    "num__is_black": 72,
    "cat__native_country_England": 73,
    "cat__native_country_Yugoslavia": 74,
    "cat__work_class_Never-worked": 75,
    "cat__native_country_France": 76,
    "cat__occupation_Armed-Forces": 77,
    "cat__native_country_Holand-Netherlands": 78,
}

# LGBM feature ranks (1 = most important)
LGBM_RANKS: Dict[str, int] = {
    "num__capital_net": 1,
    "num__age": 2,
    "num__hours_per_week": 3,
    "num__education": 4,
    "cat__relationship_Married": 5,
    "num__is_female": 6,
    "cat__work_class_Private": 7,
    "cat__work_class_Self-emp-not-inc": 8,
    "cat__occupation_Exec-managerial": 9,
    "cat__occupation_Prof-specialty": 10,
    "cat__relationship_Unmarried": 11,
    "cat__work_class_Local-gov": 12,
    "cat__occupation_Other-service": 13,
    "cat__occupation_Sales": 14,
    "cat__relationship_Not-in-family": 15,
    "cat__occupation_Farming-fishing": 16,
    "num__is_white": 17,
    "cat__native_country_United-States": 18,
    "cat__work_class_State-gov": 19,
    "cat__occupation_Transport-moving": 20,
    "cat__occupation_Adm-clerical": 21,
    "cat__occupation_Protective-serv": 22,
    "cat__native_country_None": 23,
    "cat__work_class_Self-emp-inc": 24,
    "cat__work_class_Federal-gov": 25,
    "num__is_black": 26,
    "cat__occupation_Handlers-cleaners": 27,
    "cat__occupation_Tech-support": 28,
    "cat__occupation_Craft-repair": 29,
    "cat__work_class_None": 30,
    "cat__native_country_Mexico": 31,
    "cat__occupation_Machine-op-inspct": 32,
    "cat__relationship_Own-child": 33,
    "cat__native_country_Philippines": 34,
    "cat__native_country_Columbia": 35,
    "cat__occupation_None": 36,
    "cat__occupation_Priv-house-serv": 37,
    "cat__native_country_Puerto-Rico": 38,
    "cat__native_country_Vietnam": 39,
    "cat__native_country_Canada": 40,
    "cat__native_country_Ireland": 41,
    "cat__native_country_Italy": 42,
    "cat__native_country_England": 43,
    "cat__native_country_Portugal": 44,
    "cat__native_country_Peru": 45,
    "cat__native_country_Taiwan": 46,
    "cat__native_country_South": 47,
    "cat__native_country_Cambodia": 48,
    "cat__native_country_Dominican-Republic": 49,
    "cat__native_country_Guatemala": 50,
    "cat__native_country_China": 51,
    "cat__native_country_Cuba": 52,
    "cat__native_country_France": 53,
    "cat__native_country_Trinadad&Tobago": 54,
    "cat__native_country_Nicaragua": 55,
    "cat__native_country_India": 56,
    "cat__native_country_Greece": 57,
    "cat__native_country_Scotland": 58,
    "cat__native_country_Outlying-US(Guam-USVI-etc)": 59,
    "cat__native_country_Poland": 60,
    "cat__occupation_Armed-Forces": 61,
    "cat__native_country_Yugoslavia": 62,
    "cat__native_country_Thailand": 63,
    "cat__work_class_Never-worked": 64,
    "cat__work_class_Without-pay": 65,
    "cat__native_country_Laos": 66,
    "cat__native_country_Japan": 67,
    "cat__native_country_Jamaica": 68,
    "cat__native_country_Iran": 69,
    "cat__native_country_Hungary": 70,
    "cat__native_country_Hong": 71,
    "cat__native_country_Honduras": 72,
    "cat__native_country_Haiti": 73,
    "cat__native_country_Germany": 74,
    "cat__native_country_El-Salvador": 75,
    "cat__native_country_Ecuador": 76,
    "cat__relationship_Other-relative": 77,
    "cat__native_country_Holand-Netherlands": 78,
}

# Top 5 features for each model
GLM_TOP5: List[str] = [
    "cat__relationship_Own-child",
    "num__capital_net",
    "cat__relationship_Unmarried",
    "cat__relationship_Not-in-family",
    "cat__relationship_Married",
]

LGBM_TOP5: List[str] = [
    "num__capital_net",
    "num__age",
    "num__hours_per_week",
    "num__education",
    "cat__relationship_Married",
]

MAX_RANK: int = 78


def feature_importance_rank() -> None:
    """
    Plot rank comparison of top 5 features for both GLM and LGBM models.

    Shows two charts side by side:
    - Left: Top 5 GLM features with their ranks in both models
    - Right: Top 5 LGBM features with their ranks in both models

    Higher bars indicate better rank (rank 1 = highest bar).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(5)
    width = 0.35

    # Left chart: Top 5 GLM features
    ax1 = axes[0]

    glm_ranks_for_glm_top5 = [GLM_RANKS[f] for f in GLM_TOP5]
    lgbm_ranks_for_glm_top5 = [LGBM_RANKS[f] for f in GLM_TOP5]

    # Convert ranks to "inverse rank" so higher bar = better rank
    glm_bars = [MAX_RANK - r + 1 for r in glm_ranks_for_glm_top5]
    lgbm_bars = [MAX_RANK - r + 1 for r in lgbm_ranks_for_glm_top5]

    ax1.bar(x - width / 2, glm_bars, width, label="GLM Rank", color="steelblue")
    ax1.bar(x + width / 2, lgbm_bars, width, label="LGBM Rank", color="darkorange")

    ax1.set_xlabel("Feature")
    ax1.set_ylabel("Inverse Rank (higher = more important)")
    ax1.set_title("Top 5 GLM Features: Rank Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f.replace("cat__", "").replace("num__", "") for f in GLM_TOP5],
        rotation=45,
        ha="right",
    )
    ax1.grid(axis="y", alpha=0.3)

    # Add rank annotations on bars
    for i, (g_rank, l_rank) in enumerate(
        zip(glm_ranks_for_glm_top5, lgbm_ranks_for_glm_top5)
    ):
        ax1.annotate(
            f"#{g_rank}", (i - width / 2, glm_bars[i] + 0.5), ha="center", fontsize=9
        )
        ax1.annotate(
            f"#{l_rank}", (i + width / 2, lgbm_bars[i] + 0.5), ha="center", fontsize=9
        )

    # Right chart: Top 5 LGBM features
    ax2 = axes[1]

    glm_ranks_for_lgbm_top5 = [GLM_RANKS[f] for f in LGBM_TOP5]
    lgbm_ranks_for_lgbm_top5 = [LGBM_RANKS[f] for f in LGBM_TOP5]

    glm_bars = [MAX_RANK - r + 1 for r in glm_ranks_for_lgbm_top5]
    lgbm_bars = [MAX_RANK - r + 1 for r in lgbm_ranks_for_lgbm_top5]

    ax2.bar(x - width / 2, glm_bars, width, label="GLM Rank", color="steelblue")
    ax2.bar(x + width / 2, lgbm_bars, width, label="LGBM Rank", color="darkorange")

    ax2.set_xlabel("Feature")
    ax2.set_ylabel("Inverse Rank (higher = more important)")
    ax2.set_title("Top 5 LGBM Features: Rank Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f.replace("cat__", "").replace("num__", "") for f in LGBM_TOP5],
        rotation=45,
        ha="right",
    )
    ax2.grid(axis="y", alpha=0.3)

    # Add rank annotations on bars
    for i, (g_rank, l_rank) in enumerate(
        zip(glm_ranks_for_lgbm_top5, lgbm_ranks_for_lgbm_top5)
    ):
        ax2.annotate(
            f"#{g_rank}", (i - width / 2, glm_bars[i] + 0.5), ha="center", fontsize=9
        )
        ax2.annotate(
            f"#{l_rank}", (i + width / 2, lgbm_bars[i] + 0.5), ha="center", fontsize=9
        )

    plt.suptitle("Feature Importance Rank Comparison: GLM vs LGBM", fontsize=14, y=1.02)

    # Single legend in top left, outside charts
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.98))

    plt.tight_layout()
    plt.show()


def model_comparison() -> None:
    """Plot comparison of GLM, GLM_tuned, LGBM, and LGBM_tuned evaluation metrics."""
    metrics = [
        "Mean Prediction",
        "Bias",
        "MSE",
        "RMSE",
        "MAE",
        "Deviance",
        "Gini",
        "Accuracy",
    ]

    # Hard coded outcomes from running pipeline with and without tuning
    glm_values = [
        0.246279,
        0.053182,
        0.107211,
        0.327432,
        0.216885,
        0.336309,
        0.599456,
        0.845260,
    ]
    glm_tuned_values = [
        0.236305,
        0.010528,
        0.106986,
        0.327088,
        0.217394,
        0.334899,
        0.601942,
        0.845872,
    ]
    lgbm_values = [
        0.238292,
        0.019024,
        0.087443,
        0.295708,
        0.176602,
        0.275612,
        0.653701,
        0.874108,
    ]
    lgbm_tuned_values = [
        0.237590,
        0.016022,
        0.087067,
        0.295071,
        0.174207,
        0.274527,
        0.654307,
        0.872783,
    ]
    mean_outcome = 0.233843

    metric_direction: Dict[str, str] = {
        "Mean Prediction": "closer",
        "Bias": "lower",
        "MSE": "lower",
        "RMSE": "lower",
        "MAE": "lower",
        "Deviance": "lower",
        "Gini": "higher",
        "Accuracy": "higher",
    }

    all_values = [glm_values, glm_tuned_values, lgbm_values, lgbm_tuned_values]

    winners = []
    for i, metric in enumerate(metrics):
        direction = metric_direction[metric]
        metric_values = [v[i] for v in all_values]

        if direction == "lower":
            winner_idx = np.argmin(metric_values)
        elif direction == "higher":
            winner_idx = np.argmax(metric_values)
        else:  # closer to mean_outcome
            diffs = [abs(v - mean_outcome) for v in metric_values]
            winner_idx = np.argmin(diffs)

        winners.append(winner_idx)

    x = np.arange(len(metrics))
    width = 0.18  # Width of each bar

    _, ax = plt.subplots(figsize=(16, 8))

    # Define colors for each model
    colors = [
        "#5B9BD5",
        "#2E75B6",
        "#ED7D31",
        "#C55A11",
    ]  # Light blue, dark blue, light orange, dark orange

    # Plot bars for each model
    bars = []
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for idx, (values, offset) in enumerate(zip(all_values, offsets)):
        bar = ax.bar(
            x + offset * width,
            values,
            width,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.5,
        )
        bars.append(bar)

    # Add horizontal line for mean_outcome spanning the mean_preds bars
    ax.hlines(
        y=mean_outcome,
        xmin=x[0] - 2 * width,
        xmax=x[0] + 2 * width,
        colors="black",
        linestyles="dashed",
        linewidth=2,
    )

    # Add winner stars above the winning bar
    for i, winner_idx in enumerate(winners):
        winner_offset = offsets[winner_idx]
        winner_value = all_values[winner_idx][i]
        ax.annotate(
            "★",
            xy=(x[i] + winner_offset * width, winner_value),
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
        "Model Comparison: GLM vs LGBM (Base and Tuned)",
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
        Patch(facecolor=colors[0], edgecolor="black", label="GLM"),
        Patch(facecolor=colors[1], edgecolor="black", label="GLM_tuned"),
        Patch(facecolor=colors[2], edgecolor="black", label="LGBM"),
        Patch(facecolor=colors[3], edgecolor="black", label="LGBM_tuned"),
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


def categorical_pattern_contrast(df: pl.DataFrame) -> None:
    """
    Contrasting two different TYPES of categorical patterns:
    - Occupation: Gradual hierarchy (skill/education ladder)
    - Marital Status: Binary split (married vs not married)
    """
    target = "high_income"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    fig.suptitle(
        "Two Types of Categorical Patterns: Gradient vs Binary Split",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # --- Left panel: Occupation (GRADIENT pattern) ---
    ax1 = axes[0]
    pdf_occ = df.select(["occupation", target]).to_pandas()
    crosstab_occ = pd.crosstab(
        pdf_occ["occupation"], pdf_occ[target], normalize="index"
    )
    if True in crosstab_occ.columns:
        crosstab_occ = crosstab_occ.sort_values(by=True, ascending=True)

    high_income_occ = (crosstab_occ[True] * 100).round(1)

    # Gradient color scheme - smooth transition showing hierarchy
    n_occ = len(high_income_occ)
    ax1.barh(
        range(n_occ),
        high_income_occ.values,
        color=plt.cm.Blues(np.linspace(0.3, 0.9, n_occ)),
    )

    ax1.set_yticks(range(n_occ))
    ax1.set_yticklabels(crosstab_occ.index, fontsize=9)
    ax1.set_xlabel("% High Income", fontsize=10)
    ax1.set_title(
        "Occupation: A Skill & Education Ladder", fontsize=12, fontweight="bold"
    )

    # Add annotation explaining the pattern
    ax1.text(
        0.95,
        0.12,
        "Each step up the ladder\n≈ 3-5 pp income gain",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        style="italic",
        color="#37474F",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#ECEFF1",
            edgecolor="#90A4AE",
            alpha=0.9,
        ),
    )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.3)

    # --- Right panel: Marital Status (BINARY pattern) ---
    ax2 = axes[1]

    pdf_marital = df.select(["marital_status", target]).to_pandas()
    crosstab_marital = pd.crosstab(
        pdf_marital["marital_status"], pdf_marital[target], normalize="index"
    )
    if True in crosstab_marital.columns:
        crosstab_marital = crosstab_marital.sort_values(by=True, ascending=True)

    high_income_marital = (crosstab_marital[True] * 100).round(1)

    # Binary color scheme - highlight the split
    married_keywords = ["Married-civ-spouse", "Married-AF-spouse"]
    colors_marital = []
    for status in crosstab_marital.index:
        if status in married_keywords:
            colors_marital.append("#1565C0")  # Blue for married w/ spouse present
        else:
            colors_marital.append("#BBDEFB")  # Light blue for others

    ax2.barh(
        range(len(crosstab_marital)), high_income_marital.values, color=colors_marital
    )

    ax2.set_yticks(range(len(crosstab_marital)))
    ax2.set_yticklabels(crosstab_marital.index, fontsize=9)
    ax2.set_xlabel("% High Income", fontsize=10)
    ax2.set_title(
        "Marital Status: Spouse Present is What Matters", fontsize=12, fontweight="bold"
    )

    # Add dividing line and annotations
    # Find the boundary between married (spouse present) and others
    married_indices = [
        i for i, s in enumerate(crosstab_marital.index) if s in married_keywords
    ]
    if married_indices:
        boundary = min(married_indices) - 0.5
        ax2.axhline(y=boundary, color="#1565C0", linestyle="-", linewidth=2, alpha=0.7)

        # Calculate averages for each group
        married_avg = high_income_marital[
            crosstab_marital.index.isin(married_keywords)
        ].mean()
        other_avg = high_income_marital[
            ~crosstab_marital.index.isin(married_keywords)
        ].mean()

        ax2.text(
            0.95,
            0.88,
            f"Spouse present: {married_avg:.0f}% avg",
            transform=ax2.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#1565C0",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#1565C0"
            ),
        )

        ax2.text(
            0.95,
            0.08,
            f"Others: {other_avg:.0f}% avg",
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#64B5F6",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#64B5F6"
            ),
        )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", linestyle="--", alpha=0.3)

    # Match x-axis scales
    max_x = max(high_income_occ.max(), high_income_marital.max()) + 5
    ax1.set_xlim(0, max_x)
    ax2.set_xlim(0, max_x)

    # Add bottom annotation explaining the contrast
    fig.text(
        0.5,
        -0.02,
        "Occupation predicts income via a continuous hierarchy • Marital "
        "status shows a sharp divide based on spouse presence",
        ha="center",
        fontsize=10,
        style="italic",
        color="#616161",
    )

    plt.tight_layout()
    plt.show()


def visualize_data_quality_issues(df: pl.DataFrame) -> None:
    """
    Data quality visualization redesigned to tell a clear story:
    1. Left panel: Overall health summary (the good news)
    2. Right panel: Focus on columns needing attention (the work to do)

    Key insight: Target variable has the most critical issue.
    """
    target_column = "income"
    expected_target_values = ["<=50K", ">50K"]

    cols = df.columns
    total_rows = len(df)

    # --- 1. Calculate all issue types ---
    null_counts_row = df.null_count().row(0)
    null_counts = {col: count for col, count in zip(cols, null_counts_row)}

    question_counts = {}
    whitespace_counts = {}
    for col, dtype in zip(cols, df.dtypes):
        if dtype in (pl.String, pl.Categorical, pl.Object):
            question_counts[col] = df.select((pl.col(col) == "?").sum()).item()
            ws_count = df.select(
                (pl.col(col) != pl.col(col).str.strip_chars()).sum()
            ).item()
            whitespace_counts[col] = ws_count if ws_count else 0
        else:
            question_counts[col] = 0
            whitespace_counts[col] = 0

    format_counts = {col: 0 for col in cols}
    if target_column in cols:
        target_values = df[target_column].to_list()
        format_counts[target_column] = sum(
            1
            for v in target_values
            if v is not None
            and isinstance(v, str)
            and v not in expected_target_values
            and v.rstrip(".") in expected_target_values
        )

    # --- 2. Build issue summary ---
    issue_data = []
    for col in cols:
        null_val = null_counts.get(col, 0)
        q_val = question_counts.get(col, 0)
        fmt_val = format_counts.get(col, 0)
        ws_val = whitespace_counts.get(col, 0)
        total_issues = null_val + q_val + fmt_val + ws_val

        if total_issues > 0:
            issue_data.append(
                {
                    "column": col,
                    "null": null_val,
                    "Placeholder '?'": q_val,
                    "Trailing '.'": fmt_val,
                    "Whitespace": ws_val,
                    "total_issues": total_issues,
                    "pct": total_issues / total_rows * 100,
                    "is_target": col == target_column,
                }
            )

    # Sort: target LAST (at bottom, most prominent), then by severity descending
    issue_data = sorted(issue_data, key=lambda x: (x["is_target"], x["pct"]))

    # Overall stats
    total_issues = sum(d["total_issues"] for d in issue_data)
    total_cells = total_rows * len(cols)
    clean_pct = (1 - total_issues / total_cells) * 100
    cols_with_issues = len(issue_data)
    cols_clean = len(cols) - cols_with_issues

    # --- 3. Create figure with two panels ---
    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.3)

    # === LEFT PANEL: Overall Health ===
    ax_left = fig.add_subplot(gs[0])

    # Donut chart
    sizes = [clean_pct, 100 - clean_pct]
    colors_donut = ["#27ae60", "#e8e8e8"]
    wedges, _ = ax_left.pie(
        sizes,
        colors=colors_donut,
        startangle=90,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2),
    )

    # Center text
    ax_left.text(
        0,
        0.12,
        f"{clean_pct:.0f}%",
        ha="center",
        va="center",
        fontsize=32,
        fontweight="bold",
        color="#27ae60",
    )
    ax_left.text(
        0,
        -0.22,
        "Clean\nValues",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#555555",
    )

    # Info box below donut - lowered
    ax_left.text(
        0,
        -1.15,
        f"{cols_clean} of {len(cols)} columns have no issues",
        ha="center",
        va="center",
        fontsize=9,
        fontstyle="italic",
        color="#555555",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#f8f9fa",
            edgecolor="#dee2e6",
            linewidth=1,
        ),
    )

    ax_left.set_xlim(-1.5, 1.5)
    ax_left.set_ylim(-1.5, 1.2)
    ax_left.axis("off")
    ax_left.set_title("Overall Data Health", fontsize=13, fontweight="bold", pad=10)

    # === RIGHT PANEL: Issues to Address ===
    ax_right = fig.add_subplot(gs[1])

    columns = [d["column"] for d in issue_data]
    y_pos = np.arange(len(columns))
    bar_height = 0.6

    # Distinct color palette - spread across spectrum
    colors_issues = {
        "null": "#db0284",  # Yellow
        "Placeholder '?'": "#8e44ad",  # Purple
        "Trailing '.'": "#d40000",  # Orange
        "Whitespace": "#2980b9",  # Blue
    }

    # Plot stacked bars
    left = [0.0] * len(issue_data)  # Initialize as float

    for issue_type in ["null", "Placeholder '?'", "Trailing '.'", "Whitespace"]:
        values = [d[issue_type] / total_rows * 100 for d in issue_data]
        ax_right.barh(
            y_pos,
            values,
            height=bar_height,
            left=left,
            color=colors_issues[issue_type],
            edgecolor="white",
            linewidth=0.5,
            label=issue_type,
        )
        # Fix for E741: changed 'l' to 'left_val'
        left = [left_val + v for left_val, v in zip(left, values)]

    # Find target index
    target_idx = next(i for i, d in enumerate(issue_data) if d["is_target"])

    # Add percentage labels
    for i, d in enumerate(issue_data):
        pct = d["pct"]
        if pct >= 1:
            ax_right.text(
                pct + 0.8,
                i,
                f"{int(round(pct))}%",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#333333",
            )

    # TARGET label
    target_pct = issue_data[target_idx]["pct"]
    ax_right.annotate(
        "TARGET",
        xy=(target_pct + 5, target_idx),
        xytext=(target_pct + 12, target_idx),
        fontsize=10,
        fontweight="bold",
        color="#856404",
        va="center",
        arrowprops=dict(arrowstyle="-", color="#856404", linewidth=1.5),
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#fff3cd",
            edgecolor="#856404",
            linewidth=1,
        ),
    )

    # Explanatory text directly under TARGET label
    ax_right.text(
        target_pct + 12,
        target_idx - 0.55,
        "Target has 4 labels instead of 2\n"
        "(e.g. '>50K' and '>50K.' treated as different)\n"
        "Must fix before modelling",
        ha="center",
        va="top",
        fontsize=9,
        fontstyle="italic",
        color="#555555",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#f8f9fa",
            edgecolor="#dee2e6",
            linewidth=1,
        ),
    )

    # Y-axis labels
    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(columns, fontsize=11)

    # Make target label bold
    for i, label in enumerate(ax_right.get_yticklabels()):
        if issue_data[i]["is_target"]:
            label.set_fontweight("bold")

    ax_right.set_xlim(0, max(d["pct"] for d in issue_data) + 28)
    ax_right.xaxis.set_visible(False)

    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)
    ax_right.spines["left"].set_color("#333333")

    ax_right.set_title(
        "Four Columns with Issues", fontsize=13, fontweight="bold", pad=10
    )

    # Vertical legend at bottom left of bar chart
    legend_items = [
        ("Trailing '.'", colors_issues["Trailing '.'"]),
        ("null", colors_issues["null"]),
        ("Placeholder '?'", colors_issues["Placeholder '?'"]),
        ("Whitespace", colors_issues["Whitespace"]),
    ]

    legend_x = 0.5
    legend_y_start = 0.38
    line_spacing = 0.09

    for idx, (label, color) in enumerate(legend_items):
        y_pos_legend = legend_y_start - idx * line_spacing
        ax_right.text(
            legend_x,
            y_pos_legend,
            label,
            transform=ax_right.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    # Main title
    fig.suptitle(
        "Data Quality Assessment: Mostly Clean, Four Fixes Required",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.95)
    plt.show()
