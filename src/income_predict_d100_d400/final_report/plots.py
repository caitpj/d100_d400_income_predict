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


def binary_column_issue(
    df: pl.DataFrame,
    column_name: str,
    expected_values: Optional[List[str]] = None,
) -> Optional[Tuple[Figure, Axes]]:
    """
    Visualizes data quality issues in a binary column by showing counts of each unique value.
    Values that don't match the expected binary values are highlighted in red.

    Parameters:
    -----------
    df : polars.DataFrame
        The DataFrame containing the column to analyze
    column_name : str
        The name of the column to analyze
    expected_values : list, optional
        List of expected valid values (default: ['<=50K', '>50K'])

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    if expected_values is None:
        expected_values = ["<=50K", ">50K"]

    # Get value counts for the column
    value_counts_df = df.group_by(column_name).len().sort("len", descending=True)
    unique_values = value_counts_df[column_name].to_list()
    counts = value_counts_df["len"].to_list()

    # Determine which values are correct vs incorrect
    colors = []
    labels = []
    for value in unique_values:
        if value in expected_values:
            colors.append("#2ecc71")  # Green for correct values
            labels.append("Correctly labeled")
        else:
            colors.append("#e74c3c")  # Red for incorrect values
            labels.append("Not correctly labeled")

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(unique_values))
    bars = ax.bar(x_pos, counts, color=colors, alpha=0.8)

    # Customize the plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(unique_values, rotation=45, ha="right")
    ax.set_xlabel("Unique Values", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title(
        "Data Quality Issues: Target Variable Is Not Currently Binary",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on top of bars
    max_count = max(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_count * 0.01,
            str(int(count)),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Create custom legend
    legend_elements = [
        Patch(facecolor="#2ecc71", alpha=0.8, label="Correctly labeled"),
        Patch(facecolor="#e74c3c", alpha=0.8, label="Not correctly labeled"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, shadow=True)

    plt.tight_layout()
    plt.show()

    return None


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

    # Get the data for both columns, removing NaN values
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


def occupation_correlation(df: pl.DataFrame) -> None:
    """
    Plots a 100% stacked bar chart for occupation vs high_income.
    Bars are ordered by the proportion with high_income=True (descending).
    """
    target = "high_income"
    feature = "occupation"

    # Convert to pandas for easy cross-tabulation
    # (recreating pd.crosstab logic manually in Polars for plotting is verbose)
    pdf = df.select([feature, target]).to_pandas()

    # Create crosstab with proportions
    crosstab = pd.crosstab(pdf[feature], pdf[target], normalize="index")

    # Sort by the proportion with high_income=True (descending)
    if True in crosstab.columns:
        crosstab = crosstab.sort_values(by=True, ascending=False)

    # Create figure and plot on the same axes
    fig, ax = plt.subplots(figsize=(12, 6))
    crosstab.plot(kind="bar", stacked=True, ax=ax)

    ax.set_title(f"{feature} Distribution by {target}", fontsize=12)
    ax.set_ylabel("Proportion")
    ax.set_xlabel(feature)
    ax.legend(title=target, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_missing_data(df: pl.DataFrame) -> None:
    """
    Visualizes data quality issues in a DataFrame by showing counts of NaN and '?' values
    for each column in a horizontal stacked bar chart.

    Parameters:
    -----------
    df : polars.DataFrame
        The DataFrame to analyze for data quality issues

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Calculate counts using Polars
    # Null counts
    null_counts_row = df.null_count().row(0)
    cols = df.columns
    nan_counts = pd.Series(null_counts_row, index=cols)

    # Question mark counts (assuming string columns)
    q_counts = []
    for col, dtype in zip(cols, df.dtypes):
        if dtype in (pl.String, pl.Categorical, pl.Object):
            # Safe check for string equality
            count = df.select((pl.col(col) == "?").sum()).item()
            q_counts.append(count)
        else:
            q_counts.append(0)

    question_counts = pd.Series(q_counts, index=cols)

    # Sort by total issues (ascending for horizontal bar chart)
    total_issues = nan_counts + question_counts
    sorted_indices = total_issues.sort_values(ascending=True).index
    nan_counts = nan_counts[sorted_indices]
    question_counts = question_counts[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, max(6, len(nan_counts) * 0.4)))

    y_pos = np.arange(len(nan_counts))

    ax.barh(y_pos, nan_counts, label="NaN", color="#e74c3c", alpha=0.8)
    ax.barh(
        y_pos, question_counts, left=nan_counts, label="'?'", color="#f39c12", alpha=0.8
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(nan_counts.index)
    ax.set_xlabel("Count of Issues", fontsize=12, fontweight="bold")
    ax.set_ylabel("Column Name", fontsize=12, fontweight="bold")
    ax.set_title(
        "Data Quality Issues: NaN and '?' Values by Column",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="lower right", frameon=True, shadow=True)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    for i, (nan_val, q_val) in enumerate(zip(nan_counts, question_counts)):
        if nan_val > 0:
            ax.text(
                nan_val / 2,
                i,
                str(int(nan_val)),
                ha="center",
                va="center",
                fontweight="bold",
                color="white",
            )
        if q_val > 0:
            ax.text(
                nan_val + q_val / 2,
                i,
                str(int(q_val)),
                ha="center",
                va="center",
                fontweight="bold",
                color="white",
            )

    plt.tight_layout()

    plt.show()
