import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix


def plot_partial_dependence(model, X, top_features):
    """Plot partial dependence for top features."""
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(model, X, features=top_features, ax=ax)
    plt.suptitle("Partial Dependence Plots - Top Features", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(y_true, glm_preds, lgbm_preds):
    """Plots confusion matrix heatmaps for GLM and LGBM side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
    plt.show()


def plot_numeric_distributions(df: pd.DataFrame):
    """Plots histograms for numeric columns and bar charts for boolean/string columns."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    non_binary_numeric = [col for col in numeric_cols if df[col].nunique() > 2]

    if non_binary_numeric:
        for col in non_binary_numeric:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

    categorical_cols = df.select_dtypes(include=["bool", "object", "category"]).columns

    if len(categorical_cols) > 0:
        for col in categorical_cols:
            plt.figure(figsize=(8, 4))
            value_counts = df[col].value_counts(dropna=False)
            sns.barplot(x=value_counts.index.astype(str), y=value_counts.values)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()


def plot_numeric_boxplots(df: pd.DataFrame) -> None:
    """
    Visualizes numeric columns using boxplots to identify outliers visually.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("No numeric columns found to visualize.")
        return

    n_cols = len(numeric_cols)

    plt.figure(figsize=(15, 5))

    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(1, n_cols, i)
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot of {col}")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_target_distribution(df: pd.DataFrame, target: str = "income") -> None:
    """
    Plots the distribution of the target variable from the summary dataframe.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df.index.astype(str), y=df["Count"])
    plt.title(f"Distribution of {target}", fontsize=16)
    plt.xlabel(target, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_feature_correlations(
    correlations: pd.Series, target: str = "income", title_suffix: str = ""
) -> None:
    """
    Plots a horizontal bar chart of feature correlations with the target variable.
    """
    correlations_sorted = correlations.sort_values()

    plt.figure(figsize=(10, max(6, len(correlations_sorted) * 0.3)))
    plt.barh(range(len(correlations_sorted)), correlations_sorted.values, alpha=0.7)

    for i, value in enumerate(correlations_sorted.values):
        plt.text(
            value,
            i,
            f" {value:.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )

    plt.yticks(range(len(correlations_sorted)), correlations_sorted.index)
    plt.xlabel("Correlation Coefficient", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.title(f"Feature Correlations with {target} {title_suffix}", fontsize=16)
    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_numeric_strip(df: pd.DataFrame, target: str) -> None:
    """
    Plots strip plots for all numeric features in the DataFrame against the target.
    """
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_features:
        numeric_features.remove(target)

    if numeric_features:
        cols = 2
        rows = (len(numeric_features) + 1) // 2

        plt.figure(figsize=(12, 4 * rows))

        for i, feature in enumerate(numeric_features):
            plt.subplot(rows, cols, i + 1)
            sns.stripplot(
                data=df, x=target, y=feature, jitter=0.45, alpha=0.05, legend=False
            )
            plt.title(f"{feature} vs {target}")
            plt.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.show()


def plot_categorical_stack(df: pd.DataFrame, target: str) -> None:
    """
    Plots 100% stacked bar charts for categorical features using default package colors.
    """
    cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target in cat_features:
        cat_features.remove(target)

    if cat_features:
        cols = 2
        rows = (len(cat_features) + 1) // 2
        plt.figure(figsize=(14, 5 * rows))

        for i, feature in enumerate(cat_features):
            plt.subplot(rows, cols, i + 1)

            crosstab = pd.crosstab(df[feature], df[target], normalize="index")
            crosstab.plot(kind="bar", stacked=True, ax=plt.gca())

            plt.title(f"{feature} Distribution by {target}", fontsize=12)
            plt.ylabel("Proportion")
            plt.legend(title=target, bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.show()
