import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_numeric_distributions(df: pd.DataFrame, columns: list[str]):
    """Plots histograms for specified numeric columns."""
    for col in columns:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()
        else:
            print(f"Column {col} not found in dataframe.")
