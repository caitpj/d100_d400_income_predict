from pathlib import Path
from typing import Dict, Tuple

from income_predict_d100_d400.robust_paths import DATA_DIR, PLOTS_DIR

EXPECTED_ARTIFACTS: Dict[str, Path] = {
    "Raw Data": DATA_DIR / "census_income.parquet",
    "Cleaned Data": DATA_DIR / "cleaned_census_income.parquet",
    "Train Split": DATA_DIR / "train_split.parquet",
    "Test Split": DATA_DIR / "test_split.parquet",
    "GLM Model": DATA_DIR / "glm_model.joblib",
    "LGBM Model": DATA_DIR / "lgbm_model.joblib",
    "Train Features": DATA_DIR / "train_features.parquet",
    "Confusion Matrix Plot": PLOTS_DIR / "classification_plot.png",
    "Calibration Plot": PLOTS_DIR / "calibration_plot.png",
    "ROC Curve Plot": PLOTS_DIR / "roc_curve_plot.png",
}


def check_files(artifacts: Dict[str, Path]) -> Dict[str, Tuple[bool, str]]:
    """
    Checks if files exist and resolves their relative paths.

    Parameters:
        artifacts: Dictionary mapping descriptions to file paths.

    Returns:
        A dictionary mapping description to (exists_bool, display_path_string).
    """
    results: Dict[str, Tuple[bool, str]] = {}

    for desc, path in artifacts.items():
        exists = path.exists()

        try:
            display_path = str(path.relative_to(Path.cwd()))
        except ValueError:
            display_path = str(path)

        results[desc] = (exists, display_path)

    return results


def print_pipeline_summary() -> None:
    """
    Orchestrates the checking of artifacts and prints a formatted summary table.
    Includes a dynamic check for Partial Dependence plots (expects 5 files).
    """
    results = check_files(EXPECTED_ARTIFACTS)
    pdp_files = list(PLOTS_DIR.glob("partial_dependence_*.png"))
    pdp_count = len(pdp_files)
    pdp_exists = pdp_count >= 5

    try:
        pdp_display_path = str(
            PLOTS_DIR.relative_to(Path.cwd()) / "partial_dependence_*.png"
        )
    except ValueError:
        pdp_display_path = str(PLOTS_DIR / "partial_dependence_*.png")

    results["Partial Dependence Plots"] = (pdp_exists, pdp_display_path)

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Status':<7} | {'Description':<25} | {'Location'}")
    print("-" * 80)

    for desc, (exists, path) in results.items():
        status = "✅" if exists else "❌"
        print(f"{status:<6} | {desc:<25} | {path}")

    print("-" * 80)
