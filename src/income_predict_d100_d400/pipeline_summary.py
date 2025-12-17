from pathlib import Path
from typing import Dict, Tuple

from income_predict_d100_d400.robust_paths import DATA_DIR, PLOTS_DIR

EXPECTED_ARTIFACTS: Dict[str, Path] = {
    "GLM Model": DATA_DIR / "glm_model.joblib",
    "LGBM Model": DATA_DIR / "lgbm_model.joblib",
    "Classification Matrix": PLOTS_DIR / "classification_plot.png",
    "Calibration Plot": PLOTS_DIR / "calibration_plot.png",
    "ROC Curve Plot": PLOTS_DIR / "roc_curve_plot.png",
}


def get_relative_path(path: Path) -> str:
    """Helper to try and get a relative path for display, falling back to absolute."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def check_files() -> Dict[str, Tuple[bool, str]]:
    """
    Checks if expected artifacts exist.

    Returns:
        A dictionary mapping description to (exists_bool, path_string).
    """
    return {
        desc: (path.exists(), get_relative_path(path))
        for desc, path in EXPECTED_ARTIFACTS.items()
    }


def print_pipeline_summary() -> None:
    """
    Orchestrates the checking of artifacts and prints a formatted summary table.
    Includes a dynamic check for Partial Dependence plots.
    """
    results = check_files()
    pdp_files = list(PLOTS_DIR.glob("partial_dependence_*.png"))
    pdp_exists = len(pdp_files) >= 1

    pdp_display_path = get_relative_path(PLOTS_DIR / "partial_dependence_*.png")
    results["Partial Dependence Plots"] = (pdp_exists, pdp_display_path)

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print("Models and performance charts created:")
    print(f"{'Status':<7} | {'Description':<25} | {'Location'}")
    print("-" * 80)

    for desc, (exists, path) in results.items():
        status = "✅" if exists else "❌"
        print(f"{status:<6} | {desc:<25} | {path}")

    print("-" * 80)
    print(
        "See generated charts for more detailed interpretations of model performance."
    )
    print("=" * 80)
