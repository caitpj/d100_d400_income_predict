import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    """Return the data directory, creating it if needed."""
    env_data_dir = os.environ.get("DATA_DIR")
    if env_data_dir:
        data_dir = Path(env_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    dev_candidates = [
        PROJECT_DIR / "src" / "data",
        PROJECT_DIR / "data",
    ]

    for candidate in dev_candidates:
        if candidate.exists():
            return candidate

    # PyPI / Fallback method
    cwd_data = Path.cwd() / "data"
    cwd_data.mkdir(parents=True, exist_ok=True)
    return cwd_data


DATA_DIR = get_data_dir()
PLOTS_DIR = DATA_DIR / "plots"
