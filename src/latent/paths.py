from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT_DIR / "scripts" / "latent_space_experiment"
IMAGE_COLLECTION_DIR = EXPERIMENT_ROOT / "image_collection"
MODEL_DIR = EXPERIMENT_ROOT / "models"
PLOTS_AND_METRICS_DIR = EXPERIMENT_ROOT / "plots_and_metrics"


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path
