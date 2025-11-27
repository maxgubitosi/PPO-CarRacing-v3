from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT_DIR / "scripts" / "latent_space_experiment"
IMAGE_COLLECTION_DIR = EXPERIMENT_ROOT / "image_collection"
MODEL_DIR = EXPERIMENT_ROOT / "models"
PLOTS_AND_METRICS_DIR = EXPERIMENT_ROOT / "plots_and_metrics"
GREYSCALE_PRESETS_PATH = ROOT_DIR / "scripts" / "convert_to_greyscale" / "greyscale_presets.jsonl"


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resize_variant_suffix(resize_level: int | None) -> str | None:
    """Return the directory suffix for a given resize level (>=2)."""
    if resize_level is None or resize_level <= 1:
        return None
    return f"resize_lvl{int(resize_level)}"


def variant_subdir(base: Path, resize_level: int | None) -> Path:
    """Append the resize variant suffix to a directory if needed."""
    suffix = resize_variant_suffix(resize_level)
    return base / suffix if suffix else base
