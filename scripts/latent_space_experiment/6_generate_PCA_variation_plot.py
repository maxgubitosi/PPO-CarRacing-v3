from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from latent.paths import PLOTS_AND_METRICS_DIR, ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep each PCA component and plot how the decoder reconstruction changes "
            "for different coefficients."
        )
    )
    default_model = (
        ROOT_DIR
        / "scripts"
        / "latent_space_experiment"
        / "models"
        / "pca"
        / "dim_005"
        / "pca_model.pkl"
    )
    parser.add_argument(
        "--pca-model",
        type=Path,
        default=default_model,
        help=(
            "Path to a fitted PCA model pickle (defaults to dim_005/pca_model.pkl "
            "in scripts/latent_space_experiment/models/pca)."
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional path to the metadata.json companion file (defaults to sibling of the model).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to save the resulting grid. "
            "Defaults to scripts/latent_space_experiment/plots_and_metrics/ using the model name."
        ),
    )
    parser.add_argument(
        "--std-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-3.0, 3.0),
        help="Standard deviation range (in latent units) to sweep for every component.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=7,
        help="Number of intermediate coefficients to sample inside the std range (defaults to 7).",
    )
    parser.add_argument(
        "--components",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Subset of component indices (1-based) to plot. "
            "If omitted, every available component is shown."
        ),
    )
    return parser.parse_args()


def _load_pca_model(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_metadata(metadata_path: Path | None) -> dict | None:
    if metadata_path is None or not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _infer_latent_dim(pca_model) -> int:
    for attr in ("n_components_", "n_components"):
        latent_dim = getattr(pca_model, attr, None)
        if latent_dim is not None:
            return int(latent_dim)
    return int(pca_model.components_.shape[0])


def _resolve_output_path(model_path: Path, specified_output: Path | None) -> Path:
    if specified_output is not None:
        return specified_output.expanduser().resolve()
    model_dir_name = model_path.parent.name or "pca_model"
    default_path = PLOTS_AND_METRICS_DIR / f"pca_variations_{model_dir_name}.png"
    return default_path


def _prepare_axes(
    figures: np.ndarray | plt.Axes,
    rows: int,
    cols: int,
) -> np.ndarray:
    axes = np.asarray(figures)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        if rows == 1:
            axes = axes.reshape(1, cols)
        else:
            axes = axes.reshape(rows, 1)
    return axes


def _reshape_reconstruction(flat: np.ndarray, height: int, width: int, channels: int) -> np.ndarray:
    image = flat.reshape(height, width, channels)
    if channels == 1:
        return image[..., 0]
    return image


def _build_latent(latent_dim: int, component_idx: int, coefficient: float) -> np.ndarray:
    latent = np.zeros(latent_dim, dtype=np.float32)
    latent[component_idx] = coefficient
    return latent


def generate_variation_grid(
    model_path: Path,
    metadata_path: Path | None,
    output_path: Path,
    std_range: Sequence[float],
    num_steps: int,
    component_selection: Iterable[int] | None,
) -> Path:
    model_path = model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"PCA model not found: {model_path}")

    resolved_metadata = metadata_path.expanduser().resolve() if metadata_path else None
    metadata = _load_metadata(resolved_metadata)
    if metadata is None:
        default_metadata = model_path.parent / "metadata.json"
        metadata = _load_metadata(default_metadata)

    if metadata is None:
        raise FileNotFoundError(
            "Metadata file is required to recover image dimensions. "
            "Provide --metadata explicitly if it is stored elsewhere."
        )

    target_height, target_width = metadata.get("target_size", [0, 0])
    channel_count = int(metadata.get("channel_count", 3))
    if int(target_height) <= 0 or int(target_width) <= 0:
        raise ValueError("Metadata target_size must contain positive height and width values.")

    pca_model = _load_pca_model(model_path)
    latent_dim = _infer_latent_dim(pca_model)

    variance = getattr(pca_model, "explained_variance_", None)
    variance_ratio = getattr(pca_model, "explained_variance_ratio_", None)
    if variance is None:
        raise AttributeError("The loaded PCA model is missing explained_variance_; unable to scale coefficients.")

    component_indices: list[int]
    if component_selection:
        component_indices = []
        for comp in component_selection:
            if comp < 1 or comp > latent_dim:
                raise ValueError(f"Component index {comp} is outside valid range [1, {latent_dim}].")
            component_indices.append(int(comp - 1))
    else:
        component_indices = list(range(latent_dim))

    if not component_indices:
        raise ValueError("No components selected for plotting.")

    num_components = len(component_indices)
    if num_steps <= 0:
        raise ValueError("--num-steps must be > 0.")

    std_min, std_max = sorted(std_range[:2])
    coefficients = np.linspace(std_min, std_max, num_steps, dtype=np.float32)

    fig, axes_raw = plt.subplots(
        num_components,
        num_steps,
        figsize=(max(3, num_steps) * 1.6,
                 max(2, num_components) * 1.6),
    )
    axes = _prepare_axes(axes_raw, num_components, num_steps)

    cmap = "gray" if channel_count == 1 else None
    feature_dim = int(target_height) * int(target_width) * channel_count

    if pca_model.components_.shape[1] != feature_dim:
        raise ValueError(
            "Metadata target_size/channel_count do not match PCA feature dimension: "
            f"{target_height}x{target_width}x{channel_count} != {pca_model.components_.shape[1]}"
        )

    for row, comp_idx in enumerate(component_indices):
        comp_axes = axes[row]
        std_value = math.sqrt(max(variance[comp_idx], 0.0))
        variance_pct = None
        if variance_ratio is not None and len(variance_ratio) > comp_idx:
            variance_pct = 100.0 * float(variance_ratio[comp_idx])

        for col, std_multiplier in enumerate(coefficients):
            coeff = float(std_multiplier * std_value) if std_value > 0 else 0.0
            latent = _build_latent(latent_dim, comp_idx, coeff)
            reconstruction = pca_model.inverse_transform(latent.reshape(1, -1))
            image = reconstruction.reshape(feature_dim)
            image = np.clip(image, 0.0, 1.0)
            reshaped = _reshape_reconstruction(image, int(target_height), int(target_width), channel_count)

            ax = comp_axes[col]
            ax.imshow(reshaped, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{std_multiplier:+.2f}σ", fontsize=9)

        label = f"PC {comp_idx + 1}"
        if variance_pct is not None:
            label += f"\n({variance_pct:.1f}% var)"
        comp_axes[0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)

    title_name = model_path.parent.name or model_path.stem
    fig.suptitle(f"PCA component sweep - {title_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = output_path.expanduser().resolve()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    model_path = args.pca_model
    metadata_path = args.metadata
    output_path = _resolve_output_path(model_path, args.output)

    saved_path = generate_variation_grid(
        model_path=model_path,
        metadata_path=metadata_path,
        output_path=output_path,
        std_range=args.std_range,
        num_steps=args.num_steps,
        component_selection=args.components,
    )
    print(f"Saved PCA variation grid to {saved_path}")


if __name__ == "__main__":
    main()
