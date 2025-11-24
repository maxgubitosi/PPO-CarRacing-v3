from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataclasses import dataclass
from typing import Dict, Tuple

from latent.data import (  # noqa: E402
    DEFAULT_CROP_RATIO,
    DEFAULT_TARGET_SIZE,
    load_image_batch,
    collect_image_paths,
    shuffle_and_limit,
)
from latent.paths import (  # noqa: E402
    IMAGE_COLLECTION_DIR,
    MODEL_DIR,
    PLOTS_AND_METRICS_DIR,
    GREYSCALE_PRESETS_PATH,
    ensure_dir,
)
from latent.vae import BetaVAE  # noqa: E402
from latent.greyscale import load_greyscale_preset, GreyscalePreset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visualizations and metrics for trained latent models."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=IMAGE_COLLECTION_DIR,
        help="Directory containing collected frames.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory where trained models are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_AND_METRICS_DIR,
        help="Directory to store generated plots and metrics.",
    )
    parser.add_argument(
        "--latent-dims",
        type=int,
        nargs="+",
        default=[3, 5, 8, 12, 24],
        help="Latent dimensionalities to analyse.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Number of images to use for reconstructions and metrics.",
    )
    parser.add_argument(
        "--grid-count",
        type=int,
        default=8,
        help="Number of examples to show in reconstruction grids.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for beta-VAE inference (default: auto-detect).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["pca", "beta-vae"],
        help="Subset of models to analyse (choices: pca, beta-vae).",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=DEFAULT_CROP_RATIO,
        help="Fraction of image height to trim from the bottom before analysis.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=DEFAULT_TARGET_SIZE[0],
        help="Processed image height (set <=0 to keep original).",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=DEFAULT_TARGET_SIZE[1],
        help="Processed image width (set <=0 to keep original).",
    )
    parser.add_argument(
        "--greyscale-presets-path",
        type=Path,
        default=GREYSCALE_PRESETS_PATH,
        help="JSONL file containing greyscale presets.",
    )
    parser.add_argument(
        "--greyscale-label",
        type=str,
        default="veryheavy-medium",
        help="Preset label to apply. Provide empty string to disable greyscale preprocessing.",
    )
    return parser.parse_args()


def compute_mse(original: np.ndarray, reconstruction: np.ndarray) -> float:
    return float(np.mean((np.clip(reconstruction, 0.0, 1.0) - np.clip(original, 0.0, 1.0)) ** 2))


def compute_psnr(original: np.ndarray, reconstruction: np.ndarray) -> float:
    mse = compute_mse(original, reconstruction)
    if mse == 0.0:
        return float("inf")
    return float(20 * math.log10(1.0 / math.sqrt(mse)))


def _prepare_for_display(image: np.ndarray) -> tuple[np.ndarray, str | None]:
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0], "gray"
    if image.ndim == 2:
        return image, "gray"
    return image, None


def save_reconstruction_grid(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    output_path: Path,
    title: str,
    grid_count: int,
) -> None:
    num_samples = min(grid_count, originals.shape[0])
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    fig.suptitle(title)

    for idx in range(num_samples):
        orig, orig_cmap = _prepare_for_display(originals[idx])
        recon, recon_cmap = _prepare_for_display(reconstructions[idx])
        axes[0, idx].imshow(np.clip(orig, 0.0, 1.0), cmap=orig_cmap)
        axes[0, idx].axis("off")
        axes[0, idx].set_title(f"Orig {idx+1}")

        axes[1, idx].imshow(np.clip(recon, 0.0, 1.0), cmap=recon_cmap)
        axes[1, idx].axis("off")
        axes[1, idx].set_title(f"Recon {idx+1}")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_latent_scatter(
    latents: np.ndarray,
    output_path: Path,
    title: str,
    alpha: float = 0.6,
) -> None:
    if latents.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(latents[:, 0], latents[:, 1], s=12, alpha=alpha)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_pca_model(model_path: Path):
    import pickle

    with model_path.open("rb") as f:
        return pickle.load(f)


def analyse_pca(
    latent_dim: int,
    sample_images: np.ndarray,
    output_root: Path,
    model_dir: Path,
    grid_count: int,
) -> Dict[str, float]:
    model_path = model_dir / "pca" / f"dim_{latent_dim:03d}" / "pca_model.pkl"
    if not model_path.exists():
        print(f"[PCA] Missing model for z={latent_dim}, skipping.")
        return {}

    pca = load_pca_model(model_path)
    flat = sample_images.reshape(sample_images.shape[0], -1)
    latents = pca.transform(flat)
    recon_flat = pca.inverse_transform(latents)
    recon_images = recon_flat.reshape(sample_images.shape)

    metrics = {
        "mse": compute_mse(sample_images, recon_images),
        "psnr": compute_psnr(sample_images, recon_images),
    }

    ensure_dir(output_root)
    save_reconstruction_grid(
        sample_images,
        recon_images,
        output_root / f"pca_dim_{latent_dim:03d}_recon.png",
        title=f"PCA reconstructions (z={latent_dim})",
        grid_count=grid_count,
    )
    save_latent_scatter(
        latents,
        output_root / f"pca_dim_{latent_dim:03d}_latent_scatter.png",
        title=f"PCA latent space (first two dims, z={latent_dim})",
    )
    metrics_path = output_root / f"pca_dim_{latent_dim:03d}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_beta_vae(model_path: Path, device: str | None) -> BetaVAE:
    checkpoint = torch.load(model_path, map_location=device or "cpu")
    latent_dim = checkpoint["config"]["latent_dim"]
    input_shape = tuple(checkpoint.get("input_shape", (3, 96, 96)))
    model = BetaVAE(latent_dim=latent_dim, input_shape=input_shape)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def analyse_beta_vae(
    latent_dim: int,
    sample_images: np.ndarray,
    output_root: Path,
    model_dir: Path,
    grid_count: int,
    device: str | None,
) -> Dict[str, float]:
    model_path = model_dir / "beta_vae" / f"dim_{latent_dim:03d}" / "beta_vae.pt"
    metrics_path = model_dir / "beta_vae" / f"dim_{latent_dim:03d}" / "metrics.json"
    if not model_path.exists():
        print(f"[beta-VAE] Missing model for z={latent_dim}, skipping.")
        return {}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = load_beta_vae(model_path, device)
    model.to(device)

    tensor_images = torch.from_numpy(np.transpose(sample_images, (0, 3, 1, 2))).to(device)

    with torch.no_grad():
        recon_tensors, mu, _ = model(tensor_images)

    recon_images = np.transpose(
        recon_tensors.cpu().numpy(),
        (0, 2, 3, 1),
    )

    metrics = {
        "mse": compute_mse(sample_images, recon_images),
        "psnr": compute_psnr(sample_images, recon_images),
    }
    if metrics_path.exists():
        trained_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.update(
            {
                "training_loss": trained_metrics["loss"][-1],
                "training_recon": trained_metrics["recon"][-1],
                "training_kl": trained_metrics["kl"][-1],
            }
        )
        if "crop_ratio" in trained_metrics:
            metrics["crop_ratio"] = trained_metrics["crop_ratio"]
        if "target_size" in trained_metrics:
            metrics["target_size"] = trained_metrics["target_size"]

    ensure_dir(output_root)
    save_reconstruction_grid(
        sample_images,
        recon_images,
        output_root / f"beta_vae_dim_{latent_dim:03d}_recon.png",
        title=f"beta-VAE reconstructions (z={latent_dim})",
        grid_count=grid_count,
    )
    save_latent_scatter(
        mu.cpu().numpy(),
        output_root / f"beta_vae_dim_{latent_dim:03d}_latent_scatter.png",
        title=f"beta-VAE latent space (first two dims, z={latent_dim})",
    )
    summary_path = output_root / f"beta_vae_dim_{latent_dim:03d}_metrics.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


@dataclass(frozen=True)
class SampleSpec:
    crop_ratio: float | None
    target_size: Tuple[int, int] | None
    greyscale_preset: GreyscalePreset | None


def load_samples_for_spec(
    sample_paths,
    spec: SampleSpec,
    cache: Dict[SampleSpec, np.ndarray],
) -> np.ndarray:
    if spec not in cache:
        cache[spec] = load_image_batch(
            sample_paths,
            normalize=True,
            crop_ratio=spec.crop_ratio,
            target_size=spec.target_size,
            greyscale_preset=spec.greyscale_preset,
        )
    return cache[spec]


def load_pca_metadata(model_dir: Path, latent_dim: int) -> Dict:
    metadata_path = model_dir / "pca" / f"dim_{latent_dim:03d}" / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def spec_from_metadata(metadata: Dict, default_spec: SampleSpec) -> SampleSpec:
    if not metadata:
        return default_spec
    if "greyscale_preset" in metadata and metadata["greyscale_preset"]:
        preset = GreyscalePreset.from_record(metadata["greyscale_preset"])
        return SampleSpec(crop_ratio=None, target_size=None, greyscale_preset=preset)
    target_size = metadata.get("target_size")
    target_tuple = tuple(target_size) if target_size else None
    return SampleSpec(
        crop_ratio=metadata.get("crop_ratio"),
        target_size=target_tuple,
        greyscale_preset=None,
    )


def main() -> None:
    args = parse_args()

    ensure_dir(args.output_dir)
    image_paths = collect_image_paths(args.dataset_dir)
    sample_paths = shuffle_and_limit(image_paths, args.num_samples, args.seed)

    greyscale_preset = None
    if args.greyscale_label:
        preset_path = Path(args.greyscale_presets_path).expanduser().resolve()
        greyscale_preset = load_greyscale_preset(preset_path, args.greyscale_label)
        crop_ratio = None
        target_size = None
        print(
            f"Using greyscale preset '{args.greyscale_label}' for analysis "
            f"({greyscale_preset.output_height}x{greyscale_preset.output_width})."
        )
    else:
        crop_ratio = args.crop_ratio if args.crop_ratio > 0 else None
        target_size = (
            (args.resize_height, args.resize_width)
            if args.resize_height > 0 and args.resize_width > 0
            else None
        )

    default_spec = SampleSpec(
        crop_ratio=crop_ratio,
        target_size=target_size,
        greyscale_preset=greyscale_preset,
    )
    sample_cache: Dict[SampleSpec, np.ndarray] = {}
    sample_images = load_samples_for_spec(sample_paths, default_spec, sample_cache)

    model_aliases = {
        "pca": "pca",
        "beta-vae": "beta-vae",
        "beta_vae": "beta-vae",
        "beta": "beta-vae",
    }
    selected_models: list[str] = []
    for name in args.models:
        canonical = model_aliases.get(name.lower())
        if canonical is None:
            raise ValueError(f"Unknown model type '{name}'. Valid options: pca, beta-vae.")
        if canonical not in selected_models:
            selected_models.append(canonical)

    analysis_summary: Dict[str, Dict[int, Dict[str, float]]] = {
        "pca": {},
        "beta_vae": {},
    }

    for latent_dim in args.latent_dims:
        print(f"\n=== Analysing latent dimension {latent_dim} ===")
        if "pca" in selected_models:
            metadata = load_pca_metadata(args.model_dir, latent_dim)
            spec = spec_from_metadata(metadata, default_spec)
            pca_samples = load_samples_for_spec(sample_paths, spec, sample_cache)
            analysis_summary["pca"][latent_dim] = analyse_pca(
                latent_dim,
                pca_samples,
                args.output_dir,
                args.model_dir,
                args.grid_count,
            )
        if "beta-vae" in selected_models:
            analysis_summary["beta_vae"][latent_dim] = analyse_beta_vae(
                latent_dim,
                sample_images,
                args.output_dir,
                args.model_dir,
                args.grid_count,
                args.device,
            )

    summary_path = args.output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(analysis_summary, indent=2), encoding="utf-8")
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
