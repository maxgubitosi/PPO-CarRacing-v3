from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from latent.data import collect_image_paths, shuffle_and_limit  # noqa: E402
from latent.paths import (
    IMAGE_COLLECTION_DIR,
    MODEL_DIR,
    GREYSCALE_PRESETS_PATH,
    ensure_dir,
    variant_subdir,
)  # noqa: E402
from latent.reducers import train_incremental_pca_models  # noqa: E402
from latent.vae import BetaVAEConfig, train_beta_vae  # noqa: E402
from latent.greyscale import load_greyscale_preset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCA and beta-VAE models for latent experiments.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=IMAGE_COLLECTION_DIR,
        help="Directory containing collected frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory where trained models will be stored.",
    )
    parser.add_argument(
        "--latent-dims",
        type=int,
        nargs="+",
        default=[3, 5, 8, 12, 24],
        help="Latent dimensionalities to train.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["pca", "beta-vae"],
        help="Subset of models to train (choices: pca, beta-vae).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pca-batch-size",
        type=int,
        default=512,
        help="Batch size for Incremental PCA fitting.",
    )
    parser.add_argument(
        "--pca-max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use for PCA (default: all).",
    )
    parser.add_argument(
        "--vae-epochs",
        type=int,
        default=30,
        help="Number of epochs for beta-VAE training.",
    )
    parser.add_argument(
        "--vae-batch-size",
        type=int,
        default=128,
        help="Batch size for beta-VAE training.",
    )
    parser.add_argument(
        "--vae-beta",
        type=float,
        default=0.5,
        help="Beta coefficient for beta-VAE KL divergence (lower keeps recon quality).",
    )
    parser.add_argument(
        "--vae-lr",
        type=float,
        default=1e-3,
        help="Learning rate for beta-VAE training.",
    )
    parser.add_argument(
        "--vae-max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use for beta-VAE (default: all).",
    )
    parser.add_argument(
        "--vae-max-steps-per-epoch",
        type=int,
        default=None,
        help="Limit training steps per epoch for beta-VAE (useful for quick runs).",
    )
    parser.add_argument(
        "--vae-num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers for beta-VAE.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device for beta-VAE training (default: mps, falls back automatically).",
    )
    parser.add_argument(
        "--vae-early-stop-patience",
        type=int,
        default=3,
        help="Epoch patience for beta-VAE early stopping (set <=0 to disable).",
    )
    parser.add_argument(
        "--vae-early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum loss improvement required to reset beta-VAE early stopping patience.",
    )
    parser.add_argument(
        "--vae-early-stop-min-rel",
        type=float,
        default=0.1,
        help="Relative improvement (fraction of best loss) required to reset patience.",
    )
    parser.add_argument(
        "--vae-road-weight",
        type=float,
        default=0.0,
        help="Extra weight for dark/road pixels when computing reconstruction loss.",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=0.13,
        help="Fraction of the image height to trim from the bottom (set <=0 to disable).",
    )
    parser.add_argument(
        "--resize-level",
        type=int,
        default=None,
        help="Optional integer factor (>1) to further downscale processed frames and namespace outputs.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=48,
        help="Height to resize processed frames to (set <=0 to keep original).",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=48,
        help="Width to resize processed frames to (set <=0 to keep original).",
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
        help="Preset label to apply. Provide an empty string to disable greyscale preprocessing.",
    )
    return parser.parse_args()


def _scaled_dimension(value: int, resize_level: int | None) -> int:
    if resize_level is None or resize_level <= 1:
        return value
    return max(1, int(math.ceil(value / resize_level)))


def plot_pca_total_variance(
    latent_dims: Sequence[int],
    stats: Mapping[int, dict],
    output_path: Path,
) -> bool:
    dims: list[int] = []
    totals: list[float] = []
    for dim in latent_dims:
        metadata = stats.get(dim)
        if not metadata:
            continue
        total = metadata.get("total_explained_variance")
        if total is None:
            continue
        dims.append(int(dim))
        totals.append(float(total))

    if not dims:
        return False

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dims, totals, marker="o")
    ax.set_xlabel("Latent dimension (z)")
    ax.set_ylabel("Total explained variance")
    ax.set_title("PCA variance captured vs components")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    latent_dims = sorted({int(dim) for dim in args.latent_dims})
    if not latent_dims:
        raise ValueError("At least one latent dimension must be provided.")
    if args.resize_level is not None and args.resize_level < 1:
        raise ValueError("--resize-level must be >= 1 when specified.")

    resize_level = args.resize_level if args.resize_level and args.resize_level > 1 else None

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

    image_paths = collect_image_paths(args.dataset_dir)
    print(f"Found {len(image_paths)} images in {args.dataset_dir}")

    ensure_dir(args.output_dir)

    greyscale_preset = None
    if args.greyscale_label:
        preset_path = Path(args.greyscale_presets_path).expanduser().resolve()
        greyscale_preset = load_greyscale_preset(preset_path, args.greyscale_label)
        if resize_level is not None:
            greyscale_preset = replace(
                greyscale_preset,
                output_height=_scaled_dimension(greyscale_preset.output_height, resize_level),
                output_width=_scaled_dimension(greyscale_preset.output_width, resize_level),
            )
        print(
            f"Using greyscale preset '{args.greyscale_label}' → "
            f"{greyscale_preset.output_height}x{greyscale_preset.output_width}, "
            f"clip[{greyscale_preset.clip_min:.0f},{greyscale_preset.clip_max:.0f}]"
        )
        crop_ratio = None
        target_size = None
    else:
        crop_ratio = args.crop_ratio if args.crop_ratio > 0 else None
        target_size = (
            (
                _scaled_dimension(args.resize_height, resize_level),
                _scaled_dimension(args.resize_width, resize_level),
            )
            if args.resize_height > 0 and args.resize_width > 0
            else None
        )

    if "pca" in selected_models:
        print("\n=== Training PCA models ===")
        pca_root = ensure_dir(variant_subdir(Path(args.output_dir) / "pca", resize_level))
        pca_stats = train_incremental_pca_models(
            image_paths,
            latent_dims=latent_dims,
            output_root=pca_root,
            batch_size=args.pca_batch_size,
            max_samples=args.pca_max_samples,
            seed=args.seed,
            crop_ratio=crop_ratio,
            target_size=target_size,
            greyscale_preset=greyscale_preset,
            resize_level=resize_level,
        )
        for latent_dim in latent_dims:
            if latent_dim in pca_stats:
                stats = pca_stats[latent_dim]
                variance = stats.get("explained_variance_ratio", [])
                total_variance = stats.get("total_explained_variance", float(sum(variance)))
                display_count = min(len(variance), latent_dim)
                variance_display = ", ".join(f"{v:.4f}" for v in variance[:display_count])
                print(
                    f"PCA (z={latent_dim}) trained on {stats['n_samples']} samples; "
                    f"explained variance (first {display_count}): [{variance_display}]; "
                    f"total={total_variance:.4f}"
                )

        plot_path = pca_root / "pca_total_variance.png"
        if plot_pca_total_variance(latent_dims, pca_stats, plot_path):
            print(f"Saved PCA variance plot to {plot_path}")

    if "beta-vae" in selected_models:
        print("\n=== Training beta-VAE models ===")
        beta_root = ensure_dir(variant_subdir(Path(args.output_dir) / "beta_vae", resize_level))
        for latent_dim in latent_dims:
            vae_dir = ensure_dir(beta_root / f"dim_{latent_dim:03d}")
            vae_paths = (
                shuffle_and_limit(image_paths, args.vae_max_samples, args.seed)
                if args.vae_max_samples is not None
                else image_paths
            )

            vae_config = BetaVAEConfig(
                latent_dim=latent_dim,
                beta=args.vae_beta,
                epochs=args.vae_epochs,
                batch_size=args.vae_batch_size,
                learning_rate=args.vae_lr,
                seed=args.seed,
                num_workers=args.vae_num_workers,
                max_steps_per_epoch=args.vae_max_steps_per_epoch,
                early_stop_patience=args.vae_early_stop_patience,
                early_stop_min_delta=args.vae_early_stop_min_delta,
                early_stop_min_rel=args.vae_early_stop_min_rel,
                road_weight=args.vae_road_weight,
            )

            vae_metrics = train_beta_vae(
                vae_paths,
                output_dir=vae_dir,
                config=vae_config,
                device=args.device,
                crop_ratio=crop_ratio,
                target_size=target_size,
                greyscale_preset=greyscale_preset,
            )
            print(
                f"beta-VAE (z={latent_dim}) - "
                f"loss: {vae_metrics['loss']:.4f}, "
                f"recon: {vae_metrics['recon']:.4f}, "
                f"kl: {vae_metrics['kl']:.4f}"
            )


if __name__ == "__main__":
    main()
