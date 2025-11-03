from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from latent.data import collect_image_paths, shuffle_and_limit  # noqa: E402
from latent.paths import IMAGE_COLLECTION_DIR, MODEL_DIR, ensure_dir  # noqa: E402
from latent.reducers import train_incremental_pca_models, train_tsne  # noqa: E402
from latent.vae import BetaVAEConfig, train_beta_vae  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCA, t-SNE, and beta-VAE models.")
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
        default=[3, 12, 32],
        help="Latent dimensionalities to train.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["pca", "tsne", "beta-vae"],
        help="Subset of models to train (choices: pca, tsne, beta-vae).",
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
        "--tsne-max-samples",
        type=int,
        default=10000,
        help="Maximum number of samples to use for t-SNE training.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=None,
        help="Override t-SNE perplexity (auto if not provided).",
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
        default=4.0,
        help="Beta coefficient for beta-VAE KL divergence.",
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
        "--crop-ratio",
        type=float,
        default=0.13,
        help="Fraction of the image height to trim from the bottom (set <=0 to disable).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latent_dims = sorted({int(dim) for dim in args.latent_dims})
    if not latent_dims:
        raise ValueError("At least one latent dimension must be provided.")

    model_aliases = {
        "pca": "pca",
        "tsne": "tsne",
        "t-sne": "tsne",
        "beta-vae": "beta-vae",
        "beta_vae": "beta-vae",
        "beta": "beta-vae",
    }
    selected_models: list[str] = []
    for name in args.models:
        canonical = model_aliases.get(name.lower())
        if canonical is None:
            raise ValueError(f"Unknown model type '{name}'. Valid options: pca, tsne, beta-vae.")
        if canonical not in selected_models:
            selected_models.append(canonical)

    image_paths = collect_image_paths(args.dataset_dir)
    print(f"Found {len(image_paths)} images in {args.dataset_dir}")

    ensure_dir(args.output_dir)

    crop_ratio = args.crop_ratio if args.crop_ratio > 0 else None
    target_size = (
        (args.resize_height, args.resize_width)
        if args.resize_height > 0 and args.resize_width > 0
        else None
    )

    if "pca" in selected_models:
        print("\n=== Training PCA models ===")
        pca_root = ensure_dir(Path(args.output_dir) / "pca")
        pca_stats = train_incremental_pca_models(
            image_paths,
            latent_dims=latent_dims,
            output_root=pca_root,
            batch_size=args.pca_batch_size,
            max_samples=args.pca_max_samples,
            seed=args.seed,
            crop_ratio=crop_ratio,
            target_size=target_size,
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

    if "tsne" in selected_models:
        print("\n=== Training t-SNE models ===")
        for latent_dim in latent_dims:
            tsne_dir = ensure_dir(Path(args.output_dir) / "tsne" / f"dim_{latent_dim:03d}")
            tsne_stats = train_tsne(
                image_paths,
                latent_dim=latent_dim,
                output_dir=tsne_dir,
                max_samples=args.tsne_max_samples,
                seed=args.seed,
                perplexity=args.tsne_perplexity,
                crop_ratio=crop_ratio,
                target_size=target_size,
            )
            print(
                f"t-SNE (z={latent_dim}) trained on {tsne_stats['n_samples']} samples; "
                f"method={tsne_stats['method']}; "
                f"KL divergence: {tsne_stats['kl_divergence']:.4f}"
            )

    if "beta-vae" in selected_models:
        print("\n=== Training beta-VAE models ===")
        for latent_dim in latent_dims:
            vae_dir = ensure_dir(Path(args.output_dir) / "beta_vae" / f"dim_{latent_dim:03d}")
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
            )

            vae_metrics = train_beta_vae(
                vae_paths,
                output_dir=vae_dir,
                config=vae_config,
                device=args.device,
                crop_ratio=crop_ratio,
                target_size=target_size,
            )
            print(
                f"beta-VAE (z={latent_dim}) - "
                f"loss: {vae_metrics['loss']:.4f}, "
                f"recon: {vae_metrics['recon']:.4f}, "
                f"kl: {vae_metrics['kl']:.4f}"
            )


if __name__ == "__main__":
    main()
