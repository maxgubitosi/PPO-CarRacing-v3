from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from .data import (
    DEFAULT_CROP_RATIO,
    DEFAULT_TARGET_SIZE,
    iter_image_batches,
    shuffle_and_limit,
)
from .paths import ensure_dir
from .greyscale import GreyscalePreset


def _truncate_incremental_pca(full_pca: IncrementalPCA, latent_dim: int) -> IncrementalPCA:
    """Create a new IncrementalPCA instance containing the leading components."""
    truncated = IncrementalPCA(
        n_components=latent_dim,
        whiten=full_pca.whiten,
        batch_size=full_pca.batch_size,
    )
    truncated.components_ = full_pca.components_[:latent_dim].copy()
    truncated.explained_variance_ = full_pca.explained_variance_[:latent_dim].copy()
    truncated.explained_variance_ratio_ = full_pca.explained_variance_ratio_[:latent_dim].copy()
    truncated.singular_values_ = full_pca.singular_values_[:latent_dim].copy()
    truncated.mean_ = full_pca.mean_.copy()
    truncated.var_ = full_pca.var_.copy()
    truncated.noise_variance_ = getattr(full_pca, "noise_variance_", None)
    truncated.n_samples_seen_ = full_pca.n_samples_seen_
    truncated.n_components_ = latent_dim
    truncated.n_features_ = full_pca.components_.shape[1]
    truncated.n_features_in_ = getattr(full_pca, "n_features_in_", truncated.n_features_)
    return truncated


def train_incremental_pca_models(
    image_paths: Sequence[Path],
    latent_dims: Sequence[int],
    output_root: Path,
    batch_size: int = 512,
    max_samples: int | None = None,
    seed: int = 0,
    crop_ratio: float | None = DEFAULT_CROP_RATIO,
    target_size: Tuple[int, int] | None = DEFAULT_TARGET_SIZE,
    greyscale_preset: GreyscalePreset | None = None,
    resize_level: int | None = None,
) -> Mapping[int, dict]:
    """Fit IncrementalPCA once and export separate models for each latent dim."""
    ensure_dir(output_root)
    dims_sorted = sorted({int(dim) for dim in latent_dims})
    if not dims_sorted:
        return {}

    ordered_paths = shuffle_and_limit(image_paths, max_samples, seed)
    max_dim = dims_sorted[-1]
    ipca = IncrementalPCA(n_components=max_dim)

    for batch in tqdm(
        iter_image_batches(
            ordered_paths,
            batch_size,
            normalize=True,
            crop_ratio=crop_ratio,
            target_size=target_size,
            greyscale_preset=greyscale_preset,
        ),
        total=max(1, math.ceil(len(ordered_paths) / batch_size)),
        desc=f"IncrementalPCA (up to z={max_dim})",
        leave=False,
    ):
        flat = batch.reshape(batch.shape[0], -1)
        ipca.partial_fit(flat)

    results: dict[int, dict] = {}
    for dim in dims_sorted:
        truncated = _truncate_incremental_pca(ipca, dim)
        dim_dir = ensure_dir(output_root / f"dim_{dim:03d}")
        with (dim_dir / "pca_model.pkl").open("wb") as f:
            pickle.dump(truncated, f)

        variance_ratio = truncated.explained_variance_ratio_.tolist()
        total_variance = float(np.sum(truncated.explained_variance_ratio_))

        metadata = {
            "latent_dim": dim,
            "n_samples": len(ordered_paths),
            "batch_size": batch_size,
            "max_dim_trained": max_dim,
            "explained_variance_ratio": variance_ratio,
            "total_explained_variance": total_variance,
            "singular_values": truncated.singular_values_.tolist(),
            "crop_ratio": greyscale_preset.crop_ratio if greyscale_preset else crop_ratio,
            "target_size": (
                (greyscale_preset.output_height, greyscale_preset.output_width)
                if greyscale_preset
                else target_size
            ),
            "channel_count": 1 if greyscale_preset else 3,
            "resize_level": resize_level,
        }
        if greyscale_preset is not None:
            metadata["greyscale_preset"] = greyscale_preset.to_dict()
        (dim_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        results[dim] = metadata

    return results
