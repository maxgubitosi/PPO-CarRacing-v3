from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .greyscale import GreyscalePreset

DEFAULT_CROP_RATIO = 0.13
DEFAULT_TARGET_SIZE = (48, 48)  # (height, width)


def _apply_crop(image: Image.Image, crop_ratio: float | None) -> Image.Image:
    if crop_ratio is None or crop_ratio <= 0.0:
        return image
    crop_pixels = int(round(image.height * crop_ratio))
    if crop_pixels <= 0:
        return image
    crop_height = max(image.height - crop_pixels, 1)
    return image.crop((0, 0, image.width, crop_height))


def _apply_resize(image: Image.Image, target_size: Tuple[int, int] | None) -> Image.Image:
    if not target_size:
        return image
    height, width = target_size
    return image.resize((width, height), resample=Image.BILINEAR)


def process_image_array(
    array: np.ndarray,
    crop_ratio: float | None = DEFAULT_CROP_RATIO,
    target_size: Tuple[int, int] | None = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """Process a numpy image array by cropping the bottom section and resizing."""
    image = Image.fromarray(array)
    image = _apply_crop(image, crop_ratio)
    image = _apply_resize(image, target_size)
    return np.asarray(image)


def _prepare_array(
    array: np.ndarray,
    *,
    normalize: bool,
    crop_ratio: float | None,
    target_size: Tuple[int, int] | None,
    greyscale_preset: GreyscalePreset | None,
) -> np.ndarray:
    if greyscale_preset is not None:
        processed = greyscale_preset.apply(array, normalize=normalize)
        return processed.astype(np.float32)

    processed = process_image_array(array, crop_ratio=crop_ratio, target_size=target_size).astype(np.float32)
    if normalize:
        processed /= 255.0
    return processed


def collect_image_paths(root: Path) -> List[Path]:
    """Return all image files stored under the given root directory."""
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_paths = [path for path in sorted(root.rglob("*")) if path.suffix.lower() in extensions]
    if not image_paths:
        raise RuntimeError(f"No images found under {root}")
    return image_paths


def shuffle_and_limit(
    paths: Sequence[Path],
    max_samples: int | None,
    seed: int,
) -> List[Path]:
    """Shuffle image paths and optionally truncate to a maximum number."""
    indices = list(range(len(paths)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_samples is not None:
        indices = indices[:max_samples]
    return [paths[idx] for idx in indices]


def load_image_batch(
    paths: Sequence[Path],
    normalize: bool = True,
    crop_ratio: float | None = DEFAULT_CROP_RATIO,
    target_size: Tuple[int, int] | None = DEFAULT_TARGET_SIZE,
    greyscale_preset: GreyscalePreset | None = None,
) -> np.ndarray:
    """Load a batch of images into a numpy array."""
    batch = []
    for path in paths:
        array = imageio.imread(path)
        array = _prepare_array(
            array,
            normalize=normalize,
            crop_ratio=crop_ratio,
            target_size=target_size,
            greyscale_preset=greyscale_preset,
        )
        batch.append(array)
    return np.stack(batch, axis=0)


def iter_image_batches(
    paths: Sequence[Path],
    batch_size: int,
    normalize: bool = True,
    crop_ratio: float | None = DEFAULT_CROP_RATIO,
    target_size: Tuple[int, int] | None = DEFAULT_TARGET_SIZE,
    greyscale_preset: GreyscalePreset | None = None,
) -> Iterator[np.ndarray]:
    """Yield successive batches of images as numpy arrays."""
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        yield load_image_batch(
            batch_paths,
            normalize=normalize,
            crop_ratio=crop_ratio,
            target_size=target_size,
            greyscale_preset=greyscale_preset,
        )


class ImageDataset(Dataset):
    """PyTorch dataset that loads RGB images and returns tensors in [0, 1]."""

    def __init__(
        self,
        image_paths: Iterable[Path],
        normalize: bool = True,
        crop_ratio: float | None = DEFAULT_CROP_RATIO,
        target_size: Tuple[int, int] | None = DEFAULT_TARGET_SIZE,
        greyscale_preset: GreyscalePreset | None = None,
    ) -> None:
        self.image_paths = list(image_paths)
        if not self.image_paths:
            raise ValueError("ImageDataset requires at least one image path.")

        self.normalize = normalize
        self.crop_ratio = crop_ratio
        self.target_size = target_size
        self.greyscale_preset = greyscale_preset

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.image_paths[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
            array = np.asarray(image)

        processed = _prepare_array(
            array,
            normalize=self.normalize,
            crop_ratio=self.crop_ratio,
            target_size=self.target_size,
            greyscale_preset=self.greyscale_preset,
        )

        tensor = torch.from_numpy(processed)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.permute(2, 0, 1)
        return tensor
