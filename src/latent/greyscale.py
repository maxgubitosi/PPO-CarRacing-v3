from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class GreyscalePreset:
    label: str | None
    weights: tuple[float, float, float]
    crop_ratio: float
    downsample: bool
    clip_min: float
    clip_max: float
    output_height: int
    output_width: int

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> GreyscalePreset:
        weights_entry = record.get("weights")
        weights_source = None
        if isinstance(weights_entry, Mapping):
            normalized = weights_entry.get("normalized")
            raw = weights_entry.get("raw")
            weights_source = normalized or raw
        elif isinstance(weights_entry, Sequence):
            weights_source = weights_entry
        if not weights_source:
            raise ValueError("Preset record missing RGB weights.")
        weights = tuple(float(x) for x in weights_source)
        clip_entry = record.get("clip")
        if clip_entry:
            clip_min = float(clip_entry.get("min", 0.0))
            clip_max = float(clip_entry.get("max", 255.0))
        else:
            clip_min = float(record.get("clip_min", 0.0))
            clip_max = float(record.get("clip_max", 255.0))

        output_entry = record.get("output_resolution")
        if output_entry:
            output_height = int(output_entry.get("height", record.get("output_height", 96)))
            output_width = int(output_entry.get("width", record.get("output_width", 96)))
        else:
            output_height = int(record.get("output_height", 96))
            output_width = int(record.get("output_width", 96))
        return cls(
            label=record.get("label"),
            weights=weights,  # type: ignore[arg-type]
            crop_ratio=float(record.get("crop_ratio", 0.0) or 0.0),
            downsample=bool(record.get("downsample", False)),
            clip_min=clip_min,
            clip_max=clip_max,
            output_height=output_height,
            output_width=output_width,
        )

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "weights": list(self.weights),
            "crop_ratio": self.crop_ratio,
            "downsample": self.downsample,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "output_height": self.output_height,
            "output_width": self.output_width,
        }

    def apply(
        self,
        frame: np.ndarray,
        *,
        normalize: bool = True,
        keepdims: bool = True,
    ) -> np.ndarray:
        base = self._preprocess_frame(frame)
        gray = np.tensordot(base, self._normalized_weights(), axes=([2], [0]))

        low, high = self._clip_bounds()
        clipped = np.clip(gray, low, high)
        scale = max(high - low, 1e-6)
        normalized = np.clip((clipped - low) / scale, 0.0, 1.0).astype(np.float32)

        output = normalized if normalize else normalized * 255.0
        if keepdims:
            output = output[..., None]
        return output.astype(np.float32)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("GreyscalePreset expects RGB frames with shape (H, W, 3).")
        working = np.asarray(frame, dtype=np.float32)
        if working.max(initial=0.0) <= 1.0:
            working = working * 255.0

        height = working.shape[0]
        crop_pixels = int(round(height * max(0.0, self.crop_ratio)))
        if crop_pixels > 0 and crop_pixels < height:
            working = working[: height - crop_pixels, :, :]

        if self.downsample:
            working = working[::2, ::2, :]

        target_h = max(1, int(self.output_height))
        target_w = max(1, int(self.output_width))
        if working.shape[0] != target_h or working.shape[1] != target_w:
            interpolation = (
                cv2.INTER_AREA if target_h < working.shape[0] or target_w < working.shape[1] else cv2.INTER_LINEAR
            )
            working = cv2.resize(working, (target_w, target_h), interpolation=interpolation)
        return working

    def _clip_bounds(self) -> tuple[float, float]:
        low = float(min(self.clip_min, self.clip_max))
        high = float(max(self.clip_min, self.clip_max))
        if np.isclose(high, low):
            high = low + 1.0
        return low, high

    def _normalized_weights(self) -> np.ndarray:
        weights = np.asarray(self.weights, dtype=np.float32)
        total = float(np.sum(weights))
        if total <= 1e-6:
            return np.full(3, 1.0 / 3.0, dtype=np.float32)
        return weights / total


def iter_greyscale_presets(presets_path: Path) -> Iterable[GreyscalePreset]:
    path = Path(presets_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Greyscale presets file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            yield GreyscalePreset.from_record(record)


def load_greyscale_preset(presets_path: Path, label: str) -> GreyscalePreset:
    label_clean = label.strip().lower()
    if not label_clean:
        raise ValueError("Preset label cannot be empty.")

    for preset in iter_greyscale_presets(presets_path):
        preset_label = (preset.label or "").strip().lower()
        if preset_label == label_clean:
            return preset
    raise ValueError(f"Preset '{label}' not found in {presets_path}")

