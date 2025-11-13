from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().with_name("greyscale_presets.jsonl")
DEFAULT_CROP_RATIO = 0.13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=240)
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--crop-ratio", type=float, default=DEFAULT_CROP_RATIO)
    parser.add_argument(
        "--downsample",
        dest="downsample",
        action="store_true",
        help="Halve the frame before resizing (matches training preprocessing).",
    )
    parser.add_argument(
        "--no-downsample",
        "--skip-downsample",
        dest="downsample",
        action="store_false",
        help="Keep native resolution (default).",
    )
    parser.set_defaults(downsample=False)
    parser.add_argument("--steps-per-frame", type=int, default=8)
    return parser.parse_args()


def warmup_env(env: gym.Env, steps: int) -> Tuple[np.ndarray, int]:
    obs, _ = env.reset()
    frame = obs
    completed_steps = 0
    for idx in range(max(0, steps)):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        frame = obs
        completed_steps += 1
        if terminated or truncated:
            obs, _ = env.reset()
            frame = obs
    return frame, completed_steps


class GreyscaleTuner:
    def __init__(
        self,
        env: gym.Env,
        frame: np.ndarray,
        *,
        output_path: Path,
        crop_ratio: float,
        downsample: bool,
        steps_per_frame: int,
        initial_step: int,
    ) -> None:
        self.env = env
        self.frame = frame
        self.output_path = output_path
        self.crop_ratio = max(0.0, crop_ratio)
        self.downsample = downsample
        self.steps_per_frame = max(1, steps_per_frame)
        self._raw_weights = DEFAULT_WEIGHTS.copy()
        self._step = initial_step
        self.crop_pixels = int(round(self.frame.shape[0] * self.crop_ratio))
        self.low_clip = 0.0
        self.high_clip = 255.0
        self.raw_height = int(self.frame.shape[0])
        self.raw_width = int(self.frame.shape[1])
        preprocessed = self._apply_pre_resize(self.frame)
        self.output_height = int(
            np.clip(preprocessed.shape[0], 1, int(round(self.raw_height * (1.0 - self.crop_ratio))))
        )
        self.output_width = int(np.clip(preprocessed.shape[1], 1, 96))

        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.subplots_adjust(bottom=0.42, left=0.05, right=0.97, top=0.92)

        color_preview = self._prepare_color(self.frame)
        default_gray = self._convert(self.frame, DEFAULT_WEIGHTS)
        custom_gray = self._convert(self.frame, self._normalized_weights())

        self.color_im = self.axes[0].imshow(color_preview)
        self.custom_im = self.axes[1].imshow(custom_gray, cmap="gray", vmin=0, vmax=255)
        self.default_im = self.axes[2].imshow(default_gray, cmap="gray", vmin=0, vmax=255)

        self.axes[0].set_title("RGB preview")
        self.axes[1].set_title("Custom grayscale")
        self.axes[2].set_title("Default grayscale")
        for axis in self.axes:
            axis.axis("off")

        slider_width = 0.34
        slider_height = 0.035
        left_x = 0.09
        right_x = 0.55
        row_y = [0.36, 0.30, 0.24, 0.18]

        slider_r_ax = self.fig.add_axes([left_x, row_y[0], slider_width, slider_height])
        slider_g_ax = self.fig.add_axes([left_x, row_y[1], slider_width, slider_height])
        slider_b_ax = self.fig.add_axes([left_x, row_y[2], slider_width, slider_height])

        self.slider_r = Slider(slider_r_ax, "Red", 0.0, 2.0, valinit=float(DEFAULT_WEIGHTS[0]))
        self.slider_g = Slider(slider_g_ax, "Green", 0.0, 2.0, valinit=float(DEFAULT_WEIGHTS[1]))
        self.slider_b = Slider(slider_b_ax, "Blue", 0.0, 2.0, valinit=float(DEFAULT_WEIGHTS[2]))
        self.sliders = [self.slider_r, self.slider_g, self.slider_b]
        for slider in self.sliders:
            slider.on_changed(self._on_slider_change)

        clip_min_ax = self.fig.add_axes([right_x, row_y[0], slider_width, slider_height])
        clip_max_ax = self.fig.add_axes([right_x, row_y[1], slider_width, slider_height])
        reduce_h_ax = self.fig.add_axes([right_x, row_y[2], slider_width, slider_height])
        reduce_w_ax = self.fig.add_axes([right_x, row_y[3], slider_width, slider_height])

        self.slider_low_clip = Slider(clip_min_ax, "Clip min", 0.0, 255.0, valinit=self.low_clip)
        self.slider_high_clip = Slider(clip_max_ax, "Clip max", 0.0, 255.0, valinit=self.high_clip)
        max_height_after_crop = max(1, int(round(self.raw_height * (1.0 - self.crop_ratio))))
        self.slider_output_height = Slider(
            reduce_h_ax,
            "Height px",
            1.0,
            float(max_height_after_crop),
            valinit=float(self.output_height),
            valstep=1.0,
        )
        self.slider_output_width = Slider(
            reduce_w_ax,
            "Width px",
            1.0,
            96.0,
            valinit=float(self.output_width),
            valstep=1.0,
        )

        self.slider_low_clip.on_changed(self._on_clip_change)
        self.slider_high_clip.on_changed(self._on_clip_change)
        self.slider_output_height.on_changed(self._on_size_change)
        self.slider_output_width.on_changed(self._on_size_change)

        save_ax = self.fig.add_axes([0.12, 0.05, 0.16, 0.05])
        next_ax = self.fig.add_axes([0.32, 0.05, 0.16, 0.05])
        reset_ax = self.fig.add_axes([0.52, 0.05, 0.16, 0.05])
        label_ax = self.fig.add_axes([0.74, 0.05, 0.22, 0.05])

        self.save_button = Button(save_ax, "Save preset")
        self.next_button = Button(next_ax, "Next frame")
        self.reset_button = Button(reset_ax, "Reset weights")
        self.name_box = TextBox(label_ax, "Label", initial="preset-1")

        self.save_button.on_clicked(self._on_save)
        self.next_button.on_clicked(self._on_next_frame)
        self.reset_button.on_clicked(self._on_reset)
        self.name_box.on_submit(self._on_label_submit)

        self._update_slider_ticks()
        self.weight_text = self.fig.text(0.05, 0.94, "", fontsize=11, transform=self.fig.transFigure)
        self.clip_text = self.fig.text(0.55, 0.94, "", fontsize=11, transform=self.fig.transFigure)
        self.status_text = self.fig.text(0.05, 0.02, "", fontsize=10, transform=self.fig.transFigure)

        self._update_weight_display()
        self._update_preprocess_display()
        self._set_status("Ready")

    def run(self) -> None:
        plt.show()

    def _prepare_color(self, frame: np.ndarray) -> np.ndarray:
        return self._apply_spatial_ops(frame)

    def _apply_pre_resize(self, frame: np.ndarray) -> np.ndarray:
        working = frame
        if self.crop_pixels > 0:
            working = working[: working.shape[0] - self.crop_pixels, :, :]
        if self.downsample:
            working = working[::2, ::2, :]
        return working

    def _apply_spatial_ops(self, frame: np.ndarray) -> np.ndarray:
        base = self._apply_pre_resize(frame)
        target_h = max(1, int(self.output_height))
        target_w = max(1, int(self.output_width))
        if base.shape[0] != target_h or base.shape[1] != target_w:
            interpolation = (
                cv2.INTER_AREA
                if target_h < base.shape[0] or target_w < base.shape[1]
                else cv2.INTER_LINEAR
            )
            base = cv2.resize(base, (target_w, target_h), interpolation=interpolation)
        return base

    def _convert(self, frame: np.ndarray, weights: np.ndarray) -> np.ndarray:
        spatial = self._apply_spatial_ops(frame).astype(np.float32)
        gray = np.tensordot(spatial, weights, axes=([2], [0]))
        low, high = self._clip_bounds()
        gray = np.clip(gray, low, high)
        scale = high - low
        if scale <= 1e-6:
            scale = 1.0
        normalized = (gray - low) / scale
        return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)

    def _normalized_weights(self) -> np.ndarray:
        total = float(np.sum(self._raw_weights))
        if total <= 1e-6:
            return np.full(3, 1.0 / 3.0, dtype=np.float32)
        return (self._raw_weights / total).astype(np.float32)

    def _on_slider_change(self, _value: float) -> None:
        self._raw_weights = np.array([slider.val for slider in self.sliders], dtype=np.float32)
        self._update_weight_display()
        self._update_grayscale_images()
        self.fig.canvas.draw_idle()

    def _on_clip_change(self, _value: float) -> None:
        self.low_clip = float(self.slider_low_clip.val)
        self.high_clip = float(self.slider_high_clip.val)
        self._update_preprocess_display()
        self._update_grayscale_images()
        self.fig.canvas.draw_idle()

    def _on_size_change(self, _value: float) -> None:
        self.output_height = max(1, int(round(self.slider_output_height.val)))
        self.output_width = max(1, int(round(self.slider_output_width.val)))
        self._update_preprocess_display()
        color_preview = self._prepare_color(self.frame)
        self.color_im.set_data(color_preview)
        self._update_grayscale_images()
        self.fig.canvas.draw_idle()

    def _on_save(self, _event) -> None:
        weights = self._normalized_weights()
        label = self.name_box.text.strip()
        output_h, output_w = self._current_output_shape()
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "label": label or None,
            "weights": {
                "normalized": [float(x) for x in weights],
                "raw": [float(x) for x in self._raw_weights],
            },
            "crop_ratio": self.crop_ratio,
            "downsample": self.downsample,
            "step": self._step,
            "clip": {
                "min": float(self.low_clip),
                "max": float(self.high_clip),
            },
            "output_resolution": {
                "height": int(output_h),
                "width": int(output_w),
            },
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record))
            handle.write("\n")
        self._set_status(f"Saved preset to {self.output_path}")

    def _on_next_frame(self, _event) -> None:
        for _ in range(self.steps_per_frame):
            obs, _, terminated, truncated, _ = self.env.step(self.env.action_space.sample())
            self.frame = obs
            self._step += 1
            if terminated or truncated:
                obs, _ = self.env.reset()
                self.frame = obs
                self._step = 0
        self._refresh_frame()
        self._set_status(f"Frame updated at step {self._step}")

    def _on_reset(self, _event) -> None:
        for slider, default in zip(self.sliders, DEFAULT_WEIGHTS):
            slider.set_val(float(default))
        self.slider_low_clip.set_val(0.0)
        self.slider_high_clip.set_val(255.0)
        defaults = self._apply_pre_resize(self.frame)
        default_height = int(
            np.clip(defaults.shape[0], 1, int(round(self.raw_height * (1.0 - self.crop_ratio))))
        )
        default_width = int(np.clip(defaults.shape[1], 1, 96))
        self.slider_output_height.set_val(float(default_height))
        self.slider_output_width.set_val(float(default_width))
        self._set_status("Parameters reset to defaults")

    def _on_label_submit(self, text: str) -> None:
        clean = text.strip()
        self._set_status(f"Label set to {clean or '(none)'}")

    def _update_grayscale_images(self) -> None:
        custom_gray = self._convert(self.frame, self._normalized_weights())
        default_gray = self._convert(self.frame, DEFAULT_WEIGHTS)
        self.custom_im.set_data(custom_gray)
        self.default_im.set_data(default_gray)

    def _update_slider_ticks(self) -> None:
        base = self._apply_pre_resize(self.frame)
        self._configure_slider_ticks(self.slider_output_height, base.shape[0])
        self._configure_slider_ticks(self.slider_output_width, base.shape[1])

    def _configure_slider_ticks(self, slider: Slider, base_dimension: int) -> None:
        if base_dimension <= 0:
            return
        marks = {int(round(base_dimension))}
        for divisor in (2, 4, 8, 16):
            value = int(round(base_dimension / divisor))
            if 1 <= value <= slider.valmax:
                marks.add(value)
        ticks = sorted(marks)
        slider.ax.set_xticks(ticks)
        slider.ax.set_xticklabels([str(tick) for tick in ticks])
        slider.ax.tick_params(axis="x", labelsize=8)

    def _refresh_frame(self) -> None:
        self.crop_pixels = int(round(self.frame.shape[0] * self.crop_ratio))
        color_preview = self._prepare_color(self.frame)
        self.color_im.set_data(color_preview)
        self._update_grayscale_images()
        self._update_slider_ticks()
        self._update_preprocess_display()
        self.fig.canvas.draw_idle()

    def _update_weight_display(self) -> None:
        weights = self._normalized_weights()
        self.weight_text.set_text(
            f"Normalized weights | R: {weights[0]:.3f}  G: {weights[1]:.3f}  B: {weights[2]:.3f}"
        )

    def _update_preprocess_display(self) -> None:
        output_h, output_w = self._current_output_shape()
        base = self._apply_pre_resize(self.frame)
        self.clip_text.set_text(
            f"Clip [{self.low_clip:.0f}, {self.high_clip:.0f}] | Raw {self.raw_height}×{self.raw_width} → Base {base.shape[0]}×{base.shape[1]} → Output {output_h}×{output_w}"
        )

    def _set_status(self, message: str) -> None:
        self.status_text.set_text(message)
        self.fig.canvas.draw_idle()

    def _clip_bounds(self) -> tuple[float, float]:
        low = float(np.clip(self.low_clip, 0.0, 255.0))
        high = float(np.clip(self.high_clip, 0.0, 255.0))
        if high <= low:
            if low >= 255.0:
                low = 254.0
            high = min(255.0, low + 1.0)
        return low, high

    def _current_output_shape(self) -> tuple[int, int]:
        return int(self.output_height), int(self.output_width)

def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        continuous=not args.discrete,
        lap_complete_percent=0.95,
        domain_randomize=False,
    )
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    frame, warmup_steps = warmup_env(env, args.warmup_steps)
    print(f"Initial frame shape: {frame.shape}")

    tuner = GreyscaleTuner(
        env,
        frame,
        output_path=output_path,
        crop_ratio=args.crop_ratio,
        downsample=args.downsample,
        steps_per_frame=args.steps_per_frame,
        initial_step=warmup_steps,
    )
    try:
        tuner.run()
    finally:
        env.close()


if __name__ == "__main__":
    main()

