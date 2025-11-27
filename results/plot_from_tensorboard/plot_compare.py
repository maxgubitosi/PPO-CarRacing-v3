#!/usr/bin/env python3
"""
Simple script to plot a TensorBoard-exported JSON (primary) and overlay a comparison
JSON as a black dashed line. No CLI, configured with sensible defaults so you can
run it directly.

Saves a PNG (high resolution) in the same folder.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration (edit if needed) ---
PRIMARY_JSON = "ppo_clip_20251113-110302.json"  # Discrete Training 01 (primary model)
COMPARE_JSON = "ppo_clip_20251117-183640_084958.json"  # New model comparison (dashed black)
TITLE = "Discrete Training 01 — Comparison"
XLABEL = "Training Steps"
YLABEL = "Mean Reward"
SMOOTH_WINDOW = 2  # default smoothing window (moving average)
OUTPUT_PNG = "discrete_training_01_comparison.png"
DPI = 300
FIGSIZE = (10, 6)

# Match the same visual defaults as the main script (white background, subtle grid)
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#d0d0d0',
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.color': '#d0d0d0',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'font.size': 12,
    'legend.frameon': False,
    'lines.antialiased': True,
})


def load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with p.open('r') as f:
        return json.load(f)


def moving_average(x, w):
    if w is None or w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


def to_arrays(data):
    """Convert list of [timestamp, step, value] to (steps, values) numpy arrays."""
    steps = np.array([entry[1] for entry in data])
    values = np.array([entry[2] for entry in data])
    return steps, values


def main():
    # Load data
    primary = load_json(PRIMARY_JSON)
    compare = load_json(COMPARE_JSON)

    steps_p, values_p = to_arrays(primary)
    steps_c, values_c = to_arrays(compare)

    # Smooth primary (moving average)
    if SMOOTH_WINDOW and SMOOTH_WINDOW > 1:
        smoothed_p = moving_average(values_p, SMOOTH_WINDOW)
        smoothed_steps_p = steps_p[SMOOTH_WINDOW - 1:]
    else:
        smoothed_p = values_p
        smoothed_steps_p = steps_p

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Primary: raw (faint) and smoothed (colored)
    ax.plot(steps_p, values_p, color='#1f77b4', alpha=0.25, linewidth=1, label='Discrete (raw)')
    ax.plot(smoothed_steps_p, smoothed_p, color='#ff7f0e', linewidth=2, label=f'Discrete (smoothed, w={SMOOTH_WINDOW})')

    # Comparison: black dashed line (raw values)
    ax.plot(steps_c, values_c, color='black', linestyle='--', linewidth=1.5, label='Continuous')

    # Labels and title
    ax.set_xlabel(XLABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(YLABEL, fontsize=14, fontweight='bold')
    ax.set_title(TITLE, fontsize=16, fontweight='bold', pad=18)

    # X-axis formatting (show millions if large)
    ax.ticklabel_format(style='plain', axis='x')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if abs(x) >= 1e6 else f'{int(x):,}'))
    
    # Limit X-axis to 5M steps (where both models have data)
    ax.set_xlim(0, 5e6)

    # Grid and legend (legend without frame)
    ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
    ax.legend(loc='best', fontsize=11)

    # Stats box for primary (bigger box)
    stats_text = f"Final: {values_p[-1]:.1f}\nMax: {values_p.max():.1f}\nMean: {values_p.mean():.1f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save and show
    plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches='tight')
    print(f"Saved comparison plot to: {OUTPUT_PNG}")
    plt.show()


if __name__ == '__main__':
    main()
