#!/usr/bin/env python3
"""
Compare two steering-constrained training runs:
- Only Right: Agent can only turn right
- Only Left: Agent can only turn left
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
JSON_DIR = SCRIPT_DIR / "jsons"
PLOT_DIR = SCRIPT_DIR / "plots"

# JSON files
ONLY_RIGHT_JSON = "ppo_clip_20251129-131127.json"  # only_right constraint
ONLY_LEFT_JSON = "ppo_clip_20251129-132559.json"   # only_left constraint

# Visualization settings
TITLE = "Action Constrained Agents"
XLABEL = "Training Steps"
YLABEL = "Episode Reward"
SMOOTH_WINDOW = 3  # Light smoothing for clarity
OUTPUT_PNG = "steering_constraints_comparison.png"
DPI = 300
FIGSIZE = (12, 7)

# Colors for each constraint
COLORS = {
    'only_right': '#e74c3c',  # Red
    'only_left': '#3498db',   # Blue
}

# Style settings
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
    'legend.frameon': True,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#d0d0d0',
    'lines.antialiased': True,
})


def load_json(filepath):
    """Load TensorBoard JSON export."""
    path = JSON_DIR / filepath
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open('r') as f:
        return json.load(f)


def to_arrays(data):
    """Convert [[timestamp, step, value], ...] to (steps, values)."""
    steps = np.array([entry[1] for entry in data])
    values = np.array([entry[2] for entry in data])
    return steps, values


def moving_average(x, w):
    """Apply moving average smoothing."""
    if w is None or w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


def plot_comparison():
    """Create comparison plot of steering-constrained agents."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    
    # Create plots directory if needed
    PLOT_DIR.mkdir(exist_ok=True)
    
    print("Loading steering constraint experiment data...")
    
    # Load data
    data_right = load_json(ONLY_RIGHT_JSON)
    data_left = load_json(ONLY_LEFT_JSON)
    
    steps_right, values_right = to_arrays(data_right)
    steps_left, values_left = to_arrays(data_left)
    
    # Apply smoothing
    if SMOOTH_WINDOW and SMOOTH_WINDOW > 1:
        values_right_smooth = moving_average(values_right, SMOOTH_WINDOW)
        steps_right_smooth = steps_right[SMOOTH_WINDOW - 1:]
        
        values_left_smooth = moving_average(values_left, SMOOTH_WINDOW)
        steps_left_smooth = steps_left[SMOOTH_WINDOW - 1:]
    else:
        values_right_smooth = values_right
        steps_right_smooth = steps_right
        values_left_smooth = values_left
        steps_left_smooth = steps_left
    
    # Plot Only Right
    ax.plot(steps_right, values_right, 
           color=COLORS['only_right'], 
           alpha=0.25, 
           linewidth=1)
    ax.plot(steps_right_smooth, values_right_smooth, 
           label='Only Right', 
           color=COLORS['only_right'], 
           linewidth=2.5, 
           alpha=0.9)
    
    # Plot Only Left
    ax.plot(steps_left, values_left, 
           color=COLORS['only_left'], 
           alpha=0.25, 
           linewidth=1)
    ax.plot(steps_left_smooth, values_left_smooth, 
           label='Only Left', 
           color=COLORS['only_left'], 
           linewidth=2.5, 
           alpha=0.9)
    
    # Print statistics
    print("\n  Only Right:")
    print(f"    - Data points: {len(data_right)}")
    print(f"    - Final reward: {values_right[-1]:.1f}")
    print(f"    - Mean reward: {values_right.mean():.1f}")
    print(f"    - Max reward: {values_right.max():.1f}")
    
    print("\n  Only Left:")
    print(f"    - Data points: {len(data_left)}")
    print(f"    - Final reward: {values_left[-1]:.1f}")
    print(f"    - Mean reward: {values_left.mean():.1f}")
    print(f"    - Max reward: {values_left.max():.1f}")
    
    # Formatting
    ax.set_xlabel(XLABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(YLABEL, fontsize=14, fontweight='bold')
    ax.set_title(TITLE, fontsize=16, fontweight='bold', pad=20)
    
    # X-axis formatting (show in millions)
    ax.ticklabel_format(style='plain', axis='x')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    
    # Grid and legend
    ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8, zorder=0)
    ax.legend(loc='lower right', fontsize=13, framealpha=0.95, 
             edgecolor='#d0d0d0', facecolor='white')
    
    plt.tight_layout()
    return fig, ax


def main():
    print("=" * 70)
    print("STEERING CONSTRAINTS COMPARISON")
    print("=" * 70)
    print()
    
    # Generate plot
    fig, ax = plot_comparison()
    
    # Save
    output_path = PLOT_DIR / OUTPUT_PNG
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print()
    print(f"Saved plot to: {output_path}")
    plt.close()
    
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    print("This plot compares two agents trained with steering constraints:")
    print("  - ONLY RIGHT: Can only turn right (steering limited to [0, 1])")
    print("  - ONLY LEFT: Can only turn left (steering limited to [-1, 0])")
    print()
    print("Both agents learn to complete the track using only one turning direction,")
    print("demonstrating the adaptability of PPO to constrained action spaces.")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
