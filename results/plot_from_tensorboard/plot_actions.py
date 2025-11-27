#!/usr/bin/env python3
"""
Plot multi-action learning progression from TensorBoard data.
Shows how the agent learns to use different actions over training.

Supports two visualization modes:
1. Stacked area chart (proportion of each action, sums to 100%)
2. Multi-line chart (absolute probabilities for each action)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
# Action data across 3 training stages
# Each action has 3 files that will be concatenated chronologically
ACTION_FILES = {
    'Brake': [
        'brake.json',
        'brake1.json',
        'brake2.json',
    ],
    'Do Nothing': [
        'noop.json',
        'noop1.json',
        'noop2.json',
    ],
    'Gas': [
        'gas.json',
        'gas1.json',
        'gas2.json',
    ],
    'Left': [
        'right.json',
        'right1.json',
        'right2.json',
    ],
    'Right': [
        'left.json',
        'left1.json',
        'left2.json',
    ],
}

# Choose visualization mode: 'stacked' or 'lines'
MODE = 'stacked'  # Change to 'lines' for multi-line plot

TITLE = "Action Selection Evolution During Training"
XLABEL = "Training Steps"
YLABEL_STACKED = "Action Probability (%)"
YLABEL_LINES = "Action Probability"
SMOOTH_WINDOW = 5  # Smoothing for cleaner visualization
SMOOTH_WINDOW_HIGH = 100  # High smoothing for trend visualization
GENERATE_BOTH = True  # Generate both normal and highly smoothed versions
OUTPUT_PNG = "action_evolution.png"
OUTPUT_PNG_SMOOTH = "action_evolution_smoothed.png"
DPI = 300
FIGSIZE = (12, 6)

# Colors for each action (you can customize these)
ACTION_COLORS = {
    'Brake': '#e74c3c',      # Red
    'Do Nothing': '#95a5a6', # Gray
    'Gas': '#2ecc71',        # Green
    'Left': '#3498db',       # Blue
    'Right': '#f39c12',      # Orange
}

# Style settings (white background, visible grid)
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


def load_json(path):
    """Load TensorBoard JSON export."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with p.open('r') as f:
        return json.load(f)


def load_and_concatenate(file_list):
    """Load multiple JSON files and concatenate them chronologically."""
    all_data = []
    for filepath in file_list:
        try:
            data = load_json(filepath)
            all_data.extend(data)
        except FileNotFoundError:
            print(f"    Warning: {filepath} not found, skipping...")
    
    # Sort by step (second element in [timestamp, step, value])
    all_data.sort(key=lambda x: x[1])
    return all_data


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


def plot_stacked_area(action_data, smooth_window):
    """Create stacked area chart showing action proportions."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    
    # Find common step range across all actions
    all_steps = []
    for action_name, data in action_data.items():
        all_steps.extend(data['steps'])
    
    # Create a common grid of steps
    min_step = min(all_steps)
    max_step = max(all_steps)
    common_steps = np.linspace(min_step, max_step, 1000)  # 1000 points for smooth visualization
    
    # Interpolate all actions to the common grid and smooth
    # Use a specific order to match visual expectations (from bottom to top in stacked plot)
    action_order = ['Brake', 'Do Nothing', 'Gas', 'Left', 'Right']
    
    values_list = []
    labels = []
    colors = []
    
    for action_name in action_order:
        if action_name in action_data:
            steps = action_data[action_name]['steps']
            vals = action_data[action_name]['values']
            
            # Interpolate to common grid
            interp_vals = np.interp(common_steps, steps, vals)
            
            # Apply smoothing if requested
            if smooth_window and smooth_window > 1:
                interp_vals = moving_average(interp_vals, smooth_window)
                # Adjust common_steps for smoothing
                if len(values_list) == 0:  # Only adjust once
                    smoothed_steps = common_steps[smooth_window - 1:]
            
            values_list.append(interp_vals * 100)  # Convert to percentage
            labels.append(action_name)
            colors.append(ACTION_COLORS.get(action_name, '#333333'))
            
            # Debug: print actual mean values
            print(f"  {action_name}: mean probability = {interp_vals.mean()*100:.1f}%")
    
    # Use smoothed steps if smoothing was applied
    if smooth_window and smooth_window > 1:
        plot_steps = smoothed_steps
    else:
        plot_steps = common_steps
    
    # Create stacked area
    ax.stackplot(plot_steps, *values_list, labels=labels, colors=colors, alpha=0.8)
    
    # Formatting
    ax.set_xlabel(XLABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(YLABEL_STACKED, fontsize=14, fontweight='bold')
    ax.set_title(TITLE, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    
    # X-axis formatting
    ax.ticklabel_format(style='plain', axis='x')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Grid and legend
    ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
    # Move legend below the plot to avoid overlapping title
    ax.legend(loc='upper center', fontsize=11, ncol=5, 
             bbox_to_anchor=(0.5, -0.15), frameon=True, 
             edgecolor='#d0d0d0', facecolor='white')
    
    # Adjust layout to make room for legend below
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    return fig, ax


def plot_multiline(action_data, smooth_window):
    """Create multi-line chart showing absolute action probabilities."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    
    # Get common steps
    first_action = list(action_data.keys())[0]
    steps = action_data[first_action]['steps']
    
    # Plot each action as a line
    for action_name in ACTION_FILES.keys():
        if action_name in action_data:
            vals = action_data[action_name]['values']
            action_steps = action_data[action_name]['steps']
            
            if smooth_window and smooth_window > 1:
                vals = moving_average(vals, smooth_window)
                action_steps = action_steps[smooth_window - 1:]
            
            color = ACTION_COLORS.get(action_name, '#333333')
            ax.plot(action_steps, vals, label=action_name, color=color, 
                   linewidth=2.5, alpha=0.85)
    
    # Formatting
    ax.set_xlabel(XLABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(YLABEL_LINES, fontsize=14, fontweight='bold')
    ax.set_title(TITLE, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    
    # X-axis formatting
    ax.ticklabel_format(style='plain', axis='x')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Grid and legend
    ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    return fig, ax


def main():
    print(f"Loading action data from 3 training stages...")
    
    # Load all action data (concatenating multiple files per action)
    action_data = {}
    for action_name, file_list in ACTION_FILES.items():
        try:
            data = load_and_concatenate(file_list)
            if data:
                steps, values = to_arrays(data)
                action_data[action_name] = {'steps': steps, 'values': values}
                print(f"   {action_name}: {len(data)} data points across {len(file_list)} files")
            else:
                print(f"   {action_name}: No data loaded")
        except Exception as e:
            print(f"   {action_name}: Error loading - {e}")
    
    if not action_data:
        print("\n No action data files found! Please check file paths.")
        return
    
    # Generate normal smoothing plot
    print(f"\nCreating {MODE} plot with smoothing window = {SMOOTH_WINDOW}...")
    
    if MODE == 'stacked':
        fig, ax = plot_stacked_area(action_data, SMOOTH_WINDOW)
    else:
        fig, ax = plot_multiline(action_data, SMOOTH_WINDOW)
    
    plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved plot to: {OUTPUT_PNG}")
    plt.close()
    
    # Generate highly smoothed version if requested
    if GENERATE_BOTH:
        print(f"\nCreating {MODE} plot with high smoothing window = {SMOOTH_WINDOW_HIGH}...")
        
        if MODE == 'stacked':
            fig, ax = plot_stacked_area(action_data, SMOOTH_WINDOW_HIGH)
        else:
            fig, ax = plot_multiline(action_data, SMOOTH_WINDOW_HIGH)
        
        plt.savefig(OUTPUT_PNG_SMOOTH, dpi=DPI, bbox_inches='tight')
        print(f"✓ Saved smoothed plot to: {OUTPUT_PNG_SMOOTH}")
        plt.close()
    
    print("\n Done!")
    print("\nTo customize:")
    print(f"  - Edit SMOOTH_WINDOW (currently {SMOOTH_WINDOW}) for normal smoothing")
    print(f"  - Edit SMOOTH_WINDOW_HIGH (currently {SMOOTH_WINDOW_HIGH}) for trend visualization")
    print(f"  - Edit MODE = 'stacked' or 'lines' to change visualization style")
    print(f"  - Set GENERATE_BOTH = False to generate only one version")


if __name__ == '__main__':
    main()
