#!/usr/bin/env python3
"""
Plot multiple TensorBoard runs on the same figure.
Reads directly from TensorBoard event files - no JSON export needed.

Usage:
    python plot_multi_runs.py --runs run1 run2 run3 --metric charts/return_mean --title "My Plot"
    python plot_multi_runs.py --runs ppo_clip/ppo_clip_20251129-125106 sb3_ppo_clip/sb3_ppo_clip_20251129-000930 --metric charts/return_mean
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard package required. Install with: pip install tensorboard")
    exit(1)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_TB_DIR = PROJECT_ROOT / "results" / "tensorboard_logs"
LATENT_TB_DIR = PROJECT_ROOT / "scripts" / "latent_space_experiment" / "tensorboard_logs"
OUTPUT_DIR = SCRIPT_DIR / "plots"

COLORS = [
    '#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6',
    '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b'
]

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


def find_run_dir(run_name: str) -> Optional[Path]:
    for base_dir in [RESULTS_TB_DIR, LATENT_TB_DIR]:
        candidate = base_dir / run_name
        if candidate.exists():
            return candidate
    run_only = run_name.split('/')[-1] if '/' in run_name else run_name
    for base_dir in [RESULTS_TB_DIR, LATENT_TB_DIR]:
        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                candidate = subdir / run_only
                if candidate.exists():
                    return candidate
    return None


def load_tensorboard_scalar(run_dir: Path, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files in {run_dir}")
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    available_tags = ea.Tags().get('scalars', [])
    if tag not in available_tags:
        raise ValueError(f"Tag '{tag}' not found. Available: {available_tags}")
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


def plot_multi_runs(
    runs: List[str],
    metric: str,
    title: str,
    ylabel: str,
    output_file: str,
    labels: Optional[List[str]] = None,
    smooth: int = 10,
    show_raw: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    save_pdf: bool = False
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    for i, run in enumerate(runs):
        run_dir = find_run_dir(run)
        if run_dir is None:
            print(f"WARNING: Run '{run}' not found, skipping...")
            continue
        try:
            steps, values = load_tensorboard_scalar(run_dir, metric)
        except Exception as e:
            print(f"WARNING: Could not load metric '{metric}' from '{run}': {e}")
            continue
        label = labels[i] if labels and i < len(labels) else run.split('/')[-1]
        color = COLORS[i % len(COLORS)]
        if show_raw:
            ax.plot(steps, values, color=color, alpha=0.2, linewidth=1)
        if smooth > 1 and len(values) >= smooth:
            smoothed = moving_average(values, smooth)
            smoothed_steps = steps[smooth - 1:]
            ax.plot(smoothed_steps, smoothed, color=color, linewidth=2.5, label=label, alpha=0.9)
        else:
            ax.plot(steps, values, color=color, linewidth=2, label=label, alpha=0.9)
        print(f"  {label}: {len(steps)} points, final={values[-1]:.1f}, max={values.max():.1f}, mean={values.mean():.1f}")
    
    ax.set_xlabel("Training Steps", fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.ticklabel_format(style='plain', axis='x')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if abs(x) >= 1e6 else f'{int(x):,}'))
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8, zorder=0)
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    if save_pdf:
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
        print(f"Saved: {pdf_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot multiple TensorBoard runs together')
    parser.add_argument('--runs', nargs='+', required=True, help='Run names (e.g., ppo_clip/ppo_clip_20251129-125106)')
    parser.add_argument('--metric', default='charts/return_mean', help='TensorBoard metric tag')
    parser.add_argument('--title', default='Training Comparison', help='Plot title')
    parser.add_argument('--ylabel', default='Episode Return', help='Y-axis label')
    parser.add_argument('--output', '-o', default='comparison.png', help='Output filename')
    parser.add_argument('--labels', nargs='+', help='Custom labels for each run')
    parser.add_argument('--smooth', type=int, default=10, help='Smoothing window (0 to disable)')
    parser.add_argument('--no-raw', action='store_true', help='Hide raw data (show only smoothed)')
    parser.add_argument('--xlim', nargs=2, type=float, help='X-axis limits')
    parser.add_argument('--ylim', nargs=2, type=float, help='Y-axis limits')
    parser.add_argument('--pdf', action='store_true', help='Also save as PDF')
    args = parser.parse_args()
    
    print(f"\nPlotting {len(args.runs)} runs...")
    print(f"Metric: {args.metric}")
    print("-" * 50)
    
    plot_multi_runs(
        runs=args.runs,
        metric=args.metric,
        title=args.title,
        ylabel=args.ylabel,
        output_file=args.output,
        labels=args.labels,
        smooth=args.smooth,
        show_raw=not args.no_raw,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
        save_pdf=args.pdf
    )
    print("-" * 50)
    print("Done!")


if __name__ == '__main__':
    main()

