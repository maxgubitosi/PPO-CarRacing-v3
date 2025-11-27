#!/usr/bin/env python3
"""
Script to plot TensorBoard exported data for reports.
Creates publication-quality plots from TensorBoard JSON exports.

Usage:
    python plot_tensorboard_data.py --input data.json --title "My Training" --xlabel "Steps" --ylabel "Reward"
    python plot_tensorboard_data.py -i data.json -t "PPO Training"
    python plot_tensorboard_data.py  # Uses defaults
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Default configuration
DEFAULT_INPUT = "ppo_clip_20251117-183640.json"
DEFAULT_TITLE = "PPO Training Progress"
DEFAULT_XLABEL = "Training Steps"
DEFAULT_YLABEL = "Mean Reward"
DEFAULT_LEGEND_RAW = "Raw"
DEFAULT_LEGEND_SMOOTH = "Smoothed"
DEFAULT_SMOOTHING = 60
DEFAULT_DPI = 300
DEFAULT_FIGURE_SIZE = (10, 6)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot TensorBoard exported data with customizable labels and title.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data.json --title "PPO Training"
  %(prog)s -i data.json -t "My Experiment" -x "Episodes" -y "Reward"
  %(prog)s -i data.json --smooth 10
  %(prog)s  # Uses all defaults
        """
    )
    
    parser.add_argument('-i', '--input', 
                       type=str, 
                       default=DEFAULT_INPUT,
                       help=f'Path to input JSON file (default: {DEFAULT_INPUT})')
    
    parser.add_argument('-t', '--title', 
                       type=str, 
                       default=DEFAULT_TITLE,
                       help=f'Plot title and base name for output file (default: "{DEFAULT_TITLE}")')
    
    parser.add_argument('-x', '--xlabel', 
                       type=str, 
                       default=DEFAULT_XLABEL,
                       help=f'X-axis label (default: "{DEFAULT_XLABEL}")')
    
    parser.add_argument('-y', '--ylabel', 
                       type=str, 
                       default=DEFAULT_YLABEL,
                       help=f'Y-axis label (default: "{DEFAULT_YLABEL}")')
    
    parser.add_argument('--legend-raw', 
                       type=str, 
                       default=DEFAULT_LEGEND_RAW,
                       help=f'Legend label for raw data (default: "{DEFAULT_LEGEND_RAW}")')
    
    parser.add_argument('--legend-smooth', 
                       type=str, 
                       default=DEFAULT_LEGEND_SMOOTH,
                       help=f'Legend label for smoothed data (default: "{DEFAULT_LEGEND_SMOOTH}")')
    
    parser.add_argument('-s', '--smooth', 
                       type=int, 
                       default=DEFAULT_SMOOTHING,
                       help=f'Smoothing window size (default: {DEFAULT_SMOOTHING}, use 0 for no smoothing)')
    
    parser.add_argument('--dpi', 
                       type=int, 
                       default=DEFAULT_DPI,
                       help=f'Output resolution in DPI (default: {DEFAULT_DPI})')
    
    parser.add_argument('--no-show', 
                       action='store_true',
                       help='Do not display the plot (only save to file)')
    
    parser.add_argument('--save-pdf', 
                       action='store_true',
                       help='Also save plot as PDF (vector format for LaTeX)')
    
    parser.add_argument('--ylim', 
                       type=float,
                       nargs=2,
                       metavar=('YMIN', 'YMAX'),
                       help='Y-axis limits (e.g., --ylim 0 4)')
    
    parser.add_argument('--no-stats', 
                       action='store_true',
                       help='Do not show statistics box on plot')
    
    return parser.parse_args()


def load_tensorboard_data(filepath):
    """Load data from TensorBoard JSON export."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def plot_training_curve(data, output_file, title, xlabel, ylabel, 
                       legend_raw, legend_smooth, smoothing_window=None, 
                       dpi=DEFAULT_DPI, show_plot=True, save_pdf=False, ylim=None, show_stats=True):
    """
    Create a publication-quality plot from TensorBoard data.
    
    Parameters:
    -----------
    data : list
        List of [timestamp, step, value] entries
    output_file : str
        Path to save the output figure
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    legend_raw : str
        Legend label for raw data
    legend_smooth : str
        Legend label for smoothed data
    smoothing_window : int, optional
        Window size for moving average smoothing
    dpi : int
        Output resolution
    show_plot : bool
        Whether to display the plot
    save_pdf : bool
        Whether to save as PDF in addition to PNG
    """
    # Extract data
    timestamps = np.array([entry[0] for entry in data])
    steps = np.array([entry[1] for entry in data])
    values = np.array([entry[2] for entry in data])
    
    # Create figure
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE, dpi=dpi)
    
    # Plot raw data
    ax.plot(steps, values, alpha=0.3, color='#1f77b4', 
            linewidth=1, label=legend_raw)
    
    # Plot smoothed data if requested
    if smoothing_window and smoothing_window > 1:
        smoothed = np.convolve(values, 
                              np.ones(smoothing_window)/smoothing_window, 
                              mode='valid')
        smoothed_steps = steps[smoothing_window-1:]
        ax.plot(smoothed_steps, smoothed, color='#ff7f0e', 
                linewidth=2, label=f'{legend_smooth} (window={smoothing_window})')
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Format x-axis to show millions
    ax.ticklabel_format(style='plain', axis='x')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    
    # Grid and legend
    ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Add statistics box (larger font and padding) if requested
    if show_stats:
        stats_text = f'Final: {values[-1]:.1f}\nMax: {values.max():.1f}\nMean: {values.mean():.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', alpha=0.5))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_file}")
    
    # Also save as PDF for LaTeX reports (vector format)
    if save_pdf:
        pdf_file = output_file.replace('.png', '.pdf')
        plt.savefig(pdf_file, bbox_inches='tight', format='pdf')
        print(f"✓ Saved PDF to: {pdf_file}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig, ax

def print_statistics(data):
    """Print summary statistics of the training run."""
    values = np.array([entry[2] for entry in data])
    steps = np.array([entry[1] for entry in data])
    
    print("\n" + "="*50)
    print("TRAINING STATISTICS")
    print("="*50)
    print(f"Total steps:        {steps[-1]:,}")
    print(f"Total data points:  {len(data)}")
    print(f"Initial return:     {values[0]:.2f}")
    print(f"Final return:       {values[-1]:.2f}")
    print(f"Maximum return:     {values.max():.2f} (at step {steps[values.argmax()]:,})")
    print(f"Minimum return:     {values.min():.2f}")
    print(f"Mean return:        {values.mean():.2f}")
    print(f"Std deviation:      {values.std():.2f}")
    print(f"Improvement:        {values[-1] - values[0]:.2f} ({((values[-1]/values[0]-1)*100):.1f}%)")
    print("="*50 + "\n")

def main():
    """Main function to load data and create plots."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Custom minimal style resembling TensorBoard (white background, light grid)
    # Remove any previously set style to avoid overrides.
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
    # Ensure tight layout default spacing
    plt.rcParams['figure.autolayout'] = False
    
    # Generate output filename from title (sanitize for filename)
    output_filename = args.title.lower().replace(' ', '_').replace('/', '_')
    output_filename = ''.join(c for c in output_filename if c.isalnum() or c in '_-')
    output_file = f"{output_filename}.png"
    
    print(f"Loading data from: {args.input}")
    
    # Load data
    data = load_tensorboard_data(args.input)
    
    # Print statistics
    print_statistics(data)
    
    # Create plot
    print("Creating plot...")
    plot_training_curve(
        data=data,
        output_file=output_file,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        legend_raw=args.legend_raw,
        legend_smooth=args.legend_smooth,
        smoothing_window=args.smooth if args.smooth > 0 else None,
        dpi=args.dpi,
        show_plot=not args.no_show,
        save_pdf=args.save_pdf,
        ylim=args.ylim,
        show_stats=not args.no_stats
    )
    
    print("\n✓ Done! You can now include the image in your report.")
    if args.save_pdf:
        print("\nTip: Use the PDF for LaTeX documents (vector graphics, scales perfectly)")
    print("Tip: Use the PNG for Word/PowerPoint (already high resolution)")


if __name__ == "__main__":
    main()
