"""
1) High-quality render shape: (400, 600, 3)
2) Agent observation (state) shape: (96, 96, 3)
3) Processed observation shape: (1, 42, 48)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gymnasium as gym
import numpy as np

from environment import CarRacingPreprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize original vs preprocessed CarRacing frames."
    )
    parser.add_argument("--env-id", type=str, default="CarRacing-v3")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Number of steps to run before capturing (to get car on track)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Create environment WITH render for high-quality visualization
    # Use rgb_array to get the high-res frame programmatically
    env_render = gym.make(args.env_id, render_mode="rgb_array")  # For high-quality display
    env_state = gym.make(args.env_id, render_mode="state_pixels")  # For actual agent observation
    
    # Create preprocessed environment
    env_processed = gym.make(args.env_id, render_mode="state_pixels")
    env_processed = CarRacingPreprocess(env_processed)

    # Reset all environments with same seed
    seed = 42
    _, _ = env_render.reset(seed=seed)
    obs_state, _ = env_state.reset(seed=seed)
    obs_processed, _ = env_processed.reset(seed=seed)

    # Warm up all environments with same actions
    print(f"Warming up for {args.warmup_steps} steps...")
    for step in range(args.warmup_steps):
        # Accelerate action
        action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        
        _, _, term1, trunc1, _ = env_render.step(action)
        obs_state, _, term2, trunc2, _ = env_state.step(action)
        obs_processed, _, term3, trunc3, _ = env_processed.step(action)
        
        if term1 or term2 or term3 or trunc1 or trunc2 or trunc3:
            _, _ = env_render.reset(seed=seed)
            obs_state, _ = env_state.reset(seed=seed)
            obs_processed, _ = env_processed.reset(seed=seed)

    # Get high-quality render for visualization
    render_frame = env_render.render()
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Left: High-quality render (what humans see)
    axes[0].imshow(render_frame)
    axes[0].axis('off')
    
    # Middle: Actual agent observation (low-res)
    axes[1].imshow(obs_state)
    axes[1].axis('off')
    
    # Right: Processed observation
    processed_display = obs_processed.squeeze(0)  # Remove channel dimension
    axes[2].imshow(processed_display, cmap='gray', vmin=0.0, vmax=1.0)
    axes[2].axis('off')
    
    # Remove all padding and margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0)
    
    # Save the figure
    output_dir = ROOT_DIR / "results" / "public"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "preprocessing_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Figura guardada en: {output_path}")
    
    plt.show()

    env_render.close()
    env_state.close()
    env_processed.close()

if __name__ == "__main__":
    main()
