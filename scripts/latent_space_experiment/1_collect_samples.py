from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import pygame


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from latent.paths import IMAGE_COLLECTION_DIR, ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect human-played frames from CarRacing-v3."
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default=None,
        help="Optional name for the collection session (defaults to timestamp).",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Save every Nth frame (1 = save all frames).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Stop after saving this many images (default: unlimited).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the session directory if it already exists.",
    )
    parser.add_argument(
        "--capture-reset",
        action="store_true",
        help="Also save the first observation returned after each reset.",
    )
    return parser.parse_args()


def build_session_dir(session_name: Optional[str], overwrite: bool) -> Path:
    base_name = session_name or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir = IMAGE_COLLECTION_DIR / base_name
    if session_dir.exists() and not overwrite:
        counter = 1
        while True:
            candidate = IMAGE_COLLECTION_DIR / f"{base_name}_{counter:02d}"
            if not candidate.exists():
                session_dir = candidate
                break
            counter += 1
    ensure_dir(session_dir)
    return session_dir


def save_frame(path: Path, frame: np.ndarray) -> None:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        # Gym may return float observations in [0, 1] or [0, 255]
        max_value = float(array.max(initial=0.0))
        scale = 255.0 if max_value <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    imageio.imwrite(path, array)


def main() -> None:
    args = parse_args()

    pygame.init()
    clock = pygame.time.Clock()

    session_dir = build_session_dir(args.session_name, args.overwrite)
    print(f"Saving frames to {session_dir}")

    metadata: Dict[str, object] = {
        "session_dir": str(session_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frame_skip": args.frame_skip,
        "max_images": args.max_images,
        "capture_reset": args.capture_reset,
        "episodes": [],
    }

    env = gym.make("CarRacing-v3", render_mode="human")
    running = True
    episode_idx = 0
    saved_images = 0
    global_frame_idx = 0

    try:
        while running:
            obs, info = env.reset()
            episode_frames = 0
            episode_steps = 0
            episode_reward = 0.0

            if args.capture_reset:
                frame_path = session_dir / f"ep{episode_idx:03d}_frame{episode_frames:06d}.png"
                save_frame(frame_path, obs)
                episode_frames += 1
                saved_images += 1
                global_frame_idx += 1
                print(f"[Episode {episode_idx}] Saved reset frame to {frame_path.name}")

                if args.max_images and saved_images >= args.max_images:
                    print("Reached maximum number of images, stopping collection.")
                    break

            steer = 0.0
            playing = True

            while playing:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        playing = False
                        running = False

                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
                    playing = False
                    running = False

                if keys[pygame.K_r]:
                    print(f"[Episode {episode_idx}] Manual reset requested.")
                    playing = False
                    break

                if keys[pygame.K_LEFT]:
                    steer -= 0.06
                elif keys[pygame.K_RIGHT]:
                    steer += 0.06
                else:
                    steer *= 0.85

                steer = float(np.clip(steer, -1.0, 1.0))
                gas = 1.0 if keys[pygame.K_UP] else 0.0
                brake = 1.0 if keys[pygame.K_DOWN] else 0.0
                action = np.array([steer, gas, brake], dtype=np.float32)

                obs, reward, terminated, truncated, info = env.step(action)

                if args.frame_skip <= 1 or (episode_steps % args.frame_skip == 0):
                    frame_path = session_dir / f"ep{episode_idx:03d}_frame{episode_frames:06d}.png"
                    save_frame(frame_path, obs)
                    episode_frames += 1
                    saved_images += 1
                    global_frame_idx += 1
                    if episode_frames % 100 == 0:
                        print(
                            f"[Episode {episode_idx}] Saved {episode_frames} frames "
                            f"(total {saved_images})"
                        )

                episode_reward += reward
                episode_steps += 1

                if args.max_images and saved_images >= args.max_images:
                    print("Reached maximum number of images, stopping collection.")
                    playing = False
                    running = False

                if terminated or truncated:
                    playing = False

                clock.tick(60)

            metadata["episodes"].append(
                {
                    "episode_index": episode_idx,
                    "frames": episode_frames,
                    "steps": episode_steps,
                    "total_reward": float(np.round(episode_reward, 4)),
                }
            )

            with (session_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            episode_idx += 1

            if not running:
                break

    finally:
        env.close()
        pygame.quit()

    print("Collection finished.")
    print(f"Total images saved: {saved_images}")


if __name__ == "__main__":
    main()
