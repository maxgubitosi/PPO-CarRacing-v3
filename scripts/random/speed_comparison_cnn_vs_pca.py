from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ppo_clip.networks_discrete import DiscreteActorCritic
from ppo_clip.networks_latent import LatentActorCritic


@dataclass
class BenchmarkResult:
    name: str
    train_time_ms: float
    train_std_ms: float
    inference_time_ms: float
    inference_std_ms: float
    params: int
    obs_shape: tuple
    train_throughput: float
    inference_throughput: float


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_cnn(
    device: str,
    batch_size: int,
    num_stack: int,
    height: int,
    width: int,
    num_actions: int,
    warmup_iters: int,
    bench_iters: int,
) -> BenchmarkResult:
    obs_shape = (num_stack, height, width)
    obs_space = Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
    action_space = Discrete(num_actions)

    model = DiscreteActorCritic(obs_space, action_space).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    params = count_parameters(model)

    dummy_obs = torch.rand(batch_size, *obs_shape, device=device)
    dummy_actions = torch.randint(0, num_actions, (batch_size,), device=device)

    for _ in range(warmup_iters):
        dist, value = model.get_dist_and_value(dummy_obs)
        log_prob = dist.log_prob(dummy_actions)
        loss = -log_prob.mean() + 0.5 * value.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()

    train_times = []
    for _ in range(bench_iters):
        start = time.perf_counter()
        dist, value = model.get_dist_and_value(dummy_obs)
        log_prob = dist.log_prob(dummy_actions)
        loss = -log_prob.mean() + 0.5 * value.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        train_times.append((time.perf_counter() - start) * 1000)

    for _ in range(warmup_iters):
        with torch.no_grad():
            model.get_dist_and_value(dummy_obs)

    if device == "cuda":
        torch.cuda.synchronize()

    inference_times = []
    for _ in range(bench_iters):
        start = time.perf_counter()
        with torch.no_grad():
            model.get_dist_and_value(dummy_obs)
        if device == "cuda":
            torch.cuda.synchronize()
        inference_times.append((time.perf_counter() - start) * 1000)

    train_mean = float(np.mean(train_times))
    train_std = float(np.std(train_times))
    inf_mean = float(np.mean(inference_times))
    inf_std = float(np.std(inference_times))

    return BenchmarkResult(
        name="CNN",
        train_time_ms=train_mean,
        train_std_ms=train_std,
        inference_time_ms=inf_mean,
        inference_std_ms=inf_std,
        params=params,
        obs_shape=obs_shape,
        train_throughput=batch_size / (train_mean / 1000),
        inference_throughput=batch_size / (inf_mean / 1000),
    )


def benchmark_pca(
    name: str,
    pca_model_path: Path,
    device: str,
    batch_size: int,
    num_stack: int,
    img_height: int,
    img_width: int,
    num_actions: int,
    warmup_iters: int,
    bench_iters: int,
    hidden_dim: int | None = None,
) -> BenchmarkResult:
    with open(pca_model_path, "rb") as f:
        pca = pickle.load(f)

    latent_dim = pca.n_components_
    obs_shape = (latent_dim * num_stack,)
    obs_space = Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
    action_space = Discrete(num_actions)

    model = LatentActorCritic(obs_space, action_space, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    params = count_parameters(model)

    dummy_latent = torch.rand(batch_size, *obs_shape, device=device)
    dummy_actions = torch.randint(0, num_actions, (batch_size,), device=device)

    for _ in range(warmup_iters):
        dist, value = model.get_dist_and_value(dummy_latent)
        log_prob = dist.log_prob(dummy_actions)
        loss = -log_prob.mean() + 0.5 * value.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()

    train_times = []
    for _ in range(bench_iters):
        start = time.perf_counter()
        dist, value = model.get_dist_and_value(dummy_latent)
        log_prob = dist.log_prob(dummy_actions)
        loss = -log_prob.mean() + 0.5 * value.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        train_times.append((time.perf_counter() - start) * 1000)

    pca_mean = pca.mean_.astype(np.float32)
    pca_components = pca.components_.astype(np.float32)
    dummy_frames = np.random.rand(batch_size, img_height * img_width).astype(np.float32)

    for _ in range(warmup_iters):
        centered = dummy_frames - pca_mean
        latent_np = centered @ pca_components.T
        latent_stacked = np.tile(latent_np, num_stack)
        latent_t = torch.from_numpy(latent_stacked).to(device)
        with torch.no_grad():
            model.get_dist_and_value(latent_t)

    if device == "cuda":
        torch.cuda.synchronize()

    inference_times = []
    for _ in range(bench_iters):
        start = time.perf_counter()
        centered = dummy_frames - pca_mean
        latent_np = centered @ pca_components.T
        latent_stacked = np.tile(latent_np, num_stack)
        latent_t = torch.from_numpy(latent_stacked).to(device)
        with torch.no_grad():
            model.get_dist_and_value(latent_t)
        if device == "cuda":
            torch.cuda.synchronize()
        inference_times.append((time.perf_counter() - start) * 1000)

    train_mean = float(np.mean(train_times))
    train_std = float(np.std(train_times))
    inf_mean = float(np.mean(inference_times))
    inf_std = float(np.std(inference_times))

    return BenchmarkResult(
        name=name,
        train_time_ms=train_mean,
        train_std_ms=train_std,
        inference_time_ms=inf_mean,
        inference_std_ms=inf_std,
        params=params,
        obs_shape=obs_shape,
        train_throughput=batch_size / (train_mean / 1000),
        inference_throughput=batch_size / (inf_mean / 1000),
    )


def print_results(results: list[BenchmarkResult], batch_size: int) -> None:
    print("\n" + "=" * 100)
    print(f"SPEED COMPARISON: CNN vs PCA-based PPO (batch_size={batch_size})")
    print("=" * 100)

    print(f"\n{'Model':<25} {'Obs Shape':<20} {'Params':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r.name:<25} {str(r.obs_shape):<20} {r.params:>12,}")

    print(f"\n{'Model':<25} {'Train (ms)':<20} {'Throughput (obs/s)':>20}")
    print("-" * 70)
    for r in results:
        print(f"{r.name:<25} {r.train_time_ms:>8.3f} +/- {r.train_std_ms:<8.3f} {r.train_throughput:>18,.0f}")

    print(f"\n{'Model':<25} {'Inference (ms)':<20} {'Throughput (obs/s)':>20}")
    print("-" * 70)
    for r in results:
        print(f"{r.name:<25} {r.inference_time_ms:>8.3f} +/- {r.inference_std_ms:<8.3f} {r.inference_throughput:>18,.0f}")

    baseline = results[0]
    print(f"\n{'Model':<25} {'Train Speedup':>15} {'Inference Speedup':>20}")
    print("-" * 65)
    for r in results:
        train_speedup = baseline.train_time_ms / r.train_time_ms
        inf_speedup = baseline.inference_time_ms / r.inference_time_ms
        print(f"{r.name:<25} {train_speedup:>14.2f}x {inf_speedup:>19.2f}x")

    print("\n" + "=" * 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speed comparison CNN vs PCA PPO")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=None)
    return parser.parse_args()


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    num_stack = 2
    num_actions = 5
    cnn_height, cnn_width = 42, 48
    pca_lvl1_height, pca_lvl1_width = 42, 48
    pca_lvl4_height, pca_lvl4_width = 11, 12

    pca_lvl1_path = ROOT_DIR / f"scripts/latent_space_experiment/models/pca/resize_lvl1/dim_{args.latent_dim:03d}/pca_model.pkl"
    pca_lvl4_path = ROOT_DIR / f"scripts/latent_space_experiment/models/pca/resize_lvl4/dim_{args.latent_dim:03d}/pca_model.pkl"

    if not pca_lvl1_path.exists():
        print(f"PCA model not found: {pca_lvl1_path}")
        return
    if not pca_lvl4_path.exists():
        print(f"PCA model not found: {pca_lvl4_path}")
        return

    results = []

    print("\nBenchmarking CNN...")
    cnn_result = benchmark_cnn(
        device=device,
        batch_size=args.batch_size,
        num_stack=num_stack,
        height=cnn_height,
        width=cnn_width,
        num_actions=num_actions,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
    )
    results.append(cnn_result)

    print("Benchmarking PCA resize_lvl1 (42x48)...")
    pca_lvl1_result = benchmark_pca(
        name=f"PCA-lvl1 (z={args.latent_dim})",
        pca_model_path=pca_lvl1_path,
        device=device,
        batch_size=args.batch_size,
        num_stack=num_stack,
        img_height=pca_lvl1_height,
        img_width=pca_lvl1_width,
        num_actions=num_actions,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
        hidden_dim=args.hidden_dim,
    )
    results.append(pca_lvl1_result)

    print("Benchmarking PCA resize_lvl4 (11x12)...")
    pca_lvl4_result = benchmark_pca(
        name=f"PCA-lvl4 (z={args.latent_dim})",
        pca_model_path=pca_lvl4_path,
        device=device,
        batch_size=args.batch_size,
        num_stack=num_stack,
        img_height=pca_lvl4_height,
        img_width=pca_lvl4_width,
        num_actions=num_actions,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
        hidden_dim=args.hidden_dim,
    )
    results.append(pca_lvl4_result)

    print_results(results, args.batch_size)


if __name__ == "__main__":
    main()

