from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    DEFAULT_CROP_RATIO,
    DEFAULT_TARGET_SIZE,
    ImageDataset,
)
from .paths import ensure_dir
from .greyscale import GreyscalePreset


@dataclass
class BetaVAEConfig:
    latent_dim: int
    beta: float = 0.5
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 0
    num_workers: int = 0
    max_steps_per_epoch: Optional[int] = None
    early_stop_patience: int = 3
    early_stop_min_delta: float = 1e-4
    early_stop_min_rel: float = 0.1
    road_weight: float = 0.0


class BetaVAE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        input_shape: Sequence[int] | None = None,
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.latent_dim = latent_dim
        self.input_shape = tuple(input_shape) if input_shape is not None else (3, 96, 96)
        self.encoder_hidden_dims = list(hidden_dims)

        encoder_layers = []
        in_channels = self.input_shape[0]
        for h_dim in self.encoder_hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_out_channels = self.encoder_hidden_dims[-1]

        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            self._encoder_shapes = []
            encoded = dummy
            for block in self.encoder:
                encoded = block(encoded)
                self._encoder_shapes.append(tuple(encoded.shape[1:]))
            self._encoded_shape = self._encoder_shapes[-1]
            self._flatten_dim = encoded.view(1, -1).size(1)

        self._decoder_target_shapes = list(reversed(self._encoder_shapes[:-1]))

        self.fc_mu = nn.Linear(self._flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flatten_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self._flatten_dim)

        decoder_blocks = []
        current_channels = self.encoder_out_channels
        for target_shape in self._decoder_target_shapes:
            target_channels = target_shape[0]
            decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, target_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(target_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            current_channels = target_channels

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.output_layer = nn.Sequential(
            nn.Conv2d(current_channels, self.input_shape[0], kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, *self._encoded_shape)
        for block, target in zip(self.decoder_blocks, self._decoder_target_shapes):
            result = F.interpolate(result, size=target[1:], mode="bilinear", align_corners=False)
            result = block(result)
        result = F.interpolate(result, size=self.input_shape[1:], mode="bilinear", align_corners=False)
        result = self.output_layer(result)
        return result

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if weights is not None:
            recon_loss = torch.mean(weights * (recon_x - x) ** 2)
        else:
            recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_divergence
        return loss, recon_loss, kl_divergence


def train_beta_vae(
    image_paths: Sequence[Path],
    output_dir: Path,
    config: BetaVAEConfig,
    device: Optional[str] = None,
    crop_ratio: float | None = DEFAULT_CROP_RATIO,
    target_size: Sequence[int] | None = DEFAULT_TARGET_SIZE,
    greyscale_preset: GreyscalePreset | None = None,
) -> Dict[str, float]:
    """Train a beta-VAE on the provided images and save weights under output_dir."""
    ensure_dir(output_dir)

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if device is None:
        if mps_available:
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = device.lower()
        if device == "mps" and not mps_available:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda") and not torch.cuda.is_available():
            device = "mps" if mps_available else "cpu"

    torch.manual_seed(config.seed)

    if target_size is not None:
        target_size = (int(target_size[0]), int(target_size[1]))

    effective_crop = crop_ratio
    effective_target = (
        (int(target_size[0]), int(target_size[1])) if target_size is not None else None
    )
    if greyscale_preset is not None:
        effective_crop = None
        effective_target = None

    dataset = ImageDataset(
        image_paths,
        normalize=True,
        crop_ratio=effective_crop,
        target_size=effective_target,
        greyscale_preset=greyscale_preset,
    )
    sample_shape = tuple(dataset[0].shape)
    torch_device = torch.device(device)
    use_pin_memory = device.startswith("cuda")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
    )

    model = BetaVAE(latent_dim=config.latent_dim, input_shape=sample_shape).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float("inf")
    history = {"loss": [], "recon": [], "kl": []}
    epochs_without_improvement = 0
    patience = max(0, config.early_stop_patience or 0)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        steps = 0

        progress = tqdm(
            loader,
            desc=f"beta-VAE (z={config.latent_dim}) Epoch {epoch}/{config.epochs}",
            leave=False,
        )
        for step, batch in enumerate(progress, start=1):
            if config.max_steps_per_epoch and step > config.max_steps_per_epoch:
                break

            batch = batch.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            recon_batch, mu, logvar = model(batch)
            weights = None
            if config.road_weight > 0.0:
                weights = 1.0 + config.road_weight * (1.0 - batch)
            loss, recon_loss, kl_loss = model.loss_function(
                recon_batch, batch, mu, logvar, config.beta, weights
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()
            steps += 1

            progress.set_postfix(
                loss=running_loss / steps,
                recon=running_recon / steps,
                kl=running_kl / steps,
            )

        epoch_loss = running_loss / max(steps, 1)
        epoch_recon = running_recon / max(steps, 1)
        epoch_kl = running_kl / max(steps, 1)

        history["loss"].append(epoch_loss)
        history["recon"].append(epoch_recon)
        history["kl"].append(epoch_kl)

        improvement = best_loss - epoch_loss
        min_required = config.early_stop_min_delta
        if best_loss != float("inf") and config.early_stop_min_rel > 0.0:
            min_required = max(min_required, best_loss * config.early_stop_min_rel)
        improved = improvement > min_required
        if improved or best_loss == float("inf"):
            best_loss = epoch_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                    "input_shape": sample_shape,
                    "crop_ratio": crop_ratio,
                    "target_size": target_size,
                },
                output_dir / "beta_vae.pt",
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping beta-VAE (z={config.latent_dim}) at epoch {epoch}; "
                f"best_loss={best_loss:.4f}"
            )
            break

    recorded_target = effective_target
    if greyscale_preset is not None:
        recorded_target = (greyscale_preset.output_height, greyscale_preset.output_width)

    metrics_payload = {
        "loss": history["loss"],
        "recon": history["recon"],
        "kl": history["kl"],
        "crop_ratio": greyscale_preset.crop_ratio if greyscale_preset else crop_ratio,
        "target_size": list(recorded_target) if recorded_target is not None else None,
        "epochs_trained": len(history["loss"]),
        "early_stop_patience": patience,
        "early_stop_min_delta": config.early_stop_min_delta,
        "early_stop_triggered": patience > 0 and epochs_without_improvement >= patience,
    }
    if greyscale_preset is not None:
        metrics_payload["greyscale_preset"] = greyscale_preset.to_dict()

    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return {"loss": history["loss"][-1], "recon": history["recon"][-1], "kl": history["kl"][-1]}
