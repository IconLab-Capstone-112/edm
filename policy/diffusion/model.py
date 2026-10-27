"""Preconditioned denoiser for an action chunk, conditioned on cameras and proprioception.

The 1D U-Net is F in the EDM preconditioner. It reads a noise-scaled action chunk
and a continuous noise level (ln σ / 4), and predicts the residual that
``DiffusionPolicy`` turns into a clean chunk. Sampling lives on the policy, so
Heun's method can stay deterministic for DSRL.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
        args = timesteps.float()[:, None] * freqs[None]
        return torch.cat((args.sin(), args.cos()), dim=-1)


class SpatialSoftmax(nn.Module):
    """Feature map to the expected (x, y) image position of each of ``keypoints`` learned features."""

    def __init__(self, channels: int, keypoints: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, keypoints, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        maps = self.proj(features)
        batch, keypoints, height, width = maps.shape
        attention = maps.flatten(2).softmax(dim=-1).view(batch, keypoints, height, width)
        ys = torch.linspace(-1, 1, height, device=maps.device, dtype=maps.dtype)
        xs = torch.linspace(-1, 1, width, device=maps.device, dtype=maps.dtype)
        x = (attention.sum(dim=2) * xs).sum(dim=-1)
        y = (attention.sum(dim=3) * ys).sum(dim=-1)
        return torch.cat((x, y), dim=-1)


def _conv(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.Mish(),
    )


class ConvEncoder(nn.Module):
    """One camera stream to a vector. 96x96 in, ``out_dim`` out.

    Ends in a spatial softmax rather than average pooling. Pooling keeps what is in the
    image and drops where it is, and a grasp needs to know where the cube is.
    """

    def __init__(self, out_dim: int, keypoints: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            _conv(3, 32, 2),
            _conv(32, 64, 1),
            _conv(64, 128, 2),
            _conv(128, 128, 1),
        )
        self.keypoints = SpatialSoftmax(128, keypoints)
        self.fc = nn.Linear(2 * keypoints, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.keypoints(self.net(images)))


class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.conv1(x))
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (scale[:, :, None] + 1) + shift[:, :, None]
        h = F.mish(self.conv2(F.mish(h)))
        h = self.norm2(h)
        return h + self.skip(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, action_dim: int, cond_dim: int, down_dims: tuple[int, ...] = (64, 128, 256)):
        super().__init__()
        step_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(step_dim),
            nn.Linear(step_dim, step_dim * 4),
            nn.Mish(),
            nn.Linear(step_dim * 4, step_dim),
        )
        film_dim = step_dim + cond_dim
        self.film_dim = film_dim

        channels = [down_dims[0], *down_dims]
        self.input_conv = nn.Conv1d(action_dim, down_dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = down_dims[0]
        for level, out_ch in enumerate(down_dims):
            self.down_blocks.append(nn.ModuleList([
                ResidualBlock1d(in_ch, out_ch, film_dim),
                ResidualBlock1d(out_ch, out_ch, film_dim),
            ]))
            if level < len(down_dims) - 1:
                self.downsamples.append(nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch

        self.mid = ResidualBlock1d(down_dims[-1], down_dims[-1], film_dim)
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, out_ch in enumerate(reversed(down_dims)):
            skip_ch = out_ch
            self.up_blocks.append(nn.ModuleList([
                ResidualBlock1d(in_ch + skip_ch, out_ch, film_dim),
                ResidualBlock1d(out_ch, out_ch, film_dim),
            ]))
            if level < len(down_dims) - 1:
                self.upsamples.append(nn.ConvTranspose1d(out_ch, out_ch, 4, stride=2, padding=1))
            in_ch = out_ch
        self.output_conv = nn.Conv1d(down_dims[0], action_dim, 3, padding=1)
        # F starts at 0, so the denoiser starts at c_skip * x, the closed-form EDM denoiser
        # for a Gaussian with variance sigma_data².
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, sample: torch.Tensor, noise_level: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        """``noise_level`` is EDM's c_noise = ln(σ) / 4, one value per batch item."""
        cond = torch.cat((self.time_mlp(noise_level), global_cond), dim=-1)
        h = self.input_conv(sample)
        skips = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, cond)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)
        h = self.mid(h, cond)
        for level, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = torch.cat((h, skip), dim=1)
            for block in blocks:
                h = block(h, cond)
            if level < len(self.upsamples):
                h = self.upsamples[level](h)
        return self.output_conv(h)
