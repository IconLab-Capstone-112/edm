"""Diffusion policy over joint-delta chunks, trained and sampled like EDM.

EDM (Karras et al., 2022) does not use a discrete beta schedule. A continuous
noise level σ is drawn from a log-normal during training, the network is
preconditioned so its raw output stays O(1), and sampling integrates the
probability-flow ODE with Heun's method.

DSRL freezes this network and learns a policy over the noise ``sample`` starts
from. Pass that noise as ``noise`` with shape ``(batch, pred_horizon, action_dim)``,
drawn from a standard normal. The sampler multiplies it by ``sigma_max``
internally and adds no further randomness (EDM's S_churn is 0), so one noise
tensor always decodes to the same action chunk.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .model import ConditionalUnet1D, ConvEncoder


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        n_cameras: int,
        state_dim: int,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        n_inference_steps: int = 18,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        feature_dim: int = 64,
    ):
        super().__init__()
        if n_inference_steps < 2:
            raise ValueError("Heun's method needs at least 2 inference steps")
        self.n_cameras = n_cameras
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.n_inference_steps = n_inference_steps
        self.feature_dim = feature_dim
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)

        self.image_encoders = nn.ModuleList(ConvEncoder(feature_dim) for _ in range(n_cameras))
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, feature_dim), nn.Mish(), nn.Linear(feature_dim, feature_dim))
        cond_dim = obs_horizon * (n_cameras * feature_dim + feature_dim)
        self.unet = ConditionalUnet1D(action_dim, cond_dim)

        # σ_data is overwritten from the training actions. 0.5 is EDM's value for data in [-1, 1].
        self.register_buffer("sigma_data", torch.tensor(float(sigma_data)))
        self.register_buffer("state_low", torch.zeros(state_dim))
        self.register_buffer("state_span", torch.ones(state_dim))
        self.register_buffer("action_low", torch.zeros(action_dim))
        self.register_buffer("action_span", torch.ones(action_dim))

    @property
    def latent_shape(self) -> tuple[int, int]:
        """Shape of the standard-normal noise DSRL acts on, per batch item."""
        return (self.pred_horizon, self.action_dim)

    def set_normalizer(self, state_low, state_high, action_low, action_high, actions=None) -> None:
        self.state_low.copy_(torch.as_tensor(state_low, dtype=self.state_low.dtype))
        self.state_span.copy_(torch.as_tensor(state_high - state_low, dtype=self.state_span.dtype).clamp_min(1e-3))
        self.action_low.copy_(torch.as_tensor(action_low, dtype=self.action_low.dtype))
        self.action_span.copy_(torch.as_tensor(action_high - action_low, dtype=self.action_span.dtype).clamp_min(1e-3))
        if actions is not None:
            raw = torch.as_tensor(actions, dtype=self.action_low.dtype, device=self.action_low.device)
            normalized = (raw - self.action_low) / self.action_span * 2 - 1
            self.sigma_data.fill_(normalized.std().clamp_min(1e-3))

    def condition(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """``images`` is (B, obs, cameras, 3, H, W) in [0, 1]. ``state`` is raw proprioception."""
        images = images * 2 - 1
        state = (state - self.state_low) / self.state_span * 2 - 1
        batch, obs, cameras = images.shape[:3]
        pieces = []
        for t in range(obs):
            visual = [self.image_encoders[c](images[:, t, c]) for c in range(cameras)]
            pieces.append(torch.cat((*visual, self.state_encoder(state[:, t])), dim=-1))
        return torch.cat(pieces, dim=-1)

    def loss(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """EDM weighted denoising loss, trained on the preconditioned network output.

        λ(σ) c_out(σ)² = 1, so mean-squared error against
        (x - c_skip y) / c_out is the paper's objective without the huge weights
        that appear at small σ. Pass ``generator`` to draw the same σ and noise
        every call, which makes validation losses comparable across epochs.
        """
        cond = self.condition(images, state)
        clean = self._normalize_action(action).transpose(1, 2)
        sigma = torch.randn(clean.shape[0], device=clean.device, generator=generator)
        sigma = (sigma * self.p_std + self.p_mean).exp()[:, None, None]
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        noisy = clean + noise * sigma
        c_skip, c_out, c_in, c_noise = self._edm_coeffs(sigma)
        prediction = self.unet(c_in * noisy, c_noise, cond)
        target = (clean - c_skip * noisy) / c_out
        return nn.functional.mse_loss(prediction, target)

    @torch.no_grad()
    def sample(self, images: torch.Tensor, state: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Decode an action chunk. ``noise`` is the DSRL latent, shaped (B, pred_horizon, action_dim)."""
        return self.sample_from_condition(self.condition(images, state), noise)

    @torch.no_grad()
    def sample_from_condition(self, cond: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """``sample`` for an already encoded observation, so DSRL encodes each frame once."""
        batch = cond.shape[0]
        if noise is None:
            noise = torch.randn(batch, self.pred_horizon, self.action_dim, device=cond.device)
        elif noise.shape[1:] != self.latent_shape:
            raise ValueError(f"noise must have shape (batch, {self.pred_horizon}, {self.action_dim})")
        x = noise.transpose(1, 2) * self.sigma_max
        sigmas = self._sigma_schedule(x.device, x.dtype)
        for i in range(self.n_inference_steps):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            denoised = self._denoise(x, sigma, cond)
            derivative = (x - denoised) / sigma
            x_next = x + (sigma_next - sigma) * derivative
            if float(sigma_next) > 0:
                denoised_next = self._denoise(x_next, sigma_next, cond)
                derivative_next = (x_next - denoised_next) / sigma_next
                x_next = x + (sigma_next - sigma) * 0.5 * (derivative + derivative_next)
            x = x_next
        return self._denormalize_action(x.transpose(1, 2).clamp(-1, 1))

    def save(self, path: Path, config: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.state_dict(), "config": config}, path)

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cpu") -> tuple["DiffusionPolicy", dict]:
        blob = torch.load(path, map_location=device, weights_only=False)
        config = blob["config"]
        if "sigma_max" not in config:
            raise ValueError(f"{path} is a DDPM checkpoint. Retrain with the EDM trainer.")
        policy = cls(
            n_cameras=len(config["cameras"]),
            state_dim=len(config["proprio_names"]),
            action_dim=config["action_dim"],
            obs_horizon=config["obs_horizon"],
            pred_horizon=config["pred_horizon"],
            n_inference_steps=config["n_inference_steps"],
            sigma_data=config["sigma_data"],
            sigma_min=config["sigma_min"],
            sigma_max=config["sigma_max"],
            rho=config["rho"],
            p_mean=config["p_mean"],
            p_std=config["p_std"],
            feature_dim=config["feature_dim"],
        )
        policy.load_state_dict(blob["model"])
        policy.to(device)
        return policy, config

    def _denoise(self, noisy: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        sigma = noisy.new_full((noisy.shape[0], 1, 1), float(sigma))
        c_skip, c_out, c_in, c_noise = self._edm_coeffs(sigma)
        return c_skip * noisy + c_out * self.unet(c_in * noisy, c_noise, cond)

    def _edm_coeffs(self, sigma: torch.Tensor):
        """Preconditioning from the EDM paper. ``sigma`` is (B, 1, 1)."""
        sd = self.sigma_data.to(dtype=sigma.dtype)
        variance = sigma ** 2 + sd ** 2
        c_skip = sd ** 2 / variance
        c_out = sigma * sd / variance.sqrt()
        c_in = variance.rsqrt()
        c_noise = sigma.flatten().log() / 4
        return c_skip, c_out, c_in, c_noise

    def _sigma_schedule(self, device, dtype) -> torch.Tensor:
        """ρ-schedule from σ_max down to σ_min, then a final step at 0."""
        n = self.n_inference_steps
        ramp = torch.arange(n, device=device, dtype=dtype) / (n - 1)
        inv = 1.0 / self.rho
        sigmas = (self.sigma_max ** inv + ramp * (self.sigma_min ** inv - self.sigma_max ** inv)) ** self.rho
        return torch.cat((sigmas, sigmas.new_zeros(1)))

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_low) / self.action_span * 2 - 1

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action + 1) / 2 * self.action_span + self.action_low
