"""Train a diffusion policy on a pick-and-place dataset.

    python -m policy.diffusion.train \\
        --dataset simulation/tasks/pick_place_cube/data/pick_place_200 \\
        --out policy/runs/pick_place_200/best.pt

The checkpoint's ``sample(images, state, noise=...)`` is the hook DSRL will call.
``noise`` is standard normal with shape (batch, pred_horizon, action_dim) and fully
determines the chunk. The chunk is absolute joint and gripper targets, one per policy
step of ``frame_skip`` control ticks; ``actions.ChunkFollower`` turns it into task actions.
The trainer fits the EDM denoising objective.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from policy.diffusion.dataset import PROPRIO_NAMES, EpisodeWindows, split_episodes
from policy.diffusion.policy import DiffusionPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("policy/runs/pick_place_200/best.pt"))
    parser.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--frame-skip", type=int, default=5, help="Control ticks per policy step (50 Hz / 5 = 10 Hz)")
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=16, help="Policy steps per predicted chunk")
    parser.add_argument("--execute-steps", type=int, default=8, help="Policy steps run before replanning")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--val-episodes", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=5, help="Roll out and log videos every N epochs. 0 skips it.")
    parser.add_argument("--save-every", type=int, default=10, help="Save an epoch checkpoint this often, besides best.pt. 0 skips it.")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--eval-max-steps", type=int, default=1000, help="Control ticks per logged rollout")
    parser.add_argument("--no-wandb", action="store_true", help="Skip Weights & Biases. Videos still go under the run folder.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids, val_ids = split_episodes(args.dataset, args.val_episodes)
    print(f"Loading videos on {device.type}. Train episodes {train_ids[0]}-{train_ids[-1]}, val {val_ids}.")
    train_set = EpisodeWindows(
        args.dataset, args.cameras, args.image_size, args.obs_horizon, args.pred_horizon, train_ids, args.frame_skip
    )
    val_set = EpisodeWindows(
        args.dataset, args.cameras, args.image_size, args.obs_horizon, args.pred_horizon, val_ids, args.frame_skip
    )
    print(f"{len(train_set)} train windows, {len(val_set)} val windows, cameras {args.cameras}.")

    action_dim = train_set.episodes[0]["action"].shape[-1]
    policy = DiffusionPolicy(
        n_cameras=len(args.cameras),
        state_dim=len(PROPRIO_NAMES),
        action_dim=action_dim,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
    ).to(device)
    state = np.concatenate([episode["state"] for episode in train_set.episodes])
    action = np.concatenate([episode["action"][: episode["length"]] for episode in train_set.episodes])
    policy.set_normalizer(state.min(0), state.max(0), action.min(0), action.max(0), action)
    print(f"Normalized action std (sigma_data) {float(policy.sigma_data):.3f}.")
    config = {
        "cameras": args.cameras,
        "image_size": args.image_size,
        "obs_horizon": args.obs_horizon,
        "pred_horizon": args.pred_horizon,
        "frame_skip": args.frame_skip,
        "execute_steps": args.execute_steps,
        "action_type": "joint_target",
        "n_inference_steps": policy.n_inference_steps,
        "sigma_data": float(policy.sigma_data),
        "sigma_min": policy.sigma_min,
        "sigma_max": policy.sigma_max,
        "rho": policy.rho,
        "p_mean": policy.p_mean,
        "p_std": policy.p_std,
        "feature_dim": policy.feature_dim,
        "proprio_names": list(PROPRIO_NAMES),
        "action_dim": action_dim,
        "dataset": str(args.dataset),
    }

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _lr_scale(step, total_steps))
    # The averaged weights are what get evaluated, saved, and later steered by DSRL.
    ema = AveragedModel(policy, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True)
    best = float("inf")
    run = _start_wandb(args, config)
    try:
        for epoch in range(1, args.epochs + 1):
            policy.train()
            total, count = 0.0, 0
            for batch in train_loader:
                images, state_b, action_b = _batch(batch, device)
                loss = policy.loss(_random_shift(images), state_b, action_b)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                ema.update_parameters(policy)
                total += float(loss) * images.shape[0]
                count += images.shape[0]
            val = _evaluate(ema.module, val_loader, device)
            print(f"epoch {epoch:02d}  train {total / count:.4f}  val (EMA) {val:.4f}", flush=True)
            if val < best:
                best = val
                ema.module.save(args.out, config)
                _write_config(args.out, config)
                _log_checkpoint(run, args.out, epoch, val, ["best"])
            if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
                snapshot = args.out.parent / "epochs" / f"epoch_{epoch:04d}.pt"
                ema.module.save(snapshot, config)
                _log_checkpoint(run, snapshot, epoch, val, [f"epoch-{epoch}"])
            if run is not None:
                run.log(
                    {
                        "train/loss": total / count,
                        "val/loss": val,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=epoch,
                )
            if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
                _log_rollouts(
                    run, ema.module, config, epoch, args.out.parent, args.eval_seeds, args.eval_max_steps, device
                )
        _write_config(args.out, config)
        if run is not None:
            run.summary["best_val_loss"] = best
        print(f"Best val loss {best:.4f}. Wrote {args.out.resolve()}")
    finally:
        if run is not None:
            run.finish()


def _write_config(checkpoint: Path, config: dict) -> None:
    (checkpoint.parent / "config.json").write_text(json.dumps(config, indent=2))


def _start_wandb(args, config):
    """Online when WANDB_API_KEY is set, offline otherwise. Logs stay next to the checkpoint."""
    if args.no_wandb:
        print("W&B disabled (--no-wandb).", flush=True)
        return None
    import wandb

    key = os.environ.get("WANDB_API_KEY", "").strip()
    mode = os.environ.get("WANDB_MODE") or ("online" if key else "offline")
    run_id = os.environ.get("WANDB_RUN_ID", "").strip() or None
    entity = os.environ.get("WANDB_ENTITY", "").strip() or None
    log_dir = args.out.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    init = {
        "project": os.environ.get("WANDB_PROJECT", "lakshya-pick-place"),
        "entity": entity,
        "name": os.environ.get("WANDB_NAME") or log_dir.name,
        "dir": str(log_dir),
        "mode": mode,
        "config": config,
        "job_type": "train",
    }
    if run_id:
        init["id"] = run_id
        init["resume"] = "allow"
    run = wandb.init(**init)
    if mode == "offline":
        print(
            "W&B offline (no WANDB_API_KEY). Charts and videos are under "
            f"{log_dir / 'wandb'}. Add the key to wandb.env and sync that folder.",
            flush=True,
        )
    else:
        print(f"W&B run: {run.url}", flush=True)
    return run


def _log_checkpoint(run, path: Path, epoch: int, val: float, aliases: list[str]) -> None:
    if run is None:
        return
    import wandb

    run_name = path.parent.name if path.name == "best.pt" else path.parent.parent.name
    name = "policy-" + re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-")
    artifact = wandb.Artifact(name or "policy", type="model", metadata={"epoch": epoch, "val_loss": val})
    artifact.add_file(str(path), name=path.name)
    config_path = path.parent / "config.json"
    if config_path.is_file():
        artifact.add_file(str(config_path), name="config.json")
    run.log_artifact(artifact, aliases=aliases)


def _log_rollouts(run, policy, config, epoch: int, out_dir: Path, seeds: list[int], max_steps: int, device) -> None:
    import wandb

    from policy.diffusion.rollout import EpisodeRunner
    from tasks.pick_place_cube.scene import build_model
    from xarm_sim.paths import resolve_scene
    from xarm_sim.sim import XArmSim

    policy.eval()
    sim = XArmSim(model=build_model(resolve_scene("default"), 1, ""))
    runner = EpisodeRunner(sim, sim.arm(""), policy, config, int(config["execute_steps"]), max_steps, device)
    fps = max(1, int(round(1.0 / (0.02 * int(config["frame_skip"])))))
    logs: dict = {}
    successes: list[float] = []
    try:
        for seed in seeds:
            video = out_dir / "videos" / f"epoch_{epoch:04d}_seed_{seed}.mp4"
            result = runner.run(seed, video_path=video)
            successes.append(float(result["success"]))
            print(f"rollout epoch {epoch} seed {seed}: {result}", flush=True)
            logs[f"rollout/seed_{seed}/success"] = float(result["success"])
            logs[f"rollout/seed_{seed}/distance_m"] = result["distance_m"]
            logs[f"rollout/seed_{seed}/closest_xy_m"] = result["closest_xy_m"]
            logs[f"rollout/seed_{seed}/max_lift_m"] = result["max_lift_m"]
            if run is not None and video.is_file():
                logs[f"rollout/video_seed_{seed}"] = wandb.Video(
                    str(video), fps=fps, format="mp4", caption=f"epoch {epoch}: {' | '.join(config['cameras'])}"
                )
    finally:
        runner.close()
    if not successes:
        return
    logs["rollout/success_rate"] = sum(successes) / len(successes)
    print(f"rollout epoch {epoch}: {int(sum(successes))}/{len(successes)} succeeded", flush=True)
    if run is not None:
        run.log(logs, step=epoch)


def _batch(batch, device):
    images = batch["images"].to(device, non_blocking=True)
    state = batch["state"].to(device, non_blocking=True)
    action = batch["action"].to(device, non_blocking=True)
    return images, state, action


def _lr_scale(step: int, total_steps: int, warmup: int = 500) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _random_shift(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Shift each sample's frames by up to ``pad`` pixels, the same shift for all its cameras."""
    batch, obs, cameras, channels, height, width = images.shape
    flat = F.pad(images.reshape(batch, -1, height, width), (pad, pad, pad, pad), mode="replicate")
    dy = torch.randint(0, 2 * pad + 1, (batch, 1), device=images.device)
    dx = torch.randint(0, 2 * pad + 1, (batch, 1), device=images.device)
    rows = (dy + torch.arange(height, device=images.device))[:, :, None]
    cols = (dx + torch.arange(width, device=images.device))[:, None, :]
    shifted = flat[torch.arange(batch, device=images.device)[:, None, None], :, rows, cols]
    return shifted.permute(0, 3, 1, 2).reshape(images.shape)


@torch.no_grad()
def _evaluate(policy: DiffusionPolicy, loader: DataLoader, device) -> float:
    policy.eval()
    generator = torch.Generator(device=device).manual_seed(0)
    total, count = 0.0, 0
    for batch in loader:
        images, state, action = _batch(batch, device)
        total += float(policy.loss(images, state, action, generator=generator)) * images.shape[0]
        count += images.shape[0]
    return total / max(count, 1)


if __name__ == "__main__":
    main()
