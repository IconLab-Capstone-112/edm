"""Watch the diffusion policy pick and place in the MuJoCo window, or score it headless.

    python -m policy.diffusion.rollout --seed 0
    MUJOCO_GL=egl python -m policy.diffusion.rollout --headless --seeds 0 1 2 3 4

The scene, cube placement, and task actions are the same ones the dataset was
collected with. Every ``frame_skip`` ticks the policy sees the four cameras and the arm
state; after ``execute_steps`` policy steps it plans a new chunk of joint targets.
``ChunkFollower`` converts those targets into the task's per-tick actions.
Close the window to stop.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

from policy.diffusion.actions import ChunkFollower
from policy.diffusion.policy import DiffusionPolicy
from tasks.pick_place_cube.controller import PickPlaceController
from tasks.pick_place_cube.scene import GOAL_RADIUS, apply_action, body_pose, build_model, cube_name, set_cube
from xarm_sim.paths import resolve_scene
from xarm_sim.recording import render_camera, scene_cameras
from xarm_sim.sim import XArmSim

CONTROL_DT = 0.02
MAX_STEPS = 1000
HOLD_TICKS = 15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("policy/runs/pick_place_200/best.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", help="Headless: seeds to score")
    parser.add_argument("--headless", action="store_true", help="No window; print one result per seed")
    parser.add_argument("--execute-steps", type=int, default=0, help="Policy steps per plan. 0 uses the checkpoint's value.")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Control ticks per episode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, config = DiffusionPolicy.load(args.checkpoint, device=device)
    policy.eval()
    if config.get("action_type") != "joint_target":
        raise SystemExit(f"{args.checkpoint} predicts joint deltas. Retrain with the current trainer.")
    execute_steps = args.execute_steps or int(config["execute_steps"])

    sim = XArmSim(model=build_model(resolve_scene("default"), 1, ""))
    arm = sim.arm("")
    runner = EpisodeRunner(sim, arm, policy, config, execute_steps, args.max_steps, device)

    if args.headless:
        results = [runner.run(seed) for seed in (args.seeds or [args.seed])]
        for result in results:
            print(result, flush=True)
        print(f"{sum(r['success'] for r in results)}/{len(results)} succeeded")
        runner.close()
        return

    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    try:
        viewer.cam.lookat[:] = arm.to_world((0.30, -0.28, 0.10))
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 135
        print("Playing the policy. Close the window to stop.")
        if not viewer.is_running():
            return
        print(runner.run(args.seed, viewer))
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    finally:
        viewer.close()
        runner.close()


class EpisodeRunner:
    def __init__(self, sim, arm, policy, config, execute_steps, max_steps, device):
        self.sim, self.arm, self.policy, self.device = sim, arm, policy, device
        self.controller = PickPlaceController(sim, arm, 1, False, 0.0, CONTROL_DT, max_steps)
        self.physics_steps = max(1, int(round(CONTROL_DT / sim.timestep)))
        self.camera_names = list(config["cameras"])
        self.cameras = _camera_ids(sim.model, self.camera_names)
        self.renderer = mujoco.Renderer(sim.model, height=256, width=256)
        self.image_size = int(config["image_size"])
        self.frame_skip = int(config["frame_skip"])
        self.execute_steps = execute_steps
        self.max_steps = max_steps

    def close(self) -> None:
        self.renderer.close()

    def run(self, seed: int, viewer=None, video_path: Path | None = None) -> dict:
        sim, arm = self.sim, self.arm
        lock = viewer.lock if viewer is not None else contextlib.nullcontext
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        writer = _video_writer(video_path, self.frame_skip)
        try:
            with lock():
                sim.reset()
                cube_start, cube_yaws, targets, _ = self.controller._sample_layout(rng)
                set_cube(sim, 0, cube_start[0], cube_yaws[0])
                mujoco.mj_forward(sim.model, sim.data)
                sim.step(100)

            images: list[np.ndarray] = []
            states: list[np.ndarray] = []
            follower = ChunkFollower(self.frame_skip)
            held, closest, lift = 0, np.inf, 0.0
            distance = float(np.linalg.norm(cube_start[0] - targets[0]))
            tick = 0
            for tick in range(self.max_steps):
                if viewer is not None and not viewer.is_running():
                    break
                if tick % self.frame_skip == 0:
                    image, state, frames = self._observe()
                    if writer is not None:
                        writer.append_data(_camera_grid(frames, self.camera_names))
                    images = (images + [image])[-self.policy.obs_horizon :]
                    states = (states + [state])[-self.policy.obs_horizon :]
                    if not len(follower):
                        chunk = self._plan(images, states)[: self.execute_steps]
                        follower.load(chunk, arm.qpos(), arm.gripper_qpos())
                start = time.perf_counter()
                with lock():
                    apply_action(sim, arm, follower.action(arm.qpos(), arm.gripper_qpos()), self.physics_steps)
                cube = body_pose(sim, cube_name(0))[0]
                closest = min(closest, float(np.linalg.norm(arm.tcp_pose()[0][:2] - cube[:2])))
                lift = max(lift, float(cube[2] - cube_start[0][2]))
                distance = float(np.linalg.norm(cube - targets[0]))
                held = held + 1 if distance <= GOAL_RADIUS else 0
                if viewer is not None:
                    viewer.sync()
                    delay = CONTROL_DT - (time.perf_counter() - start)
                    if delay > 0:
                        time.sleep(delay)
                if held >= HOLD_TICKS:
                    break
            return {
                "seed": seed,
                "steps": tick + 1,
                "success": held >= HOLD_TICKS or distance <= GOAL_RADIUS,
                "distance_m": round(distance, 4),
                "closest_xy_m": round(closest, 4),
                "max_lift_m": round(lift, 4),
                "video": "" if video_path is None else str(video_path),
            }
        finally:
            if writer is not None:
                writer.close()

    def _observe(self) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        frames = []
        resized = []
        for camera in self.cameras:
            frame = render_camera(self.renderer, self.sim.data, camera)
            frames.append(frame)
            image = Image.fromarray(frame).resize((self.image_size, self.image_size), Image.BILINEAR)
            resized.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        ee_pos, _ = self.arm.tcp_pose()
        state = np.concatenate([self.arm.qpos(), [self.arm.gripper_qpos()], ee_pos]).astype(np.float32)
        return np.stack(resized), state, frames

    def _plan(self, images: list[np.ndarray], states: list[np.ndarray]) -> np.ndarray:
        horizon = self.policy.obs_horizon
        images = images[:1] * (horizon - len(images)) + images
        states = states[:1] * (horizon - len(states)) + states
        image_t = torch.as_tensor(np.stack(images), device=self.device)[None]
        state_t = torch.as_tensor(np.stack(states), device=self.device)[None]
        return self.policy.sample(image_t, state_t)[0].cpu().numpy()


def _video_writer(path: Path | None, frame_skip: int):
    if path is None:
        return None
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(round(1.0 / (CONTROL_DT * frame_skip))))
    return imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1)


def _camera_grid(frames: list[np.ndarray], names: list[str]) -> np.ndarray:
    """One frame: top row is the first pair of cameras, bottom row the rest."""
    from PIL import ImageDraw

    labeled = []
    for frame, name in zip(frames, names):
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 8 * len(name) + 10, 18), fill=(0, 0, 0))
        draw.text((4, 2), name, fill=(255, 255, 255))
        labeled.append(np.asarray(image))
    if len(labeled) == 4:
        top = np.concatenate(labeled[:2], axis=1)
        bottom = np.concatenate(labeled[2:], axis=1)
        return np.concatenate([top, bottom], axis=0)
    return np.concatenate(labeled, axis=1)


def _camera_ids(model, names: list[str]) -> list[str]:
    declared = scene_cameras(model)
    missing = [name for name in names if name not in declared]
    if missing:
        raise SystemExit(f"Scene is missing cameras {missing}")
    return [declared[name] for name in names]


if __name__ == "__main__":
    main()
