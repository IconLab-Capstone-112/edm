"""Windows of images, proprioception, and target chunks from a collected dataset.

The folder is the one ``xarm-run --save_path`` writes next to the npz: mp4s per camera
and one JSON of states and actions per episode. Cube pose is recorded in that state
vector and is left out of the policy input, so the network has to see the cube in
the cameras. Joint velocity is left out too; it is noisy and not part of the command.

The demos run at 50 Hz. The policy runs at 50 / ``frame_skip`` Hz: it sees every
``frame_skip``-th frame, and each label is the absolute joint and gripper target at the
end of that step (see ``actions.py``). Keeping only those frames also cuts memory by
``frame_skip``, which is what lets a few hundred episodes fit in RAM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .actions import JOINTS, demo_targets, policy_labels

PROPRIO_NAMES = (
    "joint_pos_0",
    "joint_pos_1",
    "joint_pos_2",
    "joint_pos_3",
    "joint_pos_4",
    "joint_pos_5",
    "joint_pos_6",
    "gripper",
    "ee_pos_x",
    "ee_pos_y",
    "ee_pos_z",
)


class EpisodeWindows(Dataset):
    """One index is one policy step i: ``obs_horizon`` frames ending at i, and the targets that follow."""

    def __init__(
        self,
        root: Path,
        cameras: list[str],
        image_size: int,
        obs_horizon: int,
        pred_horizon: int,
        episode_indices: list[int] | None = None,
        frame_skip: int = 5,
    ):
        root = Path(root)
        info = json.loads((root / "meta" / "info.json").read_text())
        state_names = info["features"]["observation.state"]["names"]
        missing = [name for name in PROPRIO_NAMES if name not in state_names]
        if missing:
            raise SystemExit(f"Dataset state is missing {missing}")
        self.proprio_index = [state_names.index(name) for name in PROPRIO_NAMES]
        self.cameras = list(cameras)
        self.image_size = image_size
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.frame_skip = frame_skip

        entries = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line]
        if episode_indices is None:
            episode_indices = list(range(len(entries)))
        self.episodes = []
        for index in episode_indices:
            self.episodes.append(self._load_episode(root, entries[index]))
            print(f"  episode {entries[index]['seed']}  {self.episodes[-1]['length']} steps", flush=True)

        # Chunks that run past the end repeat the last target: hold still where the demo stopped.
        self.index: list[tuple[int, int]] = []
        for episode_i, episode in enumerate(self.episodes):
            length = episode["length"]
            pad = np.repeat(episode["action"][-1:], pred_horizon - 1, axis=0)
            episode["action"] = np.concatenate((episode["action"], pad))
            self.index.extend((episode_i, t) for t in range(length))

    def _load_episode(self, root: Path, entry: dict) -> dict:
        path = root / "data" / f"episode_{entry['seed']:06d}.json"
        frames = json.loads(path.read_text())["frames"]
        state = np.asarray([frame["observation.state"] for frame in frames], dtype=np.float32)
        action = np.asarray([frame["action"] for frame in frames], dtype=np.float32)
        clips = [_read_clip(root / entry["videos"][camera], self.image_size, self.frame_skip) for camera in self.cameras]
        proprio = state[:, self.proprio_index]
        targets = demo_targets(proprio[:, :JOINTS], proprio[:, JOINTS], action)
        labels = policy_labels(targets, self.frame_skip)
        n = min(len(labels), *(len(clip) for clip in clips))
        # (steps, cameras, 3, H, W) uint8. Floats would take four times the memory.
        images = np.stack([clip[:n] for clip in clips], axis=1)
        return {"state": proprio[:: self.frame_skip][:n], "action": labels[:n], "images": images, "length": n}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        episode_i, t = self.index[i]
        episode = self.episodes[episode_i]
        obs_at = [max(0, t - self.obs_horizon + 1 + k) for k in range(self.obs_horizon)]
        return {
            "images": episode["images"][obs_at].astype(np.float32) / 255.0,
            "state": episode["state"][obs_at],
            "action": episode["action"][t : t + self.pred_horizon],
        }


def _read_clip(path: Path, size: int, frame_skip: int) -> np.ndarray:
    import imageio.v2 as imageio

    frames = []
    reader = imageio.get_reader(path)
    try:
        for index, frame in enumerate(reader):
            if index % frame_skip:
                continue
            image = Image.fromarray(frame).resize((size, size), Image.BILINEAR)
            frames.append(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1))
    finally:
        reader.close()
    return np.stack(frames)


def split_episodes(root: Path, val_episodes: int) -> tuple[list[int], list[int]]:
    """Successful episodes only; the last ``val_episodes`` of them are held out."""
    entries = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    ok = [index for index, entry in enumerate(entries) if entry.get("success", True)]
    if len(ok) < len(entries):
        print(f"Skipping {len(entries) - len(ok)} failed demos.")
    if val_episodes <= 0 or val_episodes >= len(ok):
        raise SystemExit(f"--val-episodes must be between 1 and {len(ok) - 1}")
    return ok[:-val_episodes], ok[-val_episodes:]
