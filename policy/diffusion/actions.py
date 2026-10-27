"""Absolute joint targets at the policy rate, and the per-tick deltas the task expects.

The task's action is a per-tick delta: the arm aims at ``qpos + action * JOINT_ACTION_DELTA``.
Deltas compound, so a small error early in an open-loop chunk sends the rest of it off course.
The policy instead predicts where the joints and gripper should be at the end of each policy
step (``frame_skip`` control ticks). ``ChunkFollower`` turns that back into task actions every
tick from the measured joints, so the arm corrects toward the plan instead of drifting.
"""

from __future__ import annotations

import numpy as np

from tasks.pick_place_cube.scene import JOINT_ACTION_DELTA

JOINTS = 7


def demo_targets(joint_pos: np.ndarray, gripper: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Per-tick absolute targets the demo commanded. Shapes (T, 7), (T,), (T, 8) -> (T, 8)."""
    delta = np.clip(actions, -1.0, 1.0) * JOINT_ACTION_DELTA
    targets = np.empty_like(actions)
    targets[:, :JOINTS] = joint_pos + delta[:, :JOINTS]
    targets[:, JOINTS] = np.clip(gripper + delta[:, JOINTS], 0.0, 1.0)
    return targets


def policy_labels(targets: np.ndarray, frame_skip: int) -> np.ndarray:
    """One label per policy step: the target commanded on that step's last tick."""
    ticks = np.minimum(np.arange(0, len(targets), frame_skip) + frame_skip - 1, len(targets) - 1)
    return targets[ticks]


def task_action(target: np.ndarray, joint_pos: np.ndarray, gripper: float) -> np.ndarray:
    """The task action that moves from the measured state toward ``target`` this tick."""
    measured = np.append(joint_pos, gripper)
    return np.clip((target - measured) / JOINT_ACTION_DELTA, -1.0, 1.0).astype(np.float32)


class ChunkFollower:
    """Plays a chunk of policy-step targets as ``frame_skip`` task actions per step.

    Joint targets are interpolated from the previous step's target, so the arm moves smoothly
    within a step. The gripper target is used as is, so a close command is not delayed.
    """

    def __init__(self, frame_skip: int):
        self.frame_skip = frame_skip
        self.targets: list[np.ndarray] = []
        self.previous: np.ndarray | None = None
        self.tick = 0

    def __len__(self) -> int:
        """Policy steps still queued."""
        return len(self.targets)

    def load(self, targets: np.ndarray, joint_pos: np.ndarray, gripper: float) -> None:
        self.targets = [t for t in np.asarray(targets, dtype=np.float64)]
        if self.previous is None:
            self.previous = np.append(joint_pos, gripper)
        self.tick = 0

    def action(self, joint_pos: np.ndarray, gripper: float) -> np.ndarray:
        goal = self.targets[0]
        fraction = (self.tick + 1) / self.frame_skip
        command = goal.copy()
        command[:JOINTS] = self.previous[:JOINTS] + fraction * (goal[:JOINTS] - self.previous[:JOINTS])
        self.tick += 1
        if self.tick == self.frame_skip:
            self.previous = self.targets.pop(0)
            self.tick = 0
        return task_action(command, joint_pos, gripper)
