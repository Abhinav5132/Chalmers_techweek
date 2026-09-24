"""
Reference Motion Tracker & Trajectory Interpolator for Unitree G1.
Interpolates discrete MoCap reference clips (.npz) into continuous high-frequency trajectories.
Fully typed for static type checkers (ty / mypy / pyright).
"""

from __future__ import annotations

import os
from typing import Any
import numpy as np


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Spherical Linear Interpolation (SLERP) between two unit quaternions [w, x, y, z].
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = float(np.clip(dot, -1.0, 1.0))

    # If quaternions are very close, use linear interpolation to avoid division by zero
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


class MotionTracker:
    """
    Loads pre-retargeted G1 motion clips (.npz) and evaluates smooth continuous
    reference trajectories for whole-body, lower-body, and upper-body joints at arbitrary rates.
    """

    # G1 29 Joint partitioning
    LOWER_BODY_INDICES: list[int] = list(range(0, 15))   # 12 leg joints + 3 waist joints
    UPPER_BODY_INDICES: list[int] = list(range(15, 29))  # 7 left arm joints + 7 right arm joints

    def __init__(self, clip_path: str) -> None:
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Motion clip not found at '{clip_path}'.")

        data: Any = np.load(clip_path)
        required_keys = ["fps", "joint_pos", "body_pos_w", "body_quat_w"]
        for k in required_keys:
            if k not in data:
                raise ValueError(f"Motion clip '{clip_path}' is missing required key: {k}")

        self.clip_path: str = clip_path
        self.fps: float = float(data["fps"][0]) if data["fps"].ndim > 0 else float(data["fps"])
        self.dt_frame: float = 1.0 / self.fps

        # Joint positions [N, 29]
        self.joint_pos: np.ndarray = np.asarray(data["joint_pos"], dtype=np.float64)
        self.n_frames: int = self.joint_pos.shape[0]
        self.duration: float = float(max(0.0, (self.n_frames - 1) * self.dt_frame))

        # Root position [N, 3] and quaternion [N, 4] (body 0 is pelvis)
        self.root_pos: np.ndarray = np.asarray(data["body_pos_w"][:, 0, :], dtype=np.float64)
        self.root_quat: np.ndarray = np.asarray(data["body_quat_w"][:, 0, :], dtype=np.float64)

    def get_state(
        self, t: float, loop: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate full motion state at continuous timestamp t (seconds).
        Returns:
            q_joints: [29] joint angles (rad)
            root_pos: [3] pelvis Cartesian XYZ (m)
            root_quat: [4] pelvis quaternion [w, x, y, z]
        """
        if self.duration <= 0.0 or self.n_frames <= 1:
            return (
                self.joint_pos[0].copy(),
                self.root_pos[0].copy(),
                self.root_quat[0].copy(),
            )

        if loop:
            t = t % self.duration
        else:
            t = float(np.clip(t, 0.0, self.duration))

        frame_exact = t / self.dt_frame
        idx0 = int(np.floor(frame_exact))
        idx1 = min(idx0 + 1, self.n_frames - 1)
        alpha = float(frame_exact - idx0)

        # Linear interpolation for joint angles
        q_joints = (1.0 - alpha) * self.joint_pos[idx0] + alpha * self.joint_pos[idx1]

        # Linear interpolation for root translation
        root_pos = (1.0 - alpha) * self.root_pos[idx0] + alpha * self.root_pos[idx1]

        # SLERP for root orientation quaternion
        root_quat = slerp(self.root_quat[idx0], self.root_quat[idx1], alpha)

        return q_joints, root_pos, root_quat

    def get_lower_body_target(self, t: float, loop: bool = True) -> np.ndarray:
        """Returns the 15 lower-body and waist joint angles at time t."""
        q_full, _, _ = self.get_state(t, loop=loop)
        return q_full[self.LOWER_BODY_INDICES].copy()

    def get_upper_body_target(self, t: float, loop: bool = True) -> np.ndarray:
        """Returns the 14 upper-body (arms and wrists) joint angles at time t."""
        q_full, _, _ = self.get_state(t, loop=loop)
        return q_full[self.UPPER_BODY_INDICES].copy()
