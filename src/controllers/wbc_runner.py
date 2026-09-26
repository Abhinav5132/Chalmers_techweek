"""
WBC tracking runtime for Unitree G1 in plain MuJoCo (CPU) using an exported
wbc-mjlab policy bundle (models/params/policy.onnx + models/params/config.yaml).

Deploy-style runtime that mirrors the wbc-g1-deploy reference: it reconstructs
the exact actor observation vector declared by the exported config.yaml (schema
``wbc_tracking_params_v1``), runs ONNX inference, and applies residual
joint-position targets via the PD gains from the same file.

It hard-fails when the policy bundle is missing. There is deliberately no
fallback: PD-following a reference clip cannot balance a free-floating base.
The exported phase-2 bundle is included in models/params/. See models/README.md
for its source and the matching scene requirement.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import mujoco
import onnxruntime as ort
import yaml


mj: Any = mujoco

# Actor observation terms this runtime implements. Other layouts (SE/EE) must be
# added before they can run here; unknown terms raise instead of guessing.
_SUPPORTED_OBS_TERMS = frozenset(
    {
        "ref_base_height",
        "ref_base_lin_vel_b",
        "ref_base_ang_vel_b",
        "ref_gravity_b",
        "ref_joint_pos",
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
    }
)

# Unitree motor torque-speed catalog (x1,x2 rad/s corners; y1,y2 Nm peaks) and
# friction (fs,fd,va) per joint group, matching wbc-mjlab robots/g1/actuators.py.
_ENVELOPE: dict[str, tuple[float, float, float, float, float, float, float, float]] = {
    "knee": (14.5, 22.7, 111.0, 131.0, 2.4, 0.24, 0.01, 1.0),
    "hip_yaw": (22.63, 35.52, 71.0, 83.3, 1.6, 0.16, 0.01, 1.0),
    "hip_pitch": (14.5, 22.7, 111.0, 131.0, 2.4, 0.24, 0.01, 1.0),
    "hip_roll": (14.5, 22.7, 111.0, 131.0, 2.4, 0.24, 0.01, 1.0),
    "shoulder": (30.86, 40.13, 24.8, 31.9, 0.6, 0.06, 0.01, 1.0),
    "elbow": (30.86, 40.13, 24.8, 31.9, 0.6, 0.06, 0.01, 1.0),
    "wrist_roll": (30.86, 40.13, 24.8, 31.9, 0.6, 0.06, 0.01, 1.0),
    "wrist_pitch": (15.3, 24.76, 4.8, 8.6, 0.6, 0.06, 0.01, 1.0),
    "wrist_yaw": (15.3, 24.76, 4.8, 8.6, 0.6, 0.06, 0.01, 1.0),
    "waist_yaw": (22.63, 35.52, 71.0, 83.3, 1.6, 0.16, 0.01, 1.0),
    "waist_pitch": (30.86, 40.13, 49.6, 63.8, 1.2, 0.12, 0.01, 2.0),
    "waist_roll": (30.86, 40.13, 49.6, 63.8, 1.2, 0.12, 0.01, 2.0),
    "ankle_pitch": (30.86, 40.13, 49.6, 63.8, 1.2, 0.12, 0.01, 2.0),
    "ankle_roll": (30.86, 40.13, 49.6, 63.8, 1.2, 0.12, 0.01, 2.0),
}


def _envelope_for(joint: str) -> tuple[float, ...]:
    for key, params in _ENVELOPE.items():
        if key in joint:
            return params
    return _ENVELOPE["elbow"]


class ClipReference:
    """Anchored motion clip: reference joints + anchor-frame body state.

    Clips are normally 50 Hz (matching the policy rate). If a clip's recorded
    fps differs, it is resampled on load so reference playback advances one
    frame per policy step at the correct speed.
    """

    def __init__(self, clip_path: str, target_fps: float | None = None, speed: float = 1.0) -> None:
        if not np.isfinite(speed) or not .25 <= speed <= 2:
            raise ValueError("Tracking speed must be 0.25–2×.")
        with np.load(clip_path, allow_pickle=False) as data:
            required = ("fps", "joint_pos", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
            for key in required:
                if key not in data:
                    raise ValueError(f"Physics tracking requires {key} in {Path(clip_path).name}.")
                if not np.isfinite(data[key]).all():
                    raise ValueError(f"Non-finite {key} in {Path(clip_path).name}.")
            fps = np.asarray(data["fps"]).reshape(-1)
            if fps.size != 1 or fps[0] <= 0:
                raise ValueError("Invalid recording frame rate.")
            self.fps = float(fps[0]) * speed
            self.joint_pos = np.array(data["joint_pos"], dtype=np.float64)
            self.body_pos_w = np.array(data["body_pos_w"], dtype=np.float64)
            self.body_quat_w = np.array(data["body_quat_w"], dtype=np.float64)
            self.body_lin_vel_w = np.array(data["body_lin_vel_w"], dtype=np.float64) * speed
            self.body_ang_vel_w = np.array(data["body_ang_vel_w"], dtype=np.float64) * speed
            self.joint_vel = np.array(data["joint_vel"], dtype=np.float64) * speed if "joint_vel" in data else np.zeros_like(self.joint_pos)
        self.n_frames = len(self.joint_pos)
        if self.joint_pos.shape != (self.n_frames, 29) or self.n_frames < 1:
            raise ValueError("Physics tracking requires nonempty 29-joint recordings.")
        if (self.body_pos_w.shape != (self.n_frames, 30, 3) or
                self.body_quat_w.shape != (self.n_frames, 30, 4) or
                self.body_lin_vel_w.shape != self.body_pos_w.shape or
                self.body_ang_vel_w.shape != self.body_pos_w.shape or
                self.joint_vel.shape != self.joint_pos.shape or not np.isfinite(self.joint_vel).all()):
            raise ValueError("Physics tracking requires the G1 30-body recording layout with valid velocities.")
        norms = np.linalg.norm(self.body_quat_w, axis=-1, keepdims=True)
        if np.any(norms < 1e-6):
            raise ValueError("Recording contains a zero quaternion.")
        self.body_quat_w /= norms
        if target_fps is not None and self.fps != target_fps:
            self._resample(target_fps)

    def _resample(self, target_fps: float) -> None:
        """Linearly/slerp-interpolate the clip to a new frame rate."""
        from controllers.clip_resample import resample_clip

        out = resample_clip(
            {
                "joint_pos": self.joint_pos,
                "joint_vel": self.joint_vel,
                "body_pos_w": self.body_pos_w,
                "body_quat_w": self.body_quat_w,
                "body_lin_vel_w": self.body_lin_vel_w,
                "body_ang_vel_w": self.body_ang_vel_w,
            },
            self.fps,
            target_fps,
        )
        self.joint_pos = out["joint_pos"]
        self.joint_vel = out["joint_vel"]
        self.body_pos_w = out["body_pos_w"]
        self.body_quat_w = out["body_quat_w"]
        self.body_lin_vel_w = out["body_lin_vel_w"]
        self.body_ang_vel_w = out["body_ang_vel_w"]
        self.n_frames = self.joint_pos.shape[0]
        self.fps = target_fps

    def anchor_frame(self, idx: int, anchor_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (pos, quat_wxyz, lin_vel_w, ang_vel_w) of the anchor body at frame idx."""
        i = min(int(idx), self.n_frames - 1)
        # NPZ quats are stored wxyz (mjlab/MuJoCo convention); MuJoCo expects wxyz.
        quat_wxyz = np.asarray(self.body_quat_w[i, anchor_idx], dtype=np.float64)
        return (
            self.body_pos_w[i, anchor_idx].copy(),
            quat_wxyz,
            self.body_lin_vel_w[i, anchor_idx].copy(),
            self.body_ang_vel_w[i, anchor_idx].copy(),
        )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    out[1:4] *= -1.0
    return out


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_yaw(q: np.ndarray) -> np.ndarray:
    """Yaw-only wxyz quaternion from a full wxyz quaternion (identity if degenerate)."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    w, x, y, z = q / norm
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)


def _rotated_vel(vel: np.ndarray, q: np.ndarray) -> np.ndarray:
    return quat_to_matrix(q) @ np.asarray(vel, dtype=np.float64)


class WbcPhysicsRunner:
    """Closed-loop WBC tracking in plain MuJoCo via an exported ONNX policy."""

    MODELS_DIR: str = "models"

    def __init__(self, mj_model: Any, mj_data: Any, params_dir: str | Path | None = None) -> None:
        self.mj_model: Any = mj_model
        self.mj_data: Any = mj_data

        params_dir = str(params_dir or Path(__file__).resolve().parents[2] / "models/params")
        config_path = os.path.join(params_dir, "config.yaml")
        onnx_path = os.path.join(params_dir, "policy.onnx")
        if not (os.path.exists(config_path) and os.path.exists(onnx_path)):
            raise RuntimeError(
                "No exported WBC policy bundle under models/params/. Restore the phase-2 bundle "
                "(policy.onnx, config.yaml, and robot_train/scene.xml)."
            )
        with open(config_path, "r") as f:
            cfg: dict = yaml.safe_load(f)

        if cfg.get("schema_version") != "wbc_tracking_params_v1" or cfg["action"]["action_mode"] != "reference_residual":
            raise ValueError("Unsupported WBC policy configuration.")
        if cfg["tracking"].get("actor_history_length", 1) != 1:
            raise ValueError("Only single-frame actor observations are supported.")
        self.joint_names: list[str] = list(cfg["joint_names"])
        self.n_joints: int = len(self.joint_names)
        self.default_joint_pos: np.ndarray = np.asarray(cfg["default_joint_pos"], dtype=np.float64)
        self.stiffness: np.ndarray = np.asarray(cfg["stiffness"], dtype=np.float64)
        self.damping: np.ndarray = np.asarray(cfg["damping"], dtype=np.float64)
        self.action_scale: np.ndarray = np.asarray(cfg["action"]["scale"], dtype=np.float64)
        self.policy_step_dt: float = float(cfg["policy_step_dt"])
        self.obs_names: list[str] = list(cfg["tracking"]["actor_observation_names"])
        for term in self.obs_names:
            if term not in _SUPPORTED_OBS_TERMS:
                raise RuntimeError(
                    f"Unsupported actor obs term {term!r}. Supported: {sorted(_SUPPORTED_OBS_TERMS)}"
                )
        self.obs_scales = {name: np.asarray(cfg["actor_observations"][name]["scale"], dtype=np.float32) for name in self.obs_names}
        self.anchor_body_name: str = str(cfg["tracking"]["anchor_body_name"])

        def _body_id(name: str) -> int:
            """Resolve a body name, tolerating the mjlab 'robot/' name prefix."""
            bid = int(mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, name))
            if bid < 0:
                bid = int(mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, f"robot/{name}"))
            if bid < 0:
                raise RuntimeError(f"Body {name!r} not found in the MuJoCo model.")
            return bid

        self.anchor_body_id: int = _body_id(self.anchor_body_name)
        self.root_body_id: int = _body_id("pelvis")
        # NPZ body index N refers to the N-th robot body; pelvis is NPZ body 0.
        self.npz_anchor_idx: int = self.anchor_body_id - self.root_body_id

        # Map each actuator to its driven joint's config index. The mjlab scene
        # orders actuators by training action groups, not by joint order, so the
        # joint-ordered torque vector must be reordered before writing ctrl.
        model_joints = [mj_model.joint(j).name.removeprefix("robot/") for j in range(mj_model.njnt) if mj_model.jnt_type[j] != mj.mjtJoint.mjJNT_FREE]
        if model_joints != self.joint_names or mj_model.nq != 36 or mj_model.nu != 29:
            raise ValueError("The model's joint order does not match the policy bundle.")
        joint_id_to_cfg: dict[int, int] = {}
        cfg_j = 0
        for j in range(mj_model.njnt):
            if mj_model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                continue
            joint_id_to_cfg[j] = cfg_j
            cfg_j += 1
        self.ctrl_idx: list[int] = [
            joint_id_to_cfg[int(mj_model.actuator_trnid[a, 0])] for a in range(mj_model.nu)
        ]

        self.substeps: int = max(1, int(round(self.policy_step_dt / float(mj_model.opt.timestep))))

        if not np.isclose(self.substeps * mj_model.opt.timestep, self.policy_step_dt):
            raise ValueError("Policy period must be an integer number of physics steps.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(onnx_path, sess_options=options, providers=["CPUExecutionProvider"])
        self.onnx_session: Any = session
        self.obs_dim: int = int(session.get_inputs()[0].shape[1])
        self.obs_buf: np.ndarray = np.zeros(self.obs_dim, dtype=np.float32)

        # Resolve IMU gyro sensor if present in the scene (matching mjlab's robot/imu_ang_vel)
        sensor_id = int(mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_SENSOR, "robot/imu_ang_vel"))
        if sensor_id < 0:
            sensor_id = int(mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_SENSOR, "imu_ang_vel"))
        self.imu_sensor_adr: int | None = (
            int(mj_model.sensor_adr[sensor_id]) if sensor_id >= 0 else None
        )

        self.clip: ClipReference | None = None
        self.frame: int = 0
        self.last_action: np.ndarray = np.zeros(self.n_joints, dtype=np.float64)
        self.last_action_scaled: np.ndarray = np.zeros(self.n_joints, dtype=np.float64)
        self.clip_origin_pos: np.ndarray = np.zeros(3, dtype=np.float64)
        self.clip_origin_quat: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.clip_ref_p0: np.ndarray = np.zeros(3, dtype=np.float64)

    @property
    def has_policy(self) -> bool:
        return True

    def clip_duration(self) -> float | None:
        """Duration of the active clip in seconds, or None if none is loaded."""
        if self.clip is None:
            return None
        return float(self.clip.n_frames) / self.clip.fps

    def load_clip(self, clip_path: str, speed: float = 1.0, *, anchor_to_current: bool = False) -> None:
        """Load a clip and resample it to the policy rate.

        If anchor_to_current is True, anchors the clip's frame-0 horizontal position
        and heading (yaw) to the robot's current pelvis pose in MuJoCo, so the clip continues
        seamlessly from where the robot currently is without resetting its position.
        """
        self.clip = ClipReference(clip_path, target_fps=1.0 / self.policy_step_dt, speed=speed)
        self.frame = 0

        ref_pos, ref_quat, _, _ = self.clip.anchor_frame(0, self.npz_anchor_idx)
        self.clip_ref_p0 = np.asarray(ref_pos, dtype=np.float64).copy()

        if not anchor_to_current:
            self.clip_origin_pos[:] = 0.0
            self.clip_origin_quat[:] = (1.0, 0.0, 0.0, 0.0)
        else:
            mj.mj_forward(self.mj_model, self.mj_data)
            robot_pos = np.asarray(self.mj_data.xpos[self.root_body_id], dtype=np.float64)
            robot_quat = np.asarray(self.mj_data.xquat[self.root_body_id], dtype=np.float64)
            yaw_robot = quat_yaw(robot_quat)
            yaw_ref = quat_yaw(ref_quat)

            self.clip_origin_pos[0] = robot_pos[0]
            self.clip_origin_pos[1] = robot_pos[1]
            self.clip_origin_pos[2] = 0.0
            self.clip_origin_quat = quat_multiply(yaw_robot, quat_conjugate(yaw_ref))

    def reset_to_initial_pose(self) -> None:
        """Initialise the robot to the first frame of the active clip (RSI-style)."""
        if self.clip is None:
            raise RuntimeError("No clip loaded; call load_clip() first.")
        mj.mj_resetData(self.mj_model, self.mj_data)
        q0 = self.clip.joint_pos[0].copy()
        pos, quat_wxyz, lin, ang = self.clip.anchor_frame(0, 0)
        self.mj_data.qpos[0:3] = np.asarray(pos, dtype=np.float64)
        self.mj_data.qpos[3:7] = quat_wxyz
        rot0 = quat_to_matrix(quat_wxyz)
        self.mj_data.qvel[0:3] = np.asarray(lin, dtype=np.float64)
        # MuJoCo freejoint rotational velocity qvel[3:6] is in local body frame
        self.mj_data.qvel[3:6] = rot0.T @ np.asarray(ang, dtype=np.float64)
        self.mj_data.qpos[7 : 7 + self.n_joints] = q0
        self.mj_data.qvel[6 : 6 + self.n_joints] = self.clip.joint_vel[0]
        self.frame = 0
        self.last_action[:] = 0.0
        self.last_action_scaled[:] = 0.0
        mj.mj_forward(self.mj_model, self.mj_data)

    def _reference(self) -> dict[str, np.ndarray]:
        if self.clip is None:
            raise RuntimeError("No clip loaded.")
        i = min(self.frame, self.clip.n_frames - 1)
        pos, quat_wxyz, lin, ang = self.clip.anchor_frame(i, self.npz_anchor_idx)
        pos = np.asarray(pos, dtype=np.float64)

        if np.all(self.clip_origin_pos == 0.0) and np.array_equal(self.clip_origin_quat, [1.0, 0.0, 0.0, 0.0]):
            ref_pos = pos
            ref_quat = quat_wxyz
            ref_lin = lin
            ref_ang = ang
        else:
            rel_pos = pos - self.clip_ref_p0
            rot_mat = quat_to_matrix(self.clip_origin_quat)
            rel_pos_rot = rot_mat @ np.array([rel_pos[0], rel_pos[1], 0.0], dtype=np.float64)

            ref_pos = np.array(
                [
                    self.clip_origin_pos[0] + rel_pos_rot[0],
                    self.clip_origin_pos[1] + rel_pos_rot[1],
                    pos[2],
                ],
                dtype=np.float64,
            )
            ref_quat = quat_multiply(self.clip_origin_quat, quat_wxyz)
            ref_lin = rot_mat @ np.asarray(lin, dtype=np.float64)
            ref_ang = rot_mat @ np.asarray(ang, dtype=np.float64)

        return {
            "ref_base_height": np.array([ref_pos[2]], dtype=np.float64),
            "ref_base_lin_vel_b": quat_to_matrix(ref_quat).T @ ref_lin,
            "ref_base_ang_vel_b": quat_to_matrix(ref_quat).T @ ref_ang,
            "ref_gravity_b": quat_to_matrix(ref_quat).T @ np.array([0.0, 0.0, -1.0]),
            "ref_joint_pos": self.clip.joint_pos[i],
        }

    def _robot_obs(self) -> dict[str, np.ndarray]:
        d = self.mj_data
        root_quat = np.asarray(d.xquat[self.root_body_id], dtype=np.float64)
        rot = quat_to_matrix(root_quat)

        if self.imu_sensor_adr is not None:
            base_ang_vel = np.asarray(
                d.sensordata[self.imu_sensor_adr : self.imu_sensor_adr + 3], dtype=np.float64
            )
        else:
            # MuJoCo returns rotational then linear velocity; use the local body frame.
            vel = np.zeros(6, dtype=np.float64)
            mj.mj_objectVelocity(self.mj_model, d, mj.mjtObj.mjOBJ_XBODY, self.root_body_id, vel, 1)
            base_ang_vel = vel[0:3]

        q = np.asarray(d.qpos[7 : 7 + self.n_joints], dtype=np.float64)
        v = np.asarray(d.qvel[6 : 6 + self.n_joints], dtype=np.float64)
        return {
            "base_ang_vel": base_ang_vel,
            "projected_gravity": rot.T @ np.array([0.0, 0.0, -1.0]),
            "joint_pos": q - self.default_joint_pos,
            "joint_vel": v,
            "actions": self.last_action,
        }

    def build_obs(self) -> np.ndarray:
        ref = self._reference()
        rob = self._robot_obs()
        out: list[np.ndarray] = []
        for term in self.obs_names:
            if term in ref:
                out.append(ref[term].astype(np.float32) * self.obs_scales[term])
            elif term in rob:
                out.append(rob[term].astype(np.float32) * self.obs_scales[term])
            else:
                raise RuntimeError(f"Missing obs term {term!r}")
        self.obs_buf = np.concatenate(out).astype(np.float32)
        if self.obs_buf.shape != (self.obs_dim,) or not np.isfinite(self.obs_buf).all():
            raise RuntimeError("Invalid policy observation shape or values.")
        return self.obs_buf

    def compute_action(self) -> np.ndarray:
        obs = self.build_obs()
        action = self.onnx_session.run(None, {self.onnx_session.get_inputs()[0].name: obs[None, :]})[0][0]
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.n_joints,) or not np.isfinite(action).all():
            raise RuntimeError("Policy produced invalid actions.")
        self.last_action = action.copy()
        self.last_action_scaled = self.action_scale * action
        return self.last_action_scaled

    def step_policy(self) -> np.ndarray:
        """One policy period: residual PD torque applied across `substeps` sim steps."""
        if self.clip is None:
            raise RuntimeError("No clip loaded; call load_clip() first.")
        scaled_action = self.compute_action()
        i = min(self.frame, self.clip.n_frames - 1)
        q_ref = self.clip.joint_pos[i]
        q_cmd = q_ref + scaled_action

        # Per-substep PD torque (matching mjlab's ideal PD + Unitree envelope).
        for _ in range(self.substeps):
            q = np.asarray(self.mj_data.qpos[7 : 7 + self.n_joints], dtype=np.float64)
            v = np.asarray(self.mj_data.qvel[6 : 6 + self.n_joints], dtype=np.float64)
            tau = self.stiffness * (q_cmd - q) - self.damping * v
            for j in range(self.n_joints):
                x1, x2, y1, y2, fs, fd, va, scale = _envelope_for(self.joint_names[j])
                same = bool((v[j] * tau[j]) > 0.0)
                peak = y1 if same else y2
                abs_v = abs(v[j])
                if abs_v > x1:
                    peak = max(0.0, peak - (peak / max(x2 - x1, 1e-6)) * (abs_v - x1))
                tau[j] = np.clip(tau[j], -peak, peak)
                tau[j] -= fs * np.tanh(v[j] / max(va, 1e-6)) + fd * v[j]
            for a, j in enumerate(self.ctrl_idx):
                self.mj_data.ctrl[a] = tau[j]
            mj.mj_step(self.mj_model, self.mj_data)

        mj.mj_forward(self.mj_model, self.mj_data)
        self.frame += 1
        return q_cmd
