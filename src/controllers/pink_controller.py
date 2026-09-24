"""
Pink & MuJoCo Whole-Body / Arm Kinematics Controller for Unitree G1 (29-DoF).
Uses Pink QP Inverse Kinematics for Cartesian tasks and low-level joint PD control.
Hard crashes if Pinocchio, Pink, or a compatible QP solver is unavailable.
"""

from __future__ import annotations

import os
from typing import Any
import numpy as np
import mujoco

import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
import qpsolvers

# MuJoCo is a compiled C-extension (_structs.so).
# We alias to Any so static type checkers (like ty) do not flag missing members on the C-module.
mj: Any = mujoco


class PinkG1Controller:
    """
    Kinematic controller for Unitree G1 humanoid (29-DoF).
    Computes joint targets for Cartesian reaching/manipulation tasks via Pink QP IK
    and calculates low-level PD actuator torques for MuJoCo motors.
    """

    # 29 Joint Actuator Names matching g1_29dof.xml order
    ACTUATOR_NAMES: list[str] = [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch",
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"
    ]

    # Standard PD gains for G1 joints (Kp, Kd, max_torque)
    DEFAULT_GAINS: dict[str, tuple[float, float, float]] = {
        "hip": (60.0, 3.0, 88.0),
        "knee": (100.0, 4.0, 139.0),
        "ankle": (40.0, 2.0, 50.0),
        "waist": (80.0, 3.0, 50.0),
        "shoulder": (30.0, 1.5, 25.0),
        "elbow": (30.0, 1.5, 25.0),
        "wrist": (15.0, 0.8, 15.0),
    }

    def __init__(
        self,
        mj_model: Any,
        mj_data: Any,
        xml_path: str
    ) -> None:
        self.mj_model: Any = mj_model
        self.mj_data: Any = mj_data
        self.xml_path: str = xml_path

        # Cache joint & actuator mappings
        self.n_ctrl: int = int(mj_model.nu)
        self.kp: np.ndarray = np.zeros(self.n_ctrl, dtype=np.float64)
        self.kd: np.ndarray = np.zeros(self.n_ctrl, dtype=np.float64)
        self.torque_limit: np.ndarray = np.zeros(self.n_ctrl, dtype=np.float64)
        self._init_gains()

        # Nominal stand pose for 29 actuated joints
        self.q_nominal_29: np.ndarray = np.zeros(self.n_ctrl, dtype=np.float64)
        self._init_nominal_posture()
        self.rest_pelvis_z: float = self._compute_rest_pelvis_z()

        # QP solver selection - prefer quadprog for active-set robustness, fallback to proxqp
        if "quadprog" in qpsolvers.available_solvers:
            self.solver: str = "quadprog"
        elif "proxqp" in qpsolvers.available_solvers:
            self.solver: str = "proxqp"
        else:
            raise RuntimeError(
                "No compatible QP solver found for Pink (quadprog or proxqp required). "
                f"Available solvers: {qpsolvers.available_solvers}"
            )

        # Build Pinocchio model and Pink configuration - hard crash on failure
        self.pin_model: pin.Model
        self.pin_data: pin.Data
        self.pink_config: pink.Configuration
        self.posture_task: PostureTask
        self._init_pink(xml_path)

    def _init_gains(self) -> None:
        """Populate PD gains based on actuator name categorization."""
        for i, name in enumerate(self.ACTUATOR_NAMES):
            if "knee" in name:
                kp, kd, limit = self.DEFAULT_GAINS["knee"]
            elif "ankle" in name:
                kp, kd, limit = self.DEFAULT_GAINS["ankle"]
            elif "hip" in name:
                kp, kd, limit = self.DEFAULT_GAINS["hip"]
            elif "waist" in name:
                kp, kd, limit = self.DEFAULT_GAINS["waist"]
            elif "shoulder" in name:
                kp, kd, limit = self.DEFAULT_GAINS["shoulder"]
            elif "elbow" in name:
                kp, kd, limit = self.DEFAULT_GAINS["elbow"]
            elif "wrist" in name:
                kp, kd, limit = self.DEFAULT_GAINS["wrist"]
            else:
                kp, kd, limit = (40.0, 2.0, 30.0)

            # Check if xml specifies ctrlrange
            if self.mj_model.actuator_ctrlrange is not None:
                limit = float(max(
                    abs(self.mj_model.actuator_ctrlrange[i, 0]),
                    abs(self.mj_model.actuator_ctrlrange[i, 1])
                ))

            self.kp[i] = kp
            self.kd[i] = kd
            self.torque_limit[i] = limit

    def _init_nominal_posture(self) -> None:
        """Set nominal calibrated standing posture values for G1 joints."""
        # Legs: calibrated standing bend (Unitree RL default)
        self.q_nominal_29[0] = -0.1   # left_hip_pitch
        self.q_nominal_29[3] = 0.3    # left_knee
        self.q_nominal_29[4] = -0.2   # left_ankle_pitch

        self.q_nominal_29[6] = -0.1   # right_hip_pitch
        self.q_nominal_29[9] = 0.3    # right_knee
        self.q_nominal_29[10] = -0.2  # right_ankle_pitch

        # Arms: rest positions
        self.q_nominal_29[15] = 0.2   # left_shoulder_pitch
        self.q_nominal_29[16] = 0.2   # left_shoulder_roll
        self.q_nominal_29[18] = 0.4   # left_elbow
        self.q_nominal_29[22] = 0.2   # right_shoulder_pitch
        self.q_nominal_29[23] = -0.2  # right_shoulder_roll
        self.q_nominal_29[25] = 0.4   # right_elbow

    def _compute_rest_pelvis_z(self) -> float:
        """
        Computes the exact pelvis height so that the foot contact geoms rest
        precisely on the ground plane (z=0.0) with zero penetration.
        """
        test_z: float = 1.0
        saved_qpos: np.ndarray = np.array(self.mj_data.qpos, copy=True)
        self.mj_data.qpos[0:3] = [0.0, 0.0, test_z]
        self.mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.mj_data.qpos[7 : 7 + self.n_ctrl] = self.q_nominal_29
        mj.mj_forward(self.mj_model, self.mj_data)

        min_z: float = float("inf")
        for i in range(self.mj_model.ngeom):
            if int(self.mj_model.geom_type[i]) == int(mj.mjtGeom.mjGEOM_PLANE):
                continue
            radius: float = (
                float(self.mj_model.geom_size[i][0])
                if int(self.mj_model.geom_type[i]) == int(mj.mjtGeom.mjGEOM_SPHERE)
                else 0.0
            )
            geom_bottom: float = float(self.mj_data.geom_xpos[i][2]) - radius
            if geom_bottom < min_z:
                min_z = geom_bottom

        # Restore simulation state
        self.mj_data.qpos[:] = saved_qpos
        mj.mj_forward(self.mj_model, self.mj_data)

        return test_z - min_z

    def _init_pink(self, xml_path: str) -> None:
        """Build Pinocchio model from URDF or MuJoCo XML and initialize Pink tasks."""
        urdf_path = xml_path.replace(".xml", ".urdf")
        if not os.path.exists(urdf_path):
            alt_urdf = os.path.join(os.path.dirname(xml_path), "g1_29dof.urdf")
            if os.path.exists(alt_urdf):
                urdf_path = alt_urdf

        if os.path.exists(urdf_path):
            self.pin_model = pin.buildModelFromUrdf(urdf_path)
        elif hasattr(pin, "mjcf"):
            self.pin_model = pin.mjcf.buildModel(xml_path)
        else:
            raise RuntimeError(
                f"Pinocchio build failed: URDF not found at {urdf_path} and installed Pinocchio wheel lacks mjcf parser. "
                f"Run 'just setup-urdf' to download the official G1 URDF."
            )

        self.pin_data = self.pin_model.createData()

        # Initial configuration matching current simulation state
        start_idx = 7 if (self.pin_model.nq == self.n_ctrl and len(self.mj_data.qpos) > self.n_ctrl) else 0
        q_init = np.asarray(self.mj_data.qpos[start_idx : start_idx + self.pin_model.nq], dtype=np.float64).copy()
        self.pink_config = pink.Configuration(self.pin_model, self.pin_data, q_init)

        # Set up posture task to maintain nominal standing configuration
        self.posture_task = PostureTask(cost=1e-1)
        self.posture_task.set_target(self.q_nominal_29)

    def get_pelvis_rot_matrix(self) -> np.ndarray:
        """Returns the 3x3 rotation matrix of the pelvis floating base in world frame."""
        w, x, y, z = self.mj_data.qpos[3:7]
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float64)

    def world_to_pelvis(self, pos_world: np.ndarray) -> np.ndarray:
        """Transforms a 3D coordinate from MuJoCo world frame into Pinocchio root (pelvis) frame."""
        p_pelvis = np.asarray(self.mj_data.qpos[0:3], dtype=np.float64)
        r_pelvis = self.get_pelvis_rot_matrix()
        return r_pelvis.T @ (np.asarray(pos_world, dtype=np.float64).reshape(3) - p_pelvis)

    def world_rot_to_pelvis(self, rot_world: np.ndarray) -> np.ndarray:
        """Transforms a 3x3 rotation matrix from MuJoCo world frame into Pinocchio root (pelvis) frame."""
        r_pelvis = self.get_pelvis_rot_matrix()
        return r_pelvis.T @ np.asarray(rot_world, dtype=np.float64)

    def solve_reach(
        self,
        target_pos: np.ndarray,
        target_rpy: np.ndarray | None = None,
        body_name: str = "right_wrist_yaw_link",
        dt: float = 0.01,
        in_world_frame: bool = True
    ) -> np.ndarray:
        """
        Solve Inverse Kinematics using Pink QP to move specified body to target_pos (Cartesian XYZ).
        If in_world_frame is True, target_pos and target_rpy are interpreted in MuJoCo world coordinates.
        Returns target joint position vector (29-DoF). Hard crashes if IK solve fails.
        """
        pos = self.world_to_pelvis(target_pos) if in_world_frame else np.asarray(target_pos, dtype=np.float64).reshape(3)
        rpy = np.asarray(target_rpy, dtype=np.float64).reshape(3) if target_rpy is not None else None

        # Sync Pink configuration with current MuJoCo state, safely clipped inside URDF limits
        start_idx = 7 if (self.pin_model.nq == self.n_ctrl and len(self.mj_data.qpos) > self.n_ctrl) else 0
        q_raw = np.asarray(self.mj_data.qpos[start_idx : start_idx + self.pin_model.nq], dtype=np.float64)
        eps = 1e-4
        q_safe = np.clip(q_raw, self.pin_model.lowerPositionLimit + eps, self.pin_model.upperPositionLimit - eps)
        self.pink_config.q = q_safe
        self.pink_config.update()

        # Build target transform in Pinocchio root (pelvis) frame
        if rpy is not None:
            rot_world = pin.utils.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            rot_matrix = self.world_rot_to_pelvis(rot_world) if in_world_frame else rot_world
        else:
            rot_matrix = self.pink_config.get_transform_frame_to_world(body_name).rotation

        target_transform = pin.SE3(rot_matrix, pos)

        frame_task = FrameTask(
            body_name,
            position_cost=5.0,
            orientation_cost=0.1 if rpy is not None else 0.0,
            gain=5.0,
        )
        frame_task.set_target(target_transform)

        tasks = [frame_task, self.posture_task]

        velocity = pink.solve_ik(self.pink_config, tasks, dt, solver=self.solver)
        self.pink_config.integrate_inplace(velocity, dt)

        if self.pin_model.nq == self.n_ctrl:
            q_solved = np.asarray(self.pink_config.q[: self.n_ctrl], dtype=np.float64)
        else:
            q_solved = np.asarray(self.pink_config.q[7 : 7 + self.n_ctrl], dtype=np.float64)

        return q_solved

    def compute_pd_torques(
        self,
        q_target: np.ndarray,
        v_target: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute joint torques via low-level PD:
        τ = Kp * (q_target - q_current) + Kd * (v_target - v_current)
        clipped to actuator torque limits.
        """
        q_target = np.asarray(q_target, dtype=np.float64).reshape(self.n_ctrl)
        q_current = np.asarray(self.mj_data.qpos[7 : 7 + self.n_ctrl], dtype=np.float64)
        v_current = np.asarray(self.mj_data.qvel[6 : 6 + self.n_ctrl], dtype=np.float64)

        if v_target is None:
            v_des = np.zeros(self.n_ctrl, dtype=np.float64)
        else:
            v_des = np.asarray(v_target, dtype=np.float64).reshape(self.n_ctrl)

        q_err = q_target - q_current
        v_err = v_des - v_current

        torques = self.kp * q_err + self.kd * v_err
        clipped_torques: np.ndarray = np.clip(torques, -self.torque_limit, self.torque_limit)

        return clipped_torques

    def step_reach(
        self,
        target_pos: np.ndarray,
        target_rpy: np.ndarray | None = None,
        body_name: str = "right_wrist_yaw_link",
        dt: float = 0.002,
        in_world_frame: bool = True
    ) -> np.ndarray:
        """Convenience method: Solves Pink IK, computes PD torques, and applies to data.ctrl."""
        q_des = self.solve_reach(target_pos, target_rpy, body_name, dt, in_world_frame=in_world_frame)
        torques = self.compute_pd_torques(q_des)
        self.mj_data.ctrl[:] = torques
        return q_des
