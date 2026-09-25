import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class G1WalkEnv(gym.Env):
    def __init__(
        self,
        model_path="../unitree_mujoco/unitree_robots/g1/scene_29dof.xml",
        target_vx=0.3,
        terrain_generator=None,
        use_pd=True,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.target_vx = target_vx
        self.dt = 0.02  # 50 Hz
        self.step_freq = 1.25  # 1.25 Hz walking cadence (0.8s per step cycle)

        self.n_substeps = 10
        self.active_actuator_ids = list(range(self.model.nu))
        self.active_joint_ids = [self.model.actuator_trnid[i, 0] for i in self.active_actuator_ids]
        n_active = len(self.active_actuator_ids)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_active,), dtype=np.float32)

        # 29 qpos + 29 qvel + 3 gravity + 3 gyro + 29 last_action + 2 clock (sin, cos) = 95 dims
        obs_dim = n_active * 3 + 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.qpos_nominal = np.zeros(n_active, dtype=np.float32)
        self.kp = np.zeros(n_active, dtype=np.float32)
        self.kd = np.zeros(n_active, dtype=np.float32)
        self._init_joint_configs()

        self.last_action = np.zeros(n_active, dtype=np.float32)
        self.last_pelvis_x = 0.0
        self.max_episode_steps = 1000
        self._step_count = 0

    def _init_joint_configs(self):
        for local_i, joint_id in enumerate(self.active_joint_ids):
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id).lower()

            # Natural bent-knee walking posture (avoids stilt singularity)
            if "hip_pitch" in jname:
                self.qpos_nominal[local_i] = -0.20
                self.kp[local_i] = 180.0
                self.kd[local_i] = 5.0
            elif "knee" in jname:
                self.qpos_nominal[local_i] = 0.40
                self.kp[local_i] = 200.0
                self.kd[local_i] = 5.0
            elif "ankle" in jname:
                if "pitch" in jname:
                    self.qpos_nominal[local_i] = -0.20
                self.kp[local_i] = 70.0
                self.kd[local_i] = 3.0
            elif "hip_roll" in jname or "hip_yaw" in jname:
                self.kp[local_i] = 120.0
                self.kd[local_i] = 4.0
            elif "waist" in jname or "torso" in jname:
                self.kp[local_i] = 150.0
                self.kd[local_i] = 5.0
            elif "elbow" in jname:
                self.kp[local_i] = 40.0
                self.kd[local_i] = 1.5
            else:
                self.kp[local_i] = 30.0
                self.kd[local_i] = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        for local_i, joint_id in enumerate(self.active_joint_ids):
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr] = self.qpos_nominal[local_i]

        # Lower initial height slightly for bent knees (~0.75m instead of 0.793m)
        self.data.qpos[2] = 0.75
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self.last_action[:] = 0.0
        self.last_pelvis_x = self.data.qpos[0]
        return self._get_obs(), {}

    def _get_target_angles(self, action):
        targets = np.zeros(len(self.active_joint_ids), dtype=np.float32)
        for local_i, joint_id in enumerate(self.active_joint_ids):
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id).lower()
            lo, hi = self.model.jnt_range[joint_id]

            if any(part in jname for part in ["shoulder", "elbow", "wrist"]):
                scale = 0.05
            elif "ankle" in jname:
                scale = 0.25
            elif "hip_pitch" in jname or "knee" in jname:
                scale = 0.35  # Generous leg swing range
            else:
                scale = 0.15

            targets[local_i] = np.clip(self.qpos_nominal[local_i] + action[local_i] * scale, lo, hi)
        return targets

    def _compute_pd(self, target_angles):
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for local_i, act_id in enumerate(self.active_actuator_ids):
            joint_id = self.active_joint_ids[local_i]
            qpos = self.data.qpos[self.model.jnt_qposadr[joint_id]]
            qvel = self.data.qvel[self.model.jnt_dofadr[joint_id]]

            torque = self.kp[local_i] * (target_angles[local_i] - qpos) - self.kd[local_i] * qvel
            frc_lo, frc_hi = self.model.actuator_ctrlrange[act_id]
            ctrl[act_id] = np.clip(torque, frc_lo, frc_hi)
        return ctrl

    def step(self, action):
        target_angles = self._get_target_angles(action)

        for _ in range(self.n_substeps):
            self.data.ctrl[:] = self._compute_pd(target_angles)
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_fall()

        if terminated:
            reward -= 10.0

        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        self.last_action[:] = action
        self.last_pelvis_x = self.data.qpos[0]

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        qpos = np.array([self.data.qpos[self.model.jnt_qposadr[j]] for j in self.active_joint_ids], dtype=np.float32)
        qvel = np.array([self.data.qvel[self.model.jnt_dofadr[j]] for j in self.active_joint_ids], dtype=np.float32)

        pelvis_id = self.model.body("pelvis").id
        xmat = self.data.xmat[pelvis_id].reshape(3, 3)
        gravity_local = xmat.T @ np.array([0, 0, -1.0])
        ang_vel = self.data.sensor("imu_gyro").data.copy()

        return np.concatenate([qpos, qvel, gravity_local, ang_vel, self.last_action]).astype(np.float32)


    def _compute_reward(self, action):
        alive_bonus = 0.5

        # Upright torso
        pelvis_id = self.model.body("pelvis").id
        xmat = self.data.xmat[pelvis_id].reshape(3, 3)
        gravity_local = xmat.T @ np.array([0, 0, -1.0])
        upright_reward = 0.5 * max(0.0, -gravity_local[2])

        # Pelvis height (~0.72m for bent knees)
        pelvis_z = self.data.qpos[2]
        height_penalty = 5.0 * np.square(pelvis_z - 0.72)

        # TRUE PROGRESS: change in X position over this 0.02s step
        # Target displacement per step = target_vx * dt = 0.3 * 0.02 = 0.006 meters
        actual_dx = self.data.qpos[0] - self.last_pelvis_x
        target_dx = self.target_vx * self.dt
        # Scaled so hitting 0.3 m/s gives +2.0 reward per step; standing still gives 0.0
        progress_reward = 2.0 * min(max(actual_dx / target_dx, 0.0), 1.2)

        # Drift and spin penalties
        lateral_penalty = 1.0 * np.square(self.data.qvel[1])
        yaw_penalty = 0.5 * np.square(self.data.qvel[5])

        # Penalize excessive joint torques and jerky actions
        torque_penalty = 0.0005 * np.sum(np.square(self.data.ctrl[self.active_actuator_ids]))
        action_rate_penalty = 0.02 * np.sum(np.square(action - self.last_action))

        return (
            alive_bonus
            + upright_reward
            + progress_reward
            - height_penalty
            - lateral_penalty
            - yaw_penalty
            - torque_penalty
            - action_rate_penalty
        )

    def _check_fall(self):
        if self.data.qpos[2] < 0.55:
            return True

        pelvis_id = self.model.body("pelvis").id
        xmat = self.data.xmat[pelvis_id].reshape(3, 3)
        gravity_local = xmat.T @ np.array([0, 0, -1.0])
        if gravity_local[2] > -0.80:  # ~35 deg tilt limit
            return True

        return False
