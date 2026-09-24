import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

# Train robot to stand (not walk yet)
class G1WalkEnv(gym.Env):
    def __init__(self, model_path="../unitree_mujoco/unitree_robots/g1/scene_29dof.xml", terrain_generator=None, use_pd=True):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.terrain_generator = terrain_generator
        self.use_pd = use_pd

        # Physics: 10 substeps * 0.002s = 0.02s (50 Hz control loop)
        self.n_substeps = 10

        # All 29 actuators in scene_29dof are active
        self.active_actuator_ids = list(range(self.model.nu))
        self.active_joint_ids = [self.model.actuator_trnid[i, 0] for i in self.active_actuator_ids]
        n_active = len(self.active_actuator_ids)  # 29

        # Action space: normalized relative joint offsets in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_active,), dtype=np.float32)

        # Observation space: 29 qpos + 29 qvel + 3 gravity + 3 gyro + 29 last_action = 93 dims
        obs_dim = n_active * 3 + 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Nominal pose (all zeros for G1) and joint-specific PD gains
        self.qpos_nominal = np.zeros(n_active, dtype=np.float32)
        self.kp = np.zeros(n_active, dtype=np.float32)
        self.kd = np.zeros(n_active, dtype=np.float32)
        self._init_joint_configs()

        self.last_action = np.zeros(n_active, dtype=np.float32)
        self.max_episode_steps = 1000
        self._step_count = 0

    def _init_joint_configs(self):
        """Set gains across all 29 DOFs so torso and legs stay rigid and upright."""
        for local_i, joint_id in enumerate(self.active_joint_ids):
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id).lower()

            # Keep nominal pose at 0.0 (natural vertical standing pose for G1)
            self.qpos_nominal[local_i] = 0.0

            # 1. Legs: high stiffness to hold 35 kg body
            if "knee" in jname:
                self.kp[local_i] = 250.0
                self.kd[local_i] = 6.0
            elif "hip_pitch" in jname:
                self.kp[local_i] = 200.0
                self.kd[local_i] = 5.0
            elif "hip_roll" in jname or "hip_yaw" in jname:
                self.kp[local_i] = 150.0
                self.kd[local_i] = 4.0
            elif "ankle" in jname:
                self.kp[local_i] = 80.0
                self.kd[local_i] = 3.0

            # 2. 29-DOF Waist (prevents torso from slumping or pitching forward)
            elif "waist" in jname or "torso" in jname:
                self.kp[local_i] = 180.0
                self.kd[local_i] = 5.0
                if "pitch" in jname:
                    self.qpos_nominal[local_i] = -0.05

            # 3. Arms / Wrists
            elif "elbow" in jname:
                self.kp[local_i] = 50.0
                self.kd[local_i] = 1.5
            else:  # shoulders / wrists
                self.kp[local_i] = 40.0
                self.kd[local_i] = 1.2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Apply nominal pose (0.0)
        for local_i, joint_id in enumerate(self.active_joint_ids):
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr] = self.qpos_nominal[local_i]

        # G1 XML default pelvis standing height
        self.data.qpos[2] = 0.793

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self.last_action[:] = 0.0
        return self._get_obs(), {}

    def _get_target_angles(self, action):
        """Action scale of 0.2 rad lets policy adjust balance without wild flailing."""
        targets = np.zeros(len(self.active_joint_ids), dtype=np.float32)
        for local_i, joint_id in enumerate(self.active_joint_ids):
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id).lower()
            lo, hi = self.model.jnt_range[joint_id]

            # Arms and wrists should stay quiet during balance learning
            if any(part in jname for part in ["shoulder", "elbow", "wrist"]):
                scale = 0.05  # minimal arm movement
            elif "ankle" in jname:
                scale = 0.25  # ankles need good authority to push back against tipping
            else:
                scale = 0.15  # hips, knees, waist

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

        # PD runs at the 500 Hz physics rate
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
        alive_bonus = 2.0

        # Penalize tilt
        pelvis_id = self.model.body("pelvis").id
        xmat = self.data.xmat[pelvis_id].reshape(3, 3)
        gravity_local = xmat.T @ np.array([0, 0, -1.0])
        upright_reward = 0.5 * max(0.0, -gravity_local[2])

        # Maintain pelvis height (~0.79m)
        height_penalty = 5.0 * np.square(self.data.qpos[2] - 0.79)

        # Penalize drift and shaking
        vel_penalty = 0.1 * np.sum(np.square(self.data.qvel[:3])) + 0.05 * np.sum(np.square(self.data.qvel[3:6]))
        torque_penalty = 0.0005 * np.sum(np.square(self.data.ctrl))
        action_rate_penalty = 0.01 * np.sum(np.square(action - self.last_action))

        return alive_bonus + upright_reward - height_penalty - vel_penalty - torque_penalty - action_rate_penalty

    def _check_fall(self):
        if self.data.qpos[2] < 0.50:
            return True

        pelvis_id = self.model.body("pelvis").id
        xmat = self.data.xmat[pelvis_id].reshape(3, 3)
        gravity_local = xmat.T @ np.array([0, 0, -1.0])
        if gravity_local[2] > -0.5:
            return True

        return False
