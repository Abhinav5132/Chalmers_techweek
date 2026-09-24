import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

# These 6 joints exist in the XML but are attached to bodies floating at
# pos="0 0 20" — disconnected from the actual kinematic chain. They have
# actuators/sensors defined but moving them does nothing to the robot.
# We exclude them so the policy's action/obs space only covers real DOFs.
DISCONNECTED_ACTUATORS = {
    "waist_roll", "waist_pitch",
    "left_wrist_pitch", "left_wrist_yaw",
    "right_wrist_pitch", "right_wrist_yaw",
}


class G1WalkEnv(gym.Env):
    def __init__(self, model_path="g1_23dof.xml", terrain_generator=None, use_pd=True):
        # Load the MuJoCo model + create its mutable simulation state (data).
        # model = static structure (bodies, joints, actuators, ranges...)
        # data  = the actual simulation state (qpos, qvel, ctrl...) that changes every step
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # rebuild/randomize the scene each episode (domain randomization).
        self.terrain_generator = terrain_generator

        # If True, actions are treated as target joint angles and converted
        # to torques via a PD controller inside _scale_action(). If False,
        # actions map directly to raw torque commands (harder to learn from scratch).
        self.use_pd = use_pd

        # Filter down to only the 23 actuators that are actually connected
        # to the robot's body
        self.active_actuator_ids = get_active_actuator_ids(self.model)

        # For each active actuator, find which joint it drives.
        # actuator_trnid[i, 0] = the joint id that actuator i is attached to.
        self.active_joint_ids = [self.model.actuator_trnid[i, 0] for i in self.active_actuator_ids]
        n_active = len(self.active_actuator_ids)  # should be 23

        # --- Action space ---
        # Normalized to [-1, 1] per active joint; _scale_action() maps this
        # to real joint angles (PD mode) or real torques (raw torque mode).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_active,), dtype=np.float32)

        # --- Observation space ---
        # 23 joint pos + 23 joint vel + 3 gravity vec (base orientation,
        # expressed as gravity direction in the base frame) + 3 base angular
        # velocity + 23 previous action (helps the policy produce smoother
        # output by knowing what it did last step) = 75
        obs_dim = n_active * 3 + 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # PD gains — proportional (kp) and derivative (kd) terms. These are
        # rough starting values; they'll likely need tuning once training
        # starts (too stiff = jerky/unstable, too soft = robot can't hold pose).
        if use_pd:
            self.kp = 40.0
            self.kd = 1.0

        # Track the previous action so it can be included in the observation.
        self.last_action = np.zeros(n_active, dtype=np.float32)

        # Episode length cap — prevents infinite episodes when the robot
        # never falls (e.g. early training where it just stands still).
        self.max_episode_steps = 1000
        self._step_count = 0

    def reset(self, seed=None, options=None):
        # Gymnasium's reset() must seed the RNG via the parent class first,
        # so self.np_random is available if you need randomness below.
        super().reset(seed=seed)

        # optionally call self.terrain_generator here to rebuild the

        # Reset all simulation state (qpos, qvel, ctrl, etc.) back to the
        # model's defaults. Because no <keyframe> is defined, "default"
        # means all joint angles at 0 which happens to already be the
        # robot's neutral standing pose (pelvis is placed at z=0.793 for
        # exactly this configuration), so no extra pose-setting is needed.
        mujoco.mj_resetData(self.model, self.data)

        # Run one forward kinematics/dynamics pass so that derived
        # quantities (e.g. sensor readings, body positions) are valid
        # before the first observation is read.
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self.last_action[:] = 0.0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Convert the policy's normalized [-1, 1] action into either PD
        # torques (use_pd=True) or raw torque commands (use_pd=False),
        # written into the full-size ctrl vector (including the 6 disconnected
        # actuators, which just stay at 0 since we never touch their slots).
        ctrl = self._scale_action(action)
        self.data.ctrl[:] = ctrl

        # Step physics multiple times per RL step. This decouples the RL
        # control frequency from MuJoCo's internal physics timestep — 4
        # substeps is a reasonable starting point; increase for finer
        # physics fidelity, decrease for faster/more reactive control.
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_fall()  # episode ends early due to failure
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps  # episode ends due to time limit

        # Remember this action for next step's observation.
        self.last_action[:] = action

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # TODO: assemble the 75-dim observation vector:
        #   - joint positions for active_joint_ids (via self.data.qpos)
        #   - joint velocities for active_joint_ids (via self.data.qvel)
        #   - gravity vector in the pelvis/base frame (orientation proxy)
        #   - base angular velocity (e.g. from the imu_gyro sensor)
        #   - self.last_action
        pass

    def _compute_reward(self, action):
        # TODO: reward shaping
        #   + forward velocity tracking (reward matching a target velocity)
        #   - torque penalty (discourage wasteful/jerky control effort)
        #   - large penalty on falling (paired with _check_fall)

        pass

    def _check_fall(self):
        # TODO: define a failure condition, e.g.:
        #   - pelvis height (self.data.qpos[2]) drops below some threshold
        #   - base tilt (from the gravity vector) exceeds some angle
        # Without this, episodes never terminate early and the policy gets
        # no clear "you failed" signal.
        pass

    def _scale_action(self, action):
        # Maps each normalized action component to either:
        #   - a PD torque command computed from a target joint angle, or
        #   - a raw torque command
        # depending on self.use_pd. Unused actuator slots (the 6 disconnected
        # ones) are left at 0 in ctrl since we never write to their indices.
        ctrl = np.zeros(self.model.nu)

        for local_i, act_id in enumerate(self.active_actuator_ids):
            joint_id = self.active_joint_ids[local_i]

            # Index into qpos/qvel for this joint's position/velocity.
            qpos_adr = self.model.jnt_qposadr[joint_id]
            qvel_adr = self.model.jnt_dofadr[joint_id]

            # This joint's physical range of motion, from the XML.
            lo, hi = self.model.jnt_range[joint_id]

            if self.use_pd:
                # Map action in [-1, 1] to a target angle within [lo, hi].
                target_angle = lo + (action[local_i] + 1) * 0.5 * (hi - lo)

                # Standard PD control law: push toward target angle,
                # damped by current velocity to avoid oscillation/overshoot.
                torque = self.kp * (target_angle - self.data.qpos[qpos_adr]) \
                    - self.kd * self.data.qvel[qvel_adr]

                # Clip to the actuator's allowed torque range so we never
                # command something physically impossible.
                frc_lo, frc_hi = self.model.actuator_ctrlrange[act_id]
                ctrl[act_id] = np.clip(torque, frc_lo, frc_hi)
            else:
                # Raw torque mode: map action in [-1, 1] directly onto the
                # actuator's torque range (no PD, no target angle).
                frc_lo, frc_hi = self.model.actuator_ctrlrange[act_id]
                ctrl[act_id] = lo + (action[local_i] + 1) * 0.5 * (frc_hi - frc_lo)

        return ctrl

    def _get_obs_dim(self):
        # TODO: not currently used since obs_dim is computed directly in
        # __init__ from n_active. Remove, or use this if obs composition
        # changes and needs to be computed separately.
        pass


def get_active_actuator_ids(model):
    # Returns the indices of actuators that are NOT in DISCONNECTED_ACTUATORS,
    # i.e. the 23 actuators actually attached to the robot's kinematic chain.
    ids = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name not in DISCONNECTED_ACTUATORS:
            ids.append(i)
    return ids