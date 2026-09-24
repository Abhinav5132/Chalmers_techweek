import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class G1WalkEnv(gym.Env):
    def __init__(self, model_path="g1_23dof.xml", terrain_generator=None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.terrain_generator = terrain_generator  # to randomize terrain on reset

        # --- Action space ---
        # decide target joint angels
        n_actuators = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actuators, ), dtype=np.float32)

        # --- Observation space ---
        obs_dim = self._get_obs_dim()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim, ), dtype=np.float32)

        self.max_episode_steps = 1000
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # optionally regenerate terrain with self.terrain_generator

        mujoco.mj_resetData(self.model, self.data)
        # TODO set initial joint pose to stable standing config
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # TODO map normalized action [-1, 1] to actual joint target ranges
        target = self._scale_action(action)
        self.data.ctrl[:] = target

        # TODO decicde sim substeps per RL step (control freq vs physics freq)
        for i in range(4):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_fall()
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}


    def _get_obs(self):
        #TODO assemble from self.data.qpos, self.data.qvel etc
        pass
    
    def _compute_reward(self, action):
        pass
    
    def _check_fall(self):
        pass
    
    def _scale_action(self, action):
        # map [-1, 1] - actuator control range from self.model.actuator_ctrlrange
        pass
    
    def _get_obs_dim(self):
        pass