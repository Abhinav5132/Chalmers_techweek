import time
import mujoco.viewer
from stable_baselines3 import PPO
from g1_walk_env import G1WalkEnv

SCENE_29_PATH = "../unitree_mujoco/unitree_robots/g1/scene_29dof.xml"

env = G1WalkEnv(SCENE_29_PATH)
model = PPO.load("g1_balance_ppo_29dof_v2")

obs, _ = env.reset()
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        viewer.sync()

        if terminated or truncated:
            obs, _ = env.reset()

        elapsed = time.time() - step_start
        if elapsed < 0.02:
            time.sleep(0.02 - elapsed)
