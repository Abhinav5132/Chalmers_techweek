from stable_baselines3 import PPO
from g1_locomotion_env import G1WalkEnv
import mujoco.viewer

env = G1WalkEnv("../unitree_mujoco/unitree_robots/g1/scene_29dof.xml", target_vx=0.1)
model = PPO.load("g1_walk_ppo_29dof_fresh")

obs, _ = env.reset()
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        viewer.sync()
        if terminated or truncated:
            obs, _ = env.reset()