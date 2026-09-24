from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from g1_walk_env import G1WalkEnv

def make_env():
    return G1WalkEnv("../unitree_mujoco/unitree_robots/g1/scene.xml")

if __name__ == "__main__":
    env = make_vec_env(make_env, n_envs=4)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=2_000_000, tb_log_name="g1_29dof_balance")
    model.save("g1_balance_ppo_29dof")