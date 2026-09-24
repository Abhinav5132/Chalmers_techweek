from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from g1_walk_env import G1WalkEnv

SCENE_29_PATH = "../unitree_mujoco/unitree_robots/g1/scene_29dof.xml"


def make_env():
    return G1WalkEnv(SCENE_29_PATH)

if __name__ == "__main__":
    # fresh start, no loading prev results
    env = make_vec_env(make_env, n_envs=4)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=1e-4,
        target_kl=None,
        n_steps=2048,
        batch_size=64,
    )

    # continues from previous
    # model = PPO.load(
    #     "g1_balance_ppo_29dof",
    #     env=env,
    #     learning_rate=1e-4,
    #     target_kl=0.02,
    # )
    model.learn(total_timesteps=2_000_000, tb_log_name="g1_29dof_balance_v2_stable")
    model.save("g1_balance_ppo_29dof_v2")