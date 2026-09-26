import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from g1_locomotion_env import G1WalkEnv

SCENE_PATH = "../unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
WALK_CHECKPOINT = "g1_walk_ppo_29dof"

# Candidate checkpoints from your standing run
STAND_CANDIDATES = [
    "g1_balance_ppo_29dof_v2",
]


def find_checkpoint():
    for name in STAND_CANDIDATES:
        zip_path = f"{name}.zip"
        if os.path.exists(zip_path) or os.path.exists(name):
            return name
    return None


def make_env():
    def _init():
        # Target a gentle forward speed of 0.2 m/s first so it doesn't dive forward
        return G1WalkEnv(SCENE_PATH, target_vx=0.2)
    return _init


if __name__ == "__main__":
    checkpoint_name = find_checkpoint()
    if checkpoint_name is None:
        print("ERROR: Could not find any standing checkpoint zip file!")
        print(f"Searched for: {[c + '.zip' for c in STAND_CANDIDATES]}")
        print("Ensure your trained standing model is in the current directory.")
        sys.exit(1)

    print(f"Found standing checkpoint: '{checkpoint_name}.zip'")

    num_envs = 8
    print(f"Creating {num_envs} vectorized environments...")
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    print(f"Loading weights from {checkpoint_name}...")
    model = PPO.load(
        checkpoint_name,
        env=env,
        learning_rate=1e-4,  # Lower learning rate for fine-tuning
        tensorboard_log="./g1_walk_tensorboard/",
    )

    print("Weights loaded successfully.")
    print("Starting fine-tuning for walking (1,000,000 timesteps)...")

    model.learn(
        total_timesteps=1_000_000,
        tb_log_name="g1_walk_run1",
        reset_num_timesteps=False,
    )

    model.save(WALK_CHECKPOINT)
    print(f"Training complete. Saved walking model to {WALK_CHECKPOINT}.zip")
