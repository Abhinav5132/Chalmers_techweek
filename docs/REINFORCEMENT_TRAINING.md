# Reinforcement training: G1 standing and walking

The reinforcement learning implementation lives on `origin/reinforcement-step`.
This guide documents commit `85833d582229cdad24a01979c696d41ed57d7491`
(`fine tuning parameters`). The documentation is maintained on
`feat/motion-studio-physics`; the commands below require a checkout of the
reinforcement branch's code.

The pipeline uses Stable-Baselines3 PPO with an `MlpPolicy` in MuJoCo:

1. Learn standing balance from scratch using a reward for remaining upright.
2. Load the standing policy and fine-tune it for forward walking with additional
   progress and foot-clearance rewards.
3. Inspect deterministic policy rollouts in the MuJoCo viewer.

This is reward-based training, separate from this branch's supervised imitation
pipelines and exported WBC motion-tracking controller. The scripts use a flat
ground scene. Despite the branch name, they do not train stair or step climbing,
and they are not connected to the web studio or Hermes/MCP commands.

## Source map

Paths in this table are relative to the reinforcement checkout.

| File | Purpose |
| --- | --- |
| `src/g1_walk_env.py` | Standing environment; its class is named `G1WalkEnv` despite training balance. |
| `src/g1_locomotion_env.py` | Walking environment, also named `G1WalkEnv`. |
| `src/scripts/train_g1_scene29.py` | Fresh standing PPO training. |
| `src/scripts/train_g1_walk_sc29.py` | Walking fine-tuning from the standing checkpoint. |
| `src/scripts/eval_29dof.py` | Standing viewer. |
| `src/scripts/eval_walk.py` | Walking viewer; checkpoint name needs adjustment (below). |
| `src/g1_balance_ppo_29dof_v2.zip` | Committed standing checkpoint. |
| `unitree_mujoco/unitree_robots/g1/scene_29dof.xml` | Flat-ground scene including the 29-DoF G1 model. |

## Set up a separate checkout

Run this from the current repository root to leave your current branch in place.
The destination directory must not already exist:

```bash
git worktree add --detach ../Chalmers_techweek-reinforcement 85833d582229cdad24a01979c696d41ed57d7491
cd ../Chalmers_techweek-reinforcement
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python mujoco numpy gymnasium stable-baselines3 tensorboard
cd src
```

These install the packages needed by the RL scripts into the separate checkout.
The branch's `pyproject.toml` and lockfile omit Gymnasium, Stable-Baselines3 and
TensorBoard, so `uv sync` alone does not prepare RL training. These extra packages
are unpinned; record their installed versions alongside results when comparing
runs. The project has other dependencies for unrelated features.

**Run all subsequent commands from the reinforcement checkout's `src/` directory.**
The scripts resolve scene paths, checkpoints and logs against the working
directory. `PYTHONPATH=.` lets scripts under `scripts/` import the environment
modules from `src/`. Direct virtual-environment executables below avoid an
automatic project sync removing the extra packages.

## Train and inspect standing

The committed standing checkpoint allows you to skip standing training and start
with its viewer or walking fine-tuning. To train a replacement:

```bash
# Preserve the bundled checkpoint before the training script overwrites it.
cp -n g1_balance_ppo_29dof_v2.zip g1_balance_ppo_29dof_v2.bundled.zip
PYTHONPATH=. ../.venv/bin/python scripts/train_g1_scene29.py
```

The active code creates a fresh policy; the commented `PPO.load` block does not
resume training. Saving happens after `learn()` completes, overwriting
`src/g1_balance_ppo_29dof_v2.zip`. There are no periodic checkpoint callbacks.

Inspect the policy on macOS:

```bash
PYTHONPATH=. ../.venv/bin/mjpython scripts/eval_29dof.py
```

On Linux with a desktop display, replace `mjpython` with `python`. Training is
headless and uses `python` on either platform. The viewer resets after a fall or
time limit and runs until its window is closed. Standing evaluation attempts to
pace each policy step to 20 ms.

## Fine-tune walking

With `g1_balance_ppo_29dof_v2.zip` present in `src/`:

```bash
PYTHONPATH=. ../.venv/bin/python scripts/train_g1_walk_sc29.py
```

The script loads that standing checkpoint, creates eight subprocess environments
at a target forward speed of **0.2 m/s**, and requests one million additional
timesteps with `reset_num_timesteps=False`. It saves
`src/g1_walk_ppo_29dof.zip`. Re-running the script starts from standing again,
not from the saved walking policy. No walking checkpoint is committed at the
documented revision.

Before using the walking viewer, edit these two lines in
`scripts/eval_walk.py` **in the separate reinforcement checkout**:

```python
env = G1WalkEnv("../unitree_mujoco/unitree_robots/g1/scene_29dof.xml", target_vx=0.2)
model = PPO.load("g1_walk_ppo_29dof")
```

The checked-in evaluator instead loads `g1_walk_ppo_29dof_fresh`, which the
trainer does not produce, and sets `target_vx=0.1`. Aligning the speed makes its
reward calculation match training. Target speed is not part of the policy
observation, so changing it in the evaluator does not command a new gait speed.

Then run on macOS (use `python` instead of `mjpython` on Linux):

```bash
PYTHONPATH=. ../.venv/bin/mjpython scripts/eval_walk.py
```

The walking viewer has no real-time sleep; apparent playback speed depends on
simulation and rendering throughput.

## Training settings and outputs

| Setting | Standing | Walking |
| --- | --- | --- |
| Initialization | Fresh `MlpPolicy` | Standing checkpoint via `PPO.load` |
| Vector environments | 4, `make_vec_env` (default in-process vectorization) | 8, `SubprocVecEnv` |
| Requested timesteps | 2,000,000 | 1,000,000 additional |
| Learning rate | `1e-4` | `1e-4` override on load |
| Rollout steps per environment | 2,048 | Inherited from loaded checkpoint |
| Minibatch size | 64 | Inherited from loaded checkpoint |
| Target KL | `None` | Inherited from loaded checkpoint |
| TensorBoard directory under `src/` | `logs/` | `g1_walk_tensorboard/` |
| TensorBoard run name | `g1_29dof_balance_v2_stable` | `g1_walk_run1` |
| Saved checkpoint under `src/` | `g1_balance_ppo_29dof_v2.zip` | `g1_walk_ppo_29dof.zip` |

PPO collects complete rollouts, so actual timestep counts can exceed the requested
budgets. Walking inherits other algorithm settings from the loaded checkpoint;
do not assume a bundled checkpoint has exactly the current standing trainer's
settings. Neither script explicitly selects a device or training seed.

From `src/`, inspect logs in a separate terminal:

```bash
../.venv/bin/tensorboard --logdir logs
# Or, for walking:
../.venv/bin/tensorboard --logdir g1_walk_tensorboard
```

Open the address printed by TensorBoard. Episode reward and length help track
learning, but neither alone establishes stable walking. The viewers do not write
success-rate reports, speed measurements or evaluation CSVs.

## Environment and motor control

Both environments control all 29 actuators. Each policy action contains 29
normalized joint offsets in `[-1, 1]`. A joint-specific scale converts these into
radian offsets from a nominal pose, with targets clipped to joint limits:

| Joint group | Standing scale | Walking scale |
| --- | --- | --- |
| Shoulders, elbows, wrists | 0.05 | 0.05 |
| Ankles | 0.25 | 0.25 |
| Hip pitch, knees | 0.15 | 0.35 |
| Hip roll | 0.15 | 0.25 |
| Remaining joints | 0.15 | 0.15 |

For each policy step, ten MuJoCo substeps recompute
`torque = kp * (target_angle - joint_angle) - kd * joint_velocity`, clipped to
the actuator's control range. With the model's default 0.002 s timestep this
gives 500 Hz PD control and 50 Hz policy updates. Joint-specific gains are defined
in each environment's `_init_joint_configs()`.

The 93-element observation concatenates, in order: 29 joint positions, 29 joint
velocities, 3 components of pelvis-frame gravity, 3 `imu_gyro` values, and 29
stored action values. There is no commanded speed, base linear velocity, terrain
map or foot-contact measurement in the observation.

Standing resets the pelvis to 0.793 m and joint angles to zero except waist pitch
at -0.05 rad. Walking resets to 0.76 m with hip pitch -0.10, knees 0.20, ankle
pitch -0.20 and waist pitch -0.05 rad. Neither reset randomizes the initial state.

Episodes truncate after 1,000 policy steps (20 simulated seconds). Standing
terminates below 0.50 m pelvis height or beyond 60 degrees of pelvis tilt;
walking terminates below 0.55 m or beyond approximately 25.8 degrees. Both apply
an additional -10 reward on a fall.

## Rewards

Let `u` be pelvis uprightness (`-gravity_local[2]`), `z` pelvis height, `a` the
current action, `a_prev` the stored action, and `tau` the final substep's motor
controls. Squared norms sum across the relevant components.

Standing reward before the fall penalty:

```text
2 + 0.5 * max(0, u)
  - 5 * (z - 0.79)^2
  - 0.1 * ||base linear velocity||^2
  - 0.05 * ||base angular velocity||^2
  - 0.0005 * ||tau||^2
  - 0.01 * ||a - a_prev||^2
```

Walking reward before the fall penalty:

```text
0.5 + 0.5 * max(0, u) + progress + clearance - dragging
    - 5 * (z - 0.76)^2
    - lateral_velocity^2 - 0.5 * yaw_velocity^2
    - 0.1 * imu_pitch_rate^2
    - 0.0005 * ||tau||^2 - 0.02 * ||a - a_prev||^2
```

- `progress = 0.5 * clip(delta_x / (target_vx * 0.02), 0, 1.2)` when not
  fallen, otherwise zero. Use a positive target speed; zero divides by zero.
- `clearance = 1` if one ankle-roll body origin is above 0.035 m and the other
  below 0.025 m, otherwise zero. These are body-origin heights, not measured
  sole clearance or contact forces.
- `dragging = 0.5` if both origins are below 0.025 m while forward progress
  exceeds 0.001 m per policy step, otherwise zero.

The clearance rule does not enforce alternation over time. Despite a source
comment mentioning jumping, there is no explicit both-feet-airborne penalty.

## Known limitations and troubleshooting

- **Scene file not found or environment import fails:** run from `src/` with
  `PYTHONPATH=.` as shown above, not from the repository root or `src/scripts/`.
- **Missing packages:** use the setup above; the branch has no RL dependency
  group or dedicated `just` training recipe. Its `src/readme.md` mistakenly labels
  an evaluation command as training.
- **Walking checkpoint not found:** train walking first and correct the evaluator's
  checkpoint name. The standing trainer saves only after completing training too.
- **macOS viewer launch error:** use the virtual environment's `mjpython` for
  passive viewers. A desktop display is required for evaluation.
- **Action history timing:** both `step()` implementations assemble the returned
  observation before updating `last_action`. Thus the history field in that
  observation is one action older than the action just applied. Preserve this
  behavior when replaying existing policies; correcting it changes their inputs.
- **Unused configuration:** `terrain_generator` and `use_pd` constructor arguments
  do not enable terrain generation or disable PD in the stepping code.
- **Evidence scope:** the branch contains a standing checkpoint, but no quantitative
  benchmark demonstrating walking reliability. Evaluation automatically resets
  after falls. Inspect falls, actual speed and sustained alternating steps over
  multiple episodes before describing a policy as successful. There is no domain
  randomization, obstacle curriculum or hardware deployment path in these scripts.

This guide was checked against source at the recorded revision. It does not
report a new training run or measured policy performance.
