"""Chained clip playback: run a trained WBC policy through a sequence of clips.

Extends the play pipeline with sequential clip chaining: when the active clip
reaches its end, the robot is placed (RSI-style) at the first frame of the next
clip in the sequence and tracking continues in full physics with the trained
policy. No pelvis anchoring — the robot balances on its own.

Usage:
  uv run python -m wbc_mjlab.scripts.chain \
    --motion-source data/g1/hackathon --chain walk step_touch bow \
    --viewer none
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner

DEFAULT_TASK_ID = "Wbc-G1"
BUNDLED_CHECKPOINT = "demos/wbc_g1/model.pt"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--checkpoint-file",
    type=str,
    default=None,
    help="Policy checkpoint (.pt). Defaults to bundled demos/wbc_g1/model.pt.",
  )
  parser.add_argument(
    "--task", type=str, default=DEFAULT_TASK_ID, help=f"Task id (default {DEFAULT_TASK_ID})."
  )
  parser.add_argument(
    "--motion-source",
    type=str,
    required=True,
    help="Dataset dir (containing npz/*.npz) or stacked .npz bundle.",
  )
  parser.add_argument(
    "--chain",
    type=str,
    nargs="+",
    default=[],
    help="Ordered clip name fragments matched against library stems. "
    "Empty = play the whole library in order.",
  )
  parser.add_argument("--loops", type=int, default=1, help="Repeat the chain N times.")
  parser.add_argument("--num-envs", type=int, default=1, help="Parallel envs.")
  parser.add_argument(
    "--max-steps", type=int, default=20000, help="Safety cap on total env steps."
  )
  parser.add_argument(
    "--log-interval", type=int, default=50, help="Steps between metric prints."
  )
  parser.add_argument(
    "--export-models",
    type=str,
    default=None,
    help="Directory to write deploy artifacts (policy.onnx + config.yaml).",
  )
  return parser.parse_args()


def repo_root() -> Path:
  from wbc_mjlab.data_paths import repo_root as _repo_root

  return _repo_root()


def resolve_checkpoint(explicit: str | None) -> Path:
  if explicit:
    p = Path(explicit).expanduser()
    if not p.is_file():
      raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p
  for candidate in (Path.cwd() / BUNDLED_CHECKPOINT, repo_root() / BUNDLED_CHECKPOINT):
    if candidate.is_file():
      return candidate.resolve()
  raise FileNotFoundError(
    f"No checkpoint given and bundled {BUNDLED_CHECKPOINT} not found. "
    "Pass --checkpoint-file."
  )


def match_clip_fragments(cmd, fragments: list[str]) -> list[int]:
  """Map ordered CLI fragments onto trajectory ids in the stacked clip library."""
  stems = [Path(str(name)).stem.lower() for name in cmd.motion.segment_names]
  traj_ids: list[int] = []
  for frag in fragments:
    frag_l = frag.lower()
    matches = [tid for tid, stem in enumerate(stems) if frag_l in stem or stem in frag_l]
    if not matches:
      known = "\n  ".join(sorted(set(stems)))
      raise SystemExit(
        f"Clip fragment {frag!r} not found in library. Known stems:\n  {known}"
      )
    traj_ids.append(matches[0])
  return traj_ids


def export_deploy_artifacts(
  runner: MjlabOnPolicyRunner,
  env: ManagerBasedRlEnv,
  out_dir: Path,
) -> None:
  """Write policy.onnx + config.yaml (+ motion library manifest) to out_dir."""
  from wbc_mjlab.export.policy_bundle import export_deploy_params
  from wbc_mjlab.tasks import last_registered_robot_id

  params_dir = out_dir / "params"
  params_dir.mkdir(parents=True, exist_ok=True)
  runner.export_policy_to_onnx(str(params_dir), "policy.onnx")
  export_deploy_params(
    out_dir,
    env.unwrapped.cfg,
    env.unwrapped,
    robot_id=last_registered_robot_id(),
    write_motion_library=True,
  )
  for f in params_dir.glob("*"):
    if f.is_file():
      shutil.copy2(f, out_dir / f.name)
  print(f"[chain] Deploy artifacts written to {out_dir.resolve()}")


def main() -> None:
  args = parse_args()
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  from wbc_mjlab.tasks import prepare_wbc_run, last_registered_robot_id

  prepare_wbc_run(task_id=args.task)

  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)

  motion_cfg = env_cfg.commands["motion"]
  motion_cfg.motion_file = str(Path(args.motion_source).expanduser().resolve())
  # Deterministic chaining: internal resets restart the same clip from frame 0.
  # Clip-to-clip switches are driven explicitly below.
  motion_cfg.rsi.sampling_mode = "start"
  env_cfg.scene.num_envs = args.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  checkpoint = resolve_checkpoint(args.checkpoint_file)
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=env.device)

  cmd = env.unwrapped.command_manager.get_term("motion")
  env_ids = torch.arange(args.num_envs, device=env.device)

  if args.export_models:
    export_deploy_artifacts(
      runner, env, Path(args.export_models).expanduser().resolve()
    )

  stems = [Path(str(name)).stem.lower() for name in cmd.motion.segment_names]
  if not args.chain:
    chain_traj_ids = list(range(cmd.motion.num_trajectories))
  else:
    chain_traj_ids = match_clip_fragments(cmd, args.chain)

  node_frames = [int(cmd.motion.segment_start_idx[t].item()) for t in chain_traj_ids]
  node_names = [Path(str(cmd.motion.segment_names[t])).stem for t in chain_traj_ids]

  print("=" * 60)
  print("WBC Chained Clip Runner")
  print(f"  Checkpoint : {checkpoint}")
  print(f"  Motion src : {motion_cfg.motion_file}")
  print(f"  Chain      : {' -> '.join(node_names)} x{args.loops}")
  print(f"  Device     : {env.device}")
  print("=" * 60)

  def force_node_start(node_idx: int) -> None:
    traj = chain_traj_ids[node_idx % len(chain_traj_ids)]
    cmd.reset_to_frame(env_ids, int(cmd.motion.segment_start_idx[traj].item()))
    cmd.update_relative_body_poses()

  wrapped.reset()
  node_idx = 0
  force_node_start(0)

  step = 0
  try:
    while step < args.max_steps:
      obs = wrapped.get_observations()
      with torch.no_grad():
        action = policy(obs)
      wrapped.step(action)
      step += 1

      traj = int(cmd.trajectory_ids[0].item())
      tstep = int(cmd.time_steps[0].item())

      # Fall recovery: enforce this node's clip start deterministically.
      terminated = bool(env.unwrapped.termination_manager.terminated.any())
      if terminated:
        name = node_names[node_idx % len(node_names)]
        print(f"[chain] episode terminated on clip {name!r}; restarting clip.")
        force_node_start(node_idx % len(chain_traj_ids))

      # Advance to the next clip just before the active one ends.
      seg_end = int(cmd.motion.segment_end_idx[traj].item())
      if int(cmd.time_steps[0].item()) >= seg_end - 1:
        node_idx += 1
        if node_idx >= len(chain_traj_ids) * args.loops:
          print("[chain] sequence complete.")
          break
        traj = chain_traj_ids[node_idx % len(chain_traj_ids)]
        cmd.reset_to_frame(env_ids, int(cmd.motion.segment_start_idx[traj].item()))
        cmd.update_relative_body_poses()
        print(
          f"[chain] node {node_idx + 1}/{len(chain_traj_ids) * args.loops}: "
          f"{Path(str(cmd.motion.segment_names[traj])).stem}"
        )

      if step % args.log_interval == 0:
        err_pos = float(cmd.metrics["error_anchor_pos"].mean().item())
        err_rot = float(cmd.metrics["error_anchor_rot"].mean().item())
        name = Path(str(cmd.motion.segment_names[traj])).stem
        print(
          f"step {step:06d} | clip {name:<28} | frame {tstep:5d} | "
          f"anchor_pos_err {err_pos:.3f} m | anchor_rot_err {err_rot:.2f} rad"
        )

    print("[chain] done.")
  finally:
    env.close()


if __name__ == "__main__":
  main()
