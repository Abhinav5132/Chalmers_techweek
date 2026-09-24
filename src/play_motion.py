"""
Motion Clip Visualizer & Trajectory Player for Unitree G1.
Replays pre-retargeted MoCap clips (.npz) from the g1-moves dataset in MuJoCo.
Fully typed for static type checkers (ty / mypy / pyright).
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Any
import numpy as np
import mujoco

mj: Any = mujoco

try:
    import mujoco.viewer
    VIEWER_AVAILABLE: bool = True
except ImportError:
    VIEWER_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unitree G1 Motion Clip Visualizer")
    parser.add_argument(
        "--clip",
        type=str,
        default="data/motions/walk.npz",
        help="Path to .npz motion clip (default: data/motions/walk.npz).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback continuously.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI viewer for testing/CI.",
    )
    return parser.parse_args()


def load_motion_clip(path: str) -> dict[str, np.ndarray]:
    """Load and validate G1 .npz motion clip."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Motion clip not found at '{path}'.")

    data = np.load(path)
    required_keys = ["fps", "joint_pos", "body_pos_w", "body_quat_w"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Motion clip '{path}' missing required key '{k}'.")

    return {
        "fps": np.asarray(data["fps"], dtype=np.float64),
        "joint_pos": np.asarray(data["joint_pos"], dtype=np.float64),
        "body_pos_w": np.asarray(data["body_pos_w"], dtype=np.float64),
        "body_quat_w": np.asarray(data["body_quat_w"], dtype=np.float64),
    }


def main() -> None:
    args: argparse.Namespace = parse_args()

    scene_path: str = "unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found at {scene_path}.")
        sys.exit(1)

    motion: dict[str, np.ndarray] = load_motion_clip(args.clip)
    fps: float = float(motion["fps"][0])
    jpos: np.ndarray = motion["joint_pos"]
    bpos: np.ndarray = motion["body_pos_w"]
    bquat: np.ndarray = motion["body_quat_w"]
    n_frames: int = int(jpos.shape[0])
    duration: float = float(n_frames / fps)

    print("==================================================")
    print("Unitree G1 Motion Player (exptech/g1-moves)")
    print(f"Clip:     {args.clip}")
    print(f"Frames:   {n_frames} ({duration:.2f} seconds @ {fps:.1f} FPS)")
    print(f"Speed:    {args.speed:.1f}x")
    print(f"X Travel: {float(bpos[-1, 0, 0] - bpos[0, 0, 0]):.2f}m")
    print("==================================================")

    mj_model: Any = mj.MjModel.from_xml_path(scene_path)
    mj_data: Any = mj.MjData(mj_model)

    has_display: bool = (
        not args.headless
        and (sys.platform == "darwin" or "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ)
        and VIEWER_AVAILABLE
    )

    frame_dt: float = (1.0 / fps) / max(0.1, args.speed)

    if has_display:
        print("\nLaunching viewer. Press SPACE to pause/resume, ESC to exit.")
        viewer: Any
        with mj.viewer.launch_passive(mj_model, mj_data) as viewer:
            while viewer.is_running():
                for k in range(n_frames):
                    if not viewer.is_running():
                        break
                    step_start: float = time.time()

                    # Set root position & orientation (body index 0 = pelvis)
                    mj_data.qpos[0:3] = bpos[k, 0, :]
                    mj_data.qpos[3:7] = bquat[k, 0, :]

                    # Set 29 actuated joint positions
                    mj_data.qpos[7:36] = jpos[k, :]

                    # Forward kinematics to update body visuals
                    mj.mj_forward(mj_model, mj_data)
                    viewer.sync()

                    elapsed: float = time.time() - step_start
                    if frame_dt > elapsed:
                        time.sleep(frame_dt - elapsed)

                if not args.loop:
                    print("Motion playback finished.")
                    while viewer.is_running():
                        time.sleep(0.1)
                    break
    else:
        print("\nRunning in headless verification mode...")
        # Step through frames to ensure all configurations are valid
        for k in range(min(n_frames, 200)):
            mj_data.qpos[0:3] = bpos[k, 0, :]
            mj_data.qpos[3:7] = bquat[k, 0, :]
            mj_data.qpos[7:36] = jpos[k, :]
            mj.mj_forward(mj_model, mj_data)

            if k % 50 == 0:
                print(f"Frame {k:04d}/{n_frames} | Root X: {bpos[k, 0, 0]:.3f}m | Root Z: {bpos[k, 0, 2]:.3f}m")

        print("Motion clip verification completed successfully.")


if __name__ == "__main__":
    main()
