"""
Test Script: Unitree G1 Standing Controller.
Demonstrates holding a stable calibrated standing posture in MuJoCo.
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

# MuJoCo is a compiled C-extension (_structs.so).
# We alias to Any so static type checkers (like ty) do not flag missing members on the C-module.
mj: Any = mujoco

try:
    import mujoco.viewer
    VIEWER_AVAILABLE: bool = True
except ImportError:
    VIEWER_AVAILABLE = False

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.controllers.pink_controller import PinkG1Controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unitree G1 Kinematics & Reaching Controller Test")
    parser.add_argument(
        "--stand",
        action="store_true",
        help="Only hold standing stance without arm reaching.",
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        default=[0.25, -0.20, 0.75],
        help="Target 3D position [X Y Z] in world coordinates for right wrist (default: 0.25 -0.20 0.75).",
    )
    parser.add_argument(
        "--anchor",
        action="store_true",
        default=True,
        help="Anchor pelvis in place (simulate testing gantry / harness).",
    )
    parser.add_argument(
        "--no-anchor",
        dest="anchor",
        action="store_false",
        help="Simulate unanchored floating-base (requires active balance).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without opening GUI viewer window.",
    )
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()

    scene_path: str = "unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
    robot_xml: str = "unitree_mujoco/unitree_robots/g1/g1_29dof.xml"

    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found at {scene_path}. Ensure unitree_mujoco is set up.")
        sys.exit(1)

    target_pos: np.ndarray = np.array(args.target, dtype=np.float64)

    print("==================================================")
    print("Unitree G1 Kinematics & Reaching Controller")
    if args.stand:
        print("Mode: STABLE STAND ONLY")
    else:
        print(f"Mode: PINK QP ARM REACH -> Target: {target_pos}")
    print(f"Pelvis Anchor (Test Gantry Harness): {args.anchor}")
    print("==================================================")

    mj_model: Any = mj.MjModel.from_xml_path(scene_path)
    mj_data: Any = mj.MjData(mj_model)

    # Initialize controller (hard crashes if Pinocchio, Pink, or QP solver is missing)
    controller: PinkG1Controller = PinkG1Controller(mj_model, mj_data, xml_path=robot_xml)
    print(f"Controller initialized with Pink QP solver: {controller.solver}")

    # Set calibrated initial posture using exact rest height (zero floor penetration)
    pelvis_z: float = controller.rest_pelvis_z
    print(f"Calibrated standing pelvis height: {pelvis_z:.4f}m (exact zero-penetration ground contact)")
    mj_data.qpos[0] = 0.0
    mj_data.qpos[1] = 0.0
    mj_data.qpos[2] = pelvis_z
    mj_data.qpos[3] = 1.0
    mj_data.qpos[4:7] = 0.0

    # Initialize actuated joints with calibrated standing angles
    mj_data.qpos[7 : 7 + controller.n_ctrl] = controller.q_nominal_29.copy()
    mj_data.ctrl[:] = controller.compute_pd_torques(controller.q_nominal_29)

    # Step once to establish contacts
    mj.mj_step(mj_model, mj_data)

    hand_name: str = "right_wrist_yaw_link"
    hand_id: int = int(mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, hand_name))

    # Check if a display is available for interactive viewer
    has_display: bool = (
        not args.headless
        and (sys.platform == "darwin" or "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ)
        and VIEWER_AVAILABLE
    )

    if has_display:
        print("\nLaunching MuJoCo passive viewer. Press ESC or close window to exit.")
        viewer: Any
        with mj.viewer.launch_passive(mj_model, mj_data) as viewer:
            dt: float = float(mj_model.opt.timestep)

            while viewer.is_running():
                step_start: float = time.time()

                if args.anchor:
                    # Enforce testing gantry / harness anchor on pelvis
                    mj_data.qpos[0:3] = [0.0, 0.0, pelvis_z]
                    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                    mj_data.qvel[0:6] = 0.0

                if args.stand:
                    torques: np.ndarray = controller.compute_pd_torques(controller.q_nominal_29)
                    mj_data.ctrl[:] = torques
                else:
                    # Reach directly toward target 3D point using Pink QP
                    controller.step_reach(target_pos, body_name=hand_name, dt=dt)

                # Step physics
                mj.mj_step(mj_model, mj_data)

                # Sync viewer
                viewer.sync()

                # Real-time sync
                elapsed: float = time.time() - step_start
                if dt > elapsed:
                    time.sleep(dt - elapsed)
    else:
        print("\nRunning 500 steps in headless mode...")
        dt_val: float = float(mj_model.opt.timestep)
        for step in range(500):
            if args.anchor:
                mj_data.qpos[0:3] = [0.0, 0.0, pelvis_z]
                mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                mj_data.qvel[0:6] = 0.0

            if args.stand:
                torques_val: np.ndarray = controller.compute_pd_torques(controller.q_nominal_29)
                mj_data.ctrl[:] = torques_val
            else:
                controller.step_reach(target_pos, body_name=hand_name, dt=dt_val)

            mj.mj_step(mj_model, mj_data)

            if step % 100 == 0 or step == 499:
                current_hand = np.asarray(mj_data.xpos[hand_id], dtype=np.float64)
                err_cm = float(np.linalg.norm(target_pos - current_hand) * 100.0)
                print(f"Step {step:04d} | Hand: {current_hand.round(3)} | Target: {target_pos.round(3)} | Error: {err_cm:.2f} cm")

        print("Headless simulation completed successfully.")


if __name__ == "__main__":
    main()

