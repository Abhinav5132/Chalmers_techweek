"""
Workflow Execution Script for Unitree G1.

Runs a sequence of RL-tracked motion clips (or a JSON workflow) through the
exported WBC policy in full MuJoCo physics. The robot balances on its own — no
pelvis anchoring and no hard-coded default sequence. The sequence is given via
`--chain walk step_touch bow` or `--workflow-json`.

Requires an exported policy bundle under models/params/ (see `just wbc-export`).
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.controllers.pink_controller import PinkG1Controller
from src.controllers.wbc_runner import WbcPhysicsRunner
from src.engine.workflow_engine import (
    WorkflowContext,
    WorkflowEngine,
    MotionClipNode,
    PinkReachNode,
    StandHoldNode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unitree G1 Workflow Scripting Runner")
    parser.add_argument(
        "--scene",
        type=str,
        default="unitree_mujoco/unitree_robots/g1/scene_29dof.xml",
        help="Path to MuJoCo scene XML.",
    )
    parser.add_argument(
        "--workflow-json",
        type=str,
        default=None,
        help="Path to JSON workflow file.",
    )
    parser.add_argument(
        "--chain",
        type=str,
        nargs="+",
        default=[],
        help="Ordered motion clips to chain, e.g. 'walk step_touch bow'.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless for automated testing/CI.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200000,
        help="Safety cap on total policy steps (headless mode).",
    )
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()

    if not args.chain and not args.workflow_json:
        print("Error: provide --chain <clips...> or --workflow-json <file>.")
        sys.exit(1)

    if not os.path.exists(args.scene):
        print(f"Error: Scene file not found at '{args.scene}'.")
        sys.exit(1)

    mj_model: Any = mj.MjModel.from_xml_path(args.scene)
    mj_data: Any = mj.MjData(mj_model)

    robot_xml: str = "unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
    pink_ctrl = PinkG1Controller(mj_model, mj_data, xml_path=robot_xml)
    wbc_runner = WbcPhysicsRunner(mj_model, mj_data)

    context = WorkflowContext(mj_model, mj_data, wbc_runner, pink_ctrl)
    engine = WorkflowEngine(context)

    if args.workflow_json:
        with open(args.workflow_json, "r") as f:
            engine.load_json(f.read())
        print(f"Loaded workflow from '{args.workflow_json}' with {len(engine.nodes)} nodes.")
    else:
        for clip_name in args.chain:
            engine.add_node(MotionClipNode(clip_name=clip_name))
        print(f"Constructed chain of {len(engine.nodes)} clips: {list(args.chain)}")

    # One engine.step == one policy period (runner steps physics internally).
    policy_dt: float = wbc_runner.policy_step_dt
    has_display: bool = (
        not args.headless
        and ("DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ)
        and VIEWER_AVAILABLE
    )

    if has_display:
        print("\nLaunching interactive viewer. Close window to exit.")
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            while viewer.is_running() and not engine.is_finished:
                step_start = time.time()
                engine.step(dt=policy_dt)
                viewer.sync()
                elapsed = time.time() - step_start
                if policy_dt > elapsed:
                    time.sleep(policy_dt - elapsed)
    else:
        print("\nRunning workflow in headless mode...")
        step_i = 0
        while not engine.is_finished and step_i < args.max_steps:
            engine.step(dt=policy_dt)
            if step_i % 1000 == 0:
                name = engine.current_node.name if engine.current_node else "Done"
                pos = mj_data.qpos[0:3]
                print(
                    f"Step {step_i:05d} | Node: {name:<22} | "
                    f"Root: [{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:.2f}]"
                )
            step_i += 1

    print("\nWorkflow runner finished.")


if __name__ == "__main__":
    main()
