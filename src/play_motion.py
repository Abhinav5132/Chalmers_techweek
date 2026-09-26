"""
Motion Clip Visualizer & Trajectory Player for Unitree G1.
Replays pre-retargeted MoCap clips (.npz) or scripted multi-clip workflows in MuJoCo.
Strips all viewer UI clutter to deliver an extremely clean 3D simulation screen.
Fully typed for static type checkers (ty / mypy / pyright).
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Any

if os.environ.get("__GLX_VENDOR_LIBRARY_NAME") == "nvidia":
    os.environ.pop("__GLX_VENDOR_LIBRARY_NAME", None)

import numpy as np
import mujoco

mj: Any = mujoco

try:
    import mujoco.viewer
    VIEWER_AVAILABLE: bool = True
except ImportError:
    VIEWER_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unitree G1 Motion Clip Visualizer & Sequence Player")
    parser.add_argument(
        "--clip",
        type=str,
        default="data/motions/walk.npz",
        help="Path to .npz motion clip (default: data/motions/walk.npz).",
    )
    parser.add_argument(
        "--chain",
        type=str,
        nargs="+",
        default=[],
        help="Sequence of clips to chain, e.g. --chain walk step_touch bow.",
    )
    parser.add_argument(
        "--workflow-json",
        type=str,
        default=None,
        help="Path to workflow JSON file specifying sequence of clips and in-node delays.",
    )
    parser.add_argument(
        "--delay-between",
        type=float,
        default=0.5,
        help="Default time between clips in seconds (default: 0.5s).",
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
    ui_group = parser.add_mutually_exclusive_group()
    ui_group.add_argument(
        "--show-ui",
        dest="show_ui",
        action="store_true",
        default=True,
        help="Show MuJoCo viewer UI panels and sidebars (default).",
    )
    ui_group.add_argument(
        "--hide-ui",
        dest="show_ui",
        action="store_false",
        help="Hide MuJoCo viewer UI panels/sidebars for a stripped, clean view.",
    )
    vsync_group = parser.add_mutually_exclusive_group()
    vsync_group.add_argument(
        "--no-vsync",
        dest="no_vsync",
        action="store_true",
        default=True,
        help="Disable VSync in the MuJoCo viewer for maximum/uncapped framerate (default).",
    )
    vsync_group.add_argument(
        "--vsync",
        dest="no_vsync",
        action="store_false",
        help="Enable VSync in the MuJoCo viewer.",
    )
    follow_group = parser.add_mutually_exclusive_group()
    follow_group.add_argument(
        "--follow",
        dest="follow",
        action="store_true",
        default=True,
        help="Camera automatically follows the robot (default).",
    )
    follow_group.add_argument(
        "--no-follow",
        dest="follow",
        action="store_false",
        help="Keep camera static at initial origin.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI viewer for testing/CI.",
    )
    return parser.parse_args()


def resolve_clip_path(clip_name_or_path: str) -> str:
    """Resolve clip name to file path."""
    if os.path.exists(clip_name_or_path):
        return clip_name_or_path
    c1 = os.path.join("data", "motions", f"{clip_name_or_path}.npz")
    if os.path.exists(c1):
        return c1
    c2 = os.path.join("data", "motions", clip_name_or_path)
    if os.path.exists(c2):
        return c2
    return clip_name_or_path


def load_motion_clip(path: str) -> dict[str, np.ndarray]:
    """Load and validate G1 .npz motion clip."""
    resolved = resolve_clip_path(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Motion clip not found at '{resolved}'.")

    data = np.load(resolved, allow_pickle=True)
    required_keys = ["fps", "joint_pos", "body_pos_w", "body_quat_w"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Motion clip '{resolved}' missing required key '{k}'.")

    return {
        "fps": np.asarray(data["fps"], dtype=np.float64),
        "joint_pos": np.asarray(data["joint_pos"], dtype=np.float64),
        "body_pos_w": np.asarray(data["body_pos_w"], dtype=np.float64),
        "body_quat_w": np.asarray(data["body_quat_w"], dtype=np.float64),
    }


def main() -> None:
    args: argparse.Namespace = parse_args()

    if args.no_vsync:
        os.environ["__GL_SYNC_TO_VBLANK"] = "0"
        os.environ["vblank_mode"] = "0"
        hook_path = Path(__file__).resolve().parent / "gui" / "libnovsync.so"
        c_src = Path(__file__).resolve().parent / "gui" / "novsync.c"
        if not hook_path.exists() and c_src.exists():
            try:
                import subprocess
                subprocess.run(
                    ["gcc", "-shared", "-fPIC", "-O2", "-o", str(hook_path), str(c_src), "-ldl"],
                    check=False,
                )
            except Exception:
                pass
        if hook_path.exists() and str(hook_path) not in os.environ.get("LD_PRELOAD", ""):
            cur_preload = os.environ.get("LD_PRELOAD", "")
            os.environ["LD_PRELOAD"] = f"{hook_path}:{cur_preload}" if cur_preload else str(hook_path)
            if not os.environ.get("_NO_VSYNC_REEXEC"):
                os.environ["_NO_VSYNC_REEXEC"] = "1"
                os.execv(sys.executable, [sys.executable] + sys.argv)

    scene_path: str = "unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found at {scene_path}.")
        sys.exit(1)

    mj_model: Any = mj.MjModel.from_xml_path(scene_path)
    mj_data: Any = mj.MjData(mj_model)

    # Determine execution sequence items
    # Each item: {"clip_path": str, "delay_after": float, "speed": float}
    items: list[dict[str, Any]] = []

    if args.workflow_json and os.path.exists(args.workflow_json):
        with open(args.workflow_json, "r") as f:
            obj = json.load(f)
        for node in obj.get("workflow", []):
            ntype = node.get("type")
            if ntype == "motion_clip":
                items.append({
                    "clip_path": resolve_clip_path(node["clip_name"]),
                    "delay_after": float(node.get("delay_after", 0.0)),
                    "speed": float(node.get("speed", args.speed)),
                    "name": node.get("name", node["clip_name"]),
                })
            elif ntype == "delay":
                items.append({
                    "clip_path": None,
                    "delay_after": float(node.get("duration", 1.0)),
                    "speed": 1.0,
                    "name": "Delay",
                })
    elif args.chain:
        for cname in args.chain:
            items.append({
                "clip_path": resolve_clip_path(cname),
                "delay_after": args.delay_between,
                "speed": args.speed,
                "name": cname,
            })
    else:
        items.append({
            "clip_path": resolve_clip_path(args.clip),
            "delay_after": 0.0,
            "speed": args.speed,
            "name": os.path.basename(args.clip),
        })

    print("==================================================")
    print(f"Unitree G1 Motion Player (Sequence of {len(items)} items)")
    for i, it in enumerate(items):
        print(f"  [{i+1}] {it['name']} (Delay after: {it['delay_after']:.2f}s, Speed: {it['speed']:.1f}x)")
    print("==================================================")

    has_display: bool = (
        not args.headless
        and ("DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ)
        and VIEWER_AVAILABLE
    )

    if has_display:
        mode_str = f"UI: {'Shown' if args.show_ui else 'Stripped'}, VSync: {'Off' if args.no_vsync else 'On'}, Follow: {'On' if args.follow else 'Off'}"
        print(f"\nLaunching 3D simulation player ({mode_str})...")
        with mj.viewer.launch_passive(
            mj_model,
            mj_data,
            show_left_ui=args.show_ui,
            show_right_ui=args.show_ui,
        ) as viewer:
            # Set camera to follow robot if requested
            if args.follow:
                try:
                    pelvis_id = mj_model.body("pelvis").id
                    viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = pelvis_id
                    viewer.cam.distance = 3.0
                    viewer.cam.elevation = -12.0
                    viewer.cam.azimuth = 90.0
                except Exception:
                    pass

            # Strip all sidebars, debug panels, and text overlays if not requested
            if not args.show_ui:
                try:
                    sim = viewer._get_sim()
                    sim.ui0_enable = False
                    sim.ui1_enable = False
                    sim.clear_texts()
                except Exception:
                    pass

            while viewer.is_running():
                for item_idx, it in enumerate(items):
                    if not viewer.is_running():
                        break

                    clip_path = it["clip_path"]
                    speed = max(0.1, it["speed"])
                    delay_after = it["delay_after"]

                    if clip_path is not None and os.path.exists(clip_path):
                        motion = load_motion_clip(clip_path)
                        fps = float(motion["fps"][0])
                        jpos = motion["joint_pos"]
                        bpos = motion["body_pos_w"]
                        bquat = motion["body_quat_w"]
                        n_frames = int(jpos.shape[0])
                        frame_dt = (1.0 / fps) / speed

                        # Play clip frames
                        for k in range(n_frames):
                            if not viewer.is_running():
                                break
                            step_start = time.time()

                            mj_data.qpos[0:3] = bpos[k, 0, :]
                            mj_data.qpos[3:7] = bquat[k, 0, :]
                            n_j = min(len(mj_data.qpos) - 7, jpos.shape[1])
                            mj_data.qpos[7 : 7 + n_j] = jpos[k, :n_j]

                            mj.mj_forward(mj_model, mj_data)

                            # If camera was switched to free mode, keep lookat centered on robot
                            if args.follow and viewer.cam.type == mj.mjtCamera.mjCAMERA_FREE:
                                viewer.cam.lookat[0:3] = bpos[k, 0, :]

                            viewer.sync()

                            elapsed = time.time() - step_start
                            if frame_dt > elapsed:
                                time.sleep(frame_dt - elapsed)

                    # Transition delay: hold pose for delay_after seconds
                    if delay_after > 0.0 and viewer.is_running():
                        t_end = time.time() + delay_after
                        while time.time() < t_end and viewer.is_running():
                            viewer.sync()
                            time.sleep(0.02)

                if not args.loop:
                    print("Playback completed. Close window to exit.")
                    while viewer.is_running():
                        time.sleep(0.1)
                    break
    else:
        print("\nRunning in headless verification mode...")
        for it in items:
            clip_path = it["clip_path"]
            if clip_path and os.path.exists(clip_path):
                motion = load_motion_clip(clip_path)
                print(f"Verified clip: {it['name']} ({len(motion['joint_pos'])} frames)")
        print("Sequence verification completed successfully.")


if __name__ == "__main__":
    main()
