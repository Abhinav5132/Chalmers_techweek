"""Local MCP tools for replaying repository motion clips in MuJoCo."""
from __future__ import annotations

import atexit
import json
import queue
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS = {
    "walk": "Recorded jazz-walk motion; pose playback, not balanced locomotion.",
    "step_touch": "Recorded side-step / step-touch motion.",
    "bow": "Recorded karate bow motion.",
}


class MotionPlayer:
    def __init__(self, root: Path = ROOT, headless: bool = False):
        self.root = root
        self.directory = (root / "data/motions").resolve()
        self.headless = headless
        self.process: subprocess.Popen | None = None
        self.current: dict[str, Any] = {}
        self.responses: queue.Queue = queue.Queue()
        self.lock = threading.RLock()
        self.log_path = root / ".robot-runtime/playback.log"

    def path(self, motion: str) -> Path:
        # Only clip IDs, never arbitrary paths or commands supplied by the model.
        if not motion or Path(motion).name != motion or motion in (".", ".."):
            raise ValueError("Use a motion ID returned by list_motions.")
        path = (self.directory / (motion + ".npz")).resolve()
        if not path.is_relative_to(self.directory) or not path.is_file():
            raise ValueError(f"Motion {motion!r} is unavailable; call list_motions.")
        return path

    def describe(self, motion: str) -> dict[str, Any]:
        with np.load(self.path(motion), allow_pickle=False) as clip:
            fps = np.asarray(clip["fps"]).reshape(-1)
            joints = clip["joint_pos"]
            pos, quat = clip["body_pos_w"], clip["body_quat_w"]
            if fps.size != 1 or not np.isfinite(fps[0]) or fps[0] <= 0:
                raise ValueError("Invalid motion frame rate.")
            if joints.ndim != 2 or joints.shape[1] != 29 or len(joints) == 0:
                raise ValueError("Motion must contain frames with 29 joint positions.")
            if (pos.ndim != 3 or quat.ndim != 3 or pos.shape[0] != len(joints)
                    or quat.shape[0] != len(joints) or pos.shape[1] < 1
                    or quat.shape[1] < 1 or pos.shape[2] != 3 or quat.shape[2] != 4):
                raise ValueError("Invalid body position/orientation frames.")
            if not all(np.isfinite(a).all() for a in (joints, pos, quat)):
                raise ValueError("Motion contains non-finite values.")
            if np.any(np.linalg.norm(quat[:, 0], axis=1) < 1e-6):
                raise ValueError("Motion contains a zero root quaternion.")
            return {"id": motion, "description": DESCRIPTIONS.get(motion, "Local motion clip."),
                    "frames": len(joints), "duration_seconds": len(joints) / float(fps[0]),
                    "mode": "kinematic_playback"}

    def list(self) -> dict[str, Any]:
        available, invalid = [], []
        for path in sorted(self.directory.glob("*.npz")):
            try:
                available.append(self.describe(path.stem))
            except (ValueError, KeyError, OSError) as exc:
                invalid.append({"id": path.stem, "error": str(exc)})
        return {"motions": available, "invalid": invalid,
                "note": "Simulation recordings only; no real robot or learned balance control."}

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        binary = "mjpython" if sys.platform == "darwin" and not self.headless else "python"
        command = [str(self.root / ".venv/bin" / binary), str(self.root / "src/robot_viewer.py")]
        if self.headless:
            command.append("--headless")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        responses: queue.Queue = queue.Queue()
        self.responses = responses
        with self.log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=self.root, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=log, text=True,
                                       bufsize=1, start_new_session=True)
        self.process = process

        def read_responses():
            assert process.stdout is not None
            for line in process.stdout:
                try:
                    responses.put(json.loads(line))
                except ValueError:
                    continue
            responses.put({"error": "Robot window closed or viewer process exited."})

        threading.Thread(target=read_responses, daemon=True).start()
        try:
            ready = responses.get(timeout=20)
            if not ready.get("ready"):
                raise RuntimeError(ready.get("error", "Viewer failed to start"))
        except Exception:
            self.close()
            raise

    def _request(self, action: str, **values) -> dict[str, Any]:
        assert self.process is not None and self.process.stdin is not None
        try:
            self.process.stdin.write(json.dumps(dict(action=action, **values)) + "\n")
            self.process.stdin.flush()
            result = self.responses.get(timeout=10)
            if result.get("error"):
                raise RuntimeError(result["error"])
            self.current = result
            return result
        except (BrokenPipeError, queue.Empty, RuntimeError) as exc:
            self.close()
            raise RuntimeError(f"Viewer communication failed: {exc}. See {self.log_path}") from exc

    def status(self) -> dict[str, Any]:
        with self.lock:
            if self.process is None:
                return {"state": "idle", "viewer_open": False}
            code = self.process.poll()
            if code is None:
                return self._request("status")
            result = {"state": "closed" if code == 0 else "failed", "viewer_open": False,
                      "exit_code": code}
            if code and self.log_path.exists():
                result["output"] = self.log_path.read_text(errors="replace")[-3000:]
            return result

    def play(self, motion: str, speed: float = 1.0, loop: bool = False) -> dict[str, Any]:
        with self.lock:
            if not math.isfinite(speed) or not 0.25 <= speed <= 2:
                raise ValueError("Playback speed must be between 0.25 and 2.0.")
            self.describe(motion)
            self._start()
            return self._request("play", motion=motion, path=str(self.path(motion)), speed=speed, loop=loop)

    def physics(self, task: str, controller: str = 'student', seconds: float = 10., push_force: float = 0.) -> dict[str, Any]:
        from physics_session import validate, saved_policy
        validate(task, controller, seconds, push_force)
        if controller == 'student':
            saved_policy(self.root)
        with self.lock:
            self._start()
            return self._request('physics', task=task, controller=controller, seconds=seconds, push_force=push_force)

    def stop(self) -> dict[str, Any]:
        """Freeze the current pose, keeping the same window available."""
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                return self.status()
            return self._request("stop")

    def close(self) -> None:
        """Close the viewer worker; a subsequent play command can reopen it."""
        with self.lock:
            process = self.process
            if process is None:
                return
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
                except ProcessLookupError:
                    pass
            if process.stdout is not None:
                process.stdout.close()
            self.process = None


player = MotionPlayer(headless=os.environ.get("ROBOT_MCP_HEADLESS") == "1")
mcp = FastMCP("g1-robot")


@mcp.tool()
def list_motions() -> dict[str, Any]:
    """List validated local robot recordings. Call before choosing a movement."""
    return player.list()


@mcp.tool()
def list_physics_skills() -> dict[str, Any]:
    """List saved learned standing/walking controllers and teacher availability."""
    from physics_session import catalog
    return catalog()


@mcp.tool()
def run_physics_motion(task: str, controller: str = 'student', seconds: float = 10.,
                       push_force: float = 0.) -> dict[str, Any]:
    """Run learned stand/walk using motor torques, gravity and ground contact.

    Use this for ordinary standing/walking requests. controller is student or teacher.
    Reuses the existing window. Each command resets a trial, not a learned transition.
    Duration 1–10 seconds; optional 0–100 N pushes last 0.2 s at seconds 2 and 6.
    Query playback_status for time, falls and completion. A paused/completed viewer
    freezes physics, not active balance. No training or real robot commands occur.
    This follows a reference; requested step length is not supported.
    """
    return player.physics(task, controller, seconds, push_force)


@mcp.tool()
def play_motion(motion: str, speed: float = 1.0, loop: bool = False) -> dict[str, Any]:
    """Play a listed motion ID in MuJoCo. Speed changes playback time, not step length.

    Reuses the same window and replaces any current motion. The window stays open on completion.
    Returns startup status, not proof of completion. Check playback_status afterward.
    """
    return player.play(motion, speed, loop)


@mcp.tool()
def play_larger_steps(scale: float = 1.25, loop: bool = False) -> dict[str, Any]:
    """Make the recorded walk visibly take larger steps and show it in the same window.

    scale is 1.05–1.5; 1.25 requests 25% more fore/aft foot reach and root travel.
    Leg inverse kinematics respects joint limits; report achieved separation and errors.
    This modifies a recording, not physical walking control or ground-contact step length.
    Use this tool when the user asks for bigger/longer steps, not playback speed or max_steps.
    """
    from longer_steps import make_longer_walk
    report = make_longer_walk(scale)
    report['playback'] = player.play(report['motion'], loop=loop)
    return report


@mcp.tool()
def playback_status() -> dict[str, Any]:
    """Read whether this session's playback is running, completed, or failed."""
    return player.status()


@mcp.tool()
def stop_motion() -> dict[str, Any]:
    """Pause movement and hold the current pose when asked to keep the window open.

    For an ordinary stop or close request, use close_robot_window instead.
    """
    return player.stop()


@mcp.tool()
def close_robot_window() -> dict[str, Any]:
    """Stop playback and close the robot simulation window. Use when asked to stop,
    close the window, or end the robot session. Hermes chat remains available;
    a subsequent play_motion opens a new viewer. Safe to call if already closed.
    """
    player.close()
    return {"state": "closed", "viewer_open": False}


@mcp.tool()
def run_step_length_experiment(motion: str = "walk", targets: list[float] | None = None,
                               repetitions: int = 3, max_steps: int = 10,
                               heading_degrees: float = 0,
                               show_simulation: bool = True) -> dict[str, Any]:
    """Measure local recording foot placements against requested lengths in metres.

    Measures the recording, saves per-touchdown CSV and trial/results files, and returns a
    comparison table. Targets DO NOT change playback; this is a recorded-pose baseline,
    not physical walking control. Show warnings, incomplete trials and null measurements
    honestly. Defaults: 0.15/0.20/0.25 m, three deterministic replays, ten valid steps.
    By default also plays that recording in the persistent robot window, replacing
    any previous motion. The visualization plays once at real-time speed and holds its
    final pose. Measurement runs faster separately; the animation is not synchronized
    to individual trial rows or driven by their target lengths. Set show_simulation=False
    only for a data-only request. Report any visualization error alongside saved results.
    """
    from step_experiment import run_experiment
    report = run_experiment(motion, targets, repetitions, max_steps, heading_degrees)
    result = {key: value for key, value in report.items() if key != "trials"}
    if show_simulation:
        try:
            result['visualization'] = player.play(motion)
        except (OSError, RuntimeError, ValueError) as exc:
            result['visualization'] = {'state': 'failed', 'error': str(exc)}
    else:
        result['visualization'] = {'state': 'not_requested'}
    return result


@mcp.tool()
def get_step_length_results(experiment_id: str = "latest") -> dict[str, Any]:
    """Retrieve a saved step-length experiment, including trial outcomes and file paths.

    Display requested/achieved distances, absolute error and completed trials; null
    measurements mean no valid steps, never zero error. Include baseline limitations.
    """
    from step_experiment import read_results
    return read_results(experiment_id)


if __name__ == "__main__":
    atexit.register(player.close)
    try:
        mcp.run(transport="stdio")
    finally:
        player.close()
