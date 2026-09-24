"""
Workflow Scripting Engine & Skill Sequencer for Unitree G1.
Orchestrates high-level skills (RL-tracked motion clips from Blender/MoCap,
Pink QP manipulation, standing postures) into executable workflows.
Fully typed for static type checkers (ty / mypy / pyright).
"""

from __future__ import annotations

import os
import json
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import mujoco

from src.controllers.wbc_runner import WbcPhysicsRunner
from src.controllers.pink_controller import PinkG1Controller

mj: Any = mujoco


class WorkflowContext:
    """Shared execution context passed to each workflow node during simulation."""

    def __init__(
        self,
        mj_model: Any,
        mj_data: Any,
        wbc_runner: WbcPhysicsRunner,
        pink_ctrl: PinkG1Controller,
    ) -> None:
        self.mj_model: Any = mj_model
        self.mj_data: Any = mj_data
        self.wbc_runner: WbcPhysicsRunner = wbc_runner
        self.pink_ctrl: PinkG1Controller = pink_ctrl
        self.sim_time: float = 0.0
        self.step_count: int = 0
        self.state_data: dict[str, Any] = {}


class WorkflowNode(ABC):
    """Abstract base class for all executable skills in the scripting engine."""

    def __init__(self, name: str, duration: float) -> None:
        self.name: str = name
        self.duration: float = max(0.01, duration)
        self.elapsed: float = 0.0

    @abstractmethod
    def on_enter(self, context: WorkflowContext) -> None:
        """Called when this skill becomes active."""
        ...

    @abstractmethod
    def update(self, context: WorkflowContext, dt: float) -> bool:
        """
        Executes one physics/control step.
        Returns True when the node's duration or goal has been completed.
        """
        ...

    @abstractmethod
    def on_exit(self, context: WorkflowContext) -> None:
        """Called when this skill finishes execution."""
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize node parameters to dictionary for visual node graph editor."""
        ...


class MotionClipNode(WorkflowNode):
    """
    Skill Node that tracks an RL-tracked animation clip in MuJoCo physics via the
    exported WBC policy. Plays the clip once (no loop); duration defaults to the
    clip length.
    """

    def __init__(
        self,
        clip_name: str,
        duration: float | None = None,
        name: str | None = None,
    ) -> None:
        node_name = name or f"TrackClip_{clip_name}"
        super().__init__(node_name, duration if duration is not None else 0.01)
        self.clip_name: str = clip_name
        self.clip_path: str = self._resolve_clip_path(clip_name)

    @staticmethod
    def _resolve_clip_path(clip_name: str) -> str:
        if os.path.exists(clip_name):
            return clip_name
        # Check data/motions/<clip_name>.npz
        candidate = os.path.join("data", "motions", f"{clip_name}.npz" if not clip_name.endswith(".npz") else clip_name)
        if os.path.exists(candidate):
            return candidate
        return clip_name

    def on_enter(self, context: WorkflowContext) -> None:
        self.elapsed = 0.0
        if os.path.exists(self.clip_path):
            context.wbc_runner.load_clip(self.clip_path)
            context.wbc_runner.reset_to_initial_pose()
            clip_dur = context.wbc_runner.clip_duration()
            if self.duration < 0.01 or clip_dur is not None:
                self.duration = clip_dur if clip_dur is not None else self.duration
            print(f"[Workflow] Started Motion Clip: '{self.clip_name}' ({self.duration:.1f}s)")
        else:
            raise FileNotFoundError(f"Motion clip file not found: {self.clip_path}")

    def update(self, context: WorkflowContext, dt: float) -> bool:
        self.elapsed += dt
        context.wbc_runner.step_policy()
        return self.elapsed >= self.duration

    def on_exit(self, context: WorkflowContext) -> None:
        print(f"[Workflow] Completed Motion Clip: '{self.clip_name}'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "motion_clip",
            "name": self.name,
            "clip_name": self.clip_name,
            "duration": self.duration,
            "loop": False,
        }


class PinkReachNode(WorkflowNode):
    """
    Skill Node that executes high-precision Cartesian arm reaching via Pink QP IK.
    """

    def __init__(
        self,
        target_pos: list[float],
        target_rpy: list[float] | None = None,
        body_name: str = "right_wrist_yaw_link",
        duration: float = 2.5,
        relative_to_robot: bool = True,
        name: str | None = None,
    ) -> None:
        node_name = name or f"PinkReach_{body_name}"
        super().__init__(node_name, duration)
        self.target_pos: np.ndarray = np.array(target_pos, dtype=np.float64)
        self.target_rpy: np.ndarray | None = (
            np.array(target_rpy, dtype=np.float64) if target_rpy is not None else None
        )
        self.body_name: str = body_name
        self.relative_to_robot: bool = relative_to_robot
        self.effective_target_pos: np.ndarray = self.target_pos.copy()

    def on_enter(self, context: WorkflowContext) -> None:
        self.elapsed = 0.0
        if self.relative_to_robot:
            pelvis_pos = np.asarray(context.mj_data.qpos[0:3], dtype=np.float64)
            self.effective_target_pos = np.array([
                pelvis_pos[0] + self.target_pos[0],
                pelvis_pos[1] + self.target_pos[1],
                self.target_pos[2],
            ], dtype=np.float64)
        else:
            self.effective_target_pos = self.target_pos.copy()

        print(f"[Workflow] Started Pink Reach: target={np.round(self.effective_target_pos, 3).tolist()} ({self.duration:.1f}s)")

    def update(self, context: WorkflowContext, dt: float) -> bool:
        self.elapsed += dt
        # Solve Pink QP IK and apply joint PD torques
        context.pink_ctrl.step_reach(
            target_pos=self.effective_target_pos,
            target_rpy=self.target_rpy,
            body_name=self.body_name,
            dt=dt,
            in_world_frame=True,
        )
        mj.mj_step(context.mj_model, context.mj_data)
        return self.elapsed >= self.duration

    def on_exit(self, context: WorkflowContext) -> None:
        print(f"[Workflow] Completed Pink Reach: '{self.body_name}'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "pink_reach",
            "name": self.name,
            "target_pos": self.target_pos.tolist(),
            "target_rpy": self.target_rpy.tolist() if self.target_rpy is not None else None,
            "body_name": self.body_name,
            "duration": self.duration,
            "relative_to_robot": self.relative_to_robot,
        }


class StandHoldNode(WorkflowNode):
    """
    Skill Node that holds a stable, calibrated standing posture.
    """

    def __init__(self, duration: float = 2.0, name: str = "Stand_Hold") -> None:
        super().__init__(name, duration)

    def on_enter(self, context: WorkflowContext) -> None:
        self.elapsed = 0.0
        print(f"[Workflow] Started Stand Hold ({self.duration:.1f}s)")

    def update(self, context: WorkflowContext, dt: float) -> bool:
        self.elapsed += dt
        torques = context.pink_ctrl.compute_pd_torques(context.pink_ctrl.q_nominal_29)
        context.mj_data.ctrl[:] = torques
        mj.mj_step(context.mj_model, context.mj_data)
        return self.elapsed >= self.duration

    def on_exit(self, context: WorkflowContext) -> None:
        print(f"[Workflow] Completed Stand Hold")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "stand_hold",
            "name": self.name,
            "duration": self.duration,
        }


class WorkflowEngine:
    """
    Sequential Scripting Engine and Dispatcher.
    Manages loading, sequencing, and executing workflow nodes in MuJoCo physics.
    """

    def __init__(self, context: WorkflowContext) -> None:
        self.context: WorkflowContext = context
        self.nodes: list[WorkflowNode] = []
        self.active_node_idx: int = 0
        self.is_finished: bool = False
        self._node_entered: bool = False

    def add_node(self, node: WorkflowNode) -> None:
        """Append a skill node to the workflow."""
        self.nodes.append(node)

    def clear(self) -> None:
        """Clear all nodes."""
        self.nodes.clear()
        self.active_node_idx = 0
        self.is_finished = False
        self._node_entered = False

    @property
    def current_node(self) -> WorkflowNode | None:
        if 0 <= self.active_node_idx < len(self.nodes):
            return self.nodes[self.active_node_idx]
        return None

    def step(self, dt: float = 0.002) -> bool:
        """
        Steps the active node. If the node completes, transitions to the next node.
        Returns True when the entire workflow has finished.
        """
        if self.is_finished or len(self.nodes) == 0:
            return True

        node = self.nodes[self.active_node_idx]
        if not self._node_entered:
            node.on_enter(self.context)
            self._node_entered = True

        done = node.update(self.context, dt)
        self.context.sim_time += dt
        self.context.step_count += 1

        if done:
            node.on_exit(self.context)
            self.active_node_idx += 1
            self._node_entered = False
            if self.active_node_idx >= len(self.nodes):
                self.is_finished = True
                print("[Workflow] Entire workflow sequence finished successfully!")
                return True

        return False

    def export_json(self) -> str:
        """Export workflow definition to JSON string for UI editor."""
        data = [n.to_dict() for n in self.nodes]
        return json.dumps({"version": "1.0", "workflow": data}, indent=2)

    def load_json(self, json_str: str) -> None:
        """Parse workflow from JSON string."""
        self.clear()
        obj = json.loads(json_str)
        workflow_list = obj.get("workflow", [])
        for item in workflow_list:
            node_type = item.get("type")
            if node_type == "motion_clip":
                self.add_node(
                    MotionClipNode(
                        clip_name=item["clip_name"],
                        duration=item.get("duration"),
                        name=item.get("name"),
                    )
                )
            elif node_type == "pink_reach":
                self.add_node(
                    PinkReachNode(
                        target_pos=item["target_pos"],
                        target_rpy=item.get("target_rpy"),
                        body_name=item.get("body_name", "right_wrist_yaw_link"),
                        duration=item.get("duration", 2.5),
                        name=item.get("name"),
                    )
                )
            elif node_type == "stand_hold":
                self.add_node(
                    StandHoldNode(
                        duration=item.get("duration", 2.0),
                        name=item.get("name", "Stand_Hold"),
                    )
                )
            else:
                print(f"[Workflow WARN] Unknown node type '{node_type}' in JSON.")
