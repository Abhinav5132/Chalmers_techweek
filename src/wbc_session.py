"""Web routine adapter for the Phase-2-Physics ONNX tracking runtime.

Only the first clip resets to its initial reference pose. Subsequent clips,
repeats and loop wraps preserve the simulated robot and anchor the new reference
to its current heading and horizontal position, as in the phase-2 workflow engine.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def catalog(root: Path = ROOT) -> dict:
    bundle = root / 'models/params'
    required = ['policy.onnx', 'config.yaml', 'robot_train/scene.xml']
    missing = [name for name in required if not (bundle / name).is_file()]
    dependencies = [name for name in ('onnxruntime', 'yaml') if importlib.util.find_spec(name) is None]
    error = None
    if missing:
        error = 'Missing phase-2 bundle files: ' + ', '.join(missing)
    elif dependencies:
        error = 'Install tracking dependencies: uv sync --group training --group tracking'
    return {'available': not missing and not dependencies, 'error': error,
            'controller': 'Phase-2 WBC ONNX', 'model_id': 'wbc',
            'transitions': 'continuous_anchored',
            'note': 'Motor-driven tracking with continuous clip transitions. Stops on detected falls. Speed changes are experimental.'}


def clip_error(path: Path) -> str | None:
    try:
        with np.load(path, allow_pickle=False) as clip:
            for name in ('body_lin_vel_w', 'body_ang_vel_w'):
                if name not in clip:
                    return f'Physics tracking requires {name}.'
                if clip[name].shape != clip['body_pos_w'].shape or not np.isfinite(clip[name]).all():
                    return f'Invalid {name} for physics tracking.'
            if clip['body_pos_w'].shape[1] != 30 or clip['body_quat_w'].shape[1] != 30:
                return 'Physics tracking requires the G1 30-body recording layout.'
            if np.any(np.linalg.norm(clip['body_quat_w'], axis=-1) < 1e-6):
                return 'Physics tracking requires nonzero body quaternions.'
        return None
    except (OSError, ValueError, KeyError) as exc:
        return str(exc)


class WbcSession:
    def __init__(self, root: Path = ROOT):
        from controllers.wbc_runner import WbcPhysicsRunner
        self.root = root
        params = root / 'models/params'
        self.model: Any = mujoco.MjModel.from_xml_path(str(params / 'robot_train/scene.xml'))
        self.data: Any = mujoco.MjData(self.model)
        self.runner = WbcPhysicsRunner(self.model, self.data, params)
        self.dt = self.runner.policy_step_dt
        self.entries: list[dict] = []
        self.clips: dict = {}
        self.state: dict = {}
        self.pending_advance = False

    def start(self, entries: list[dict], paths: dict[str, Path], loop: bool):
        from controllers.wbc_runner import ClipReference
        # Prepare every reference before replacing an existing run.
        clips = {}
        delays = []
        for entry in entries:
            delay = entry.get('delay_after', 0)
            if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or not 0 <= delay <= 10:
                raise ValueError('Transition delay must be 0–10 seconds.')
            delays.append(round(delay / self.dt))
            key = (entry['motion'], entry['speed'])
            if key not in clips:
                clips[key] = ClipReference(str(paths[entry['motion']]), target_fps=1/self.dt, speed=entry['speed'])
        self.entries, self.clips, self.loop = entries, clips, loop
        self.delays = delays
        self.hold_remaining = 0
        self.total_steps = sum((clips[e['motion'], e['speed']].n_frames + delays[i]) * e['repeats'] for i, e in enumerate(entries))
        if not loop:
            self.total_steps -= delays[-1]
        self.index = self.repeat = self.cycle_steps = self.control_steps = 0
        self.cycle = 1
        self.error_sum = self.max_error = 0.
        self.pending_advance = False
        self.state = {'state': 'running', 'mode': 'wbc_tracking', 'model_id': 'wbc',
                      'total': self.total_steps * self.dt, 'elapsed': 0., 'loop': loop,
                      'controller': 'Phase-2 WBC ONNX', 'fell': False, 'reset_count': 0,
                      'completed_clips': 0, 'cycle': 1, 'simulation_seconds': 0.,
                      'phase': 'clip', 'transition_remaining': 0.,
                      'control_steps': 0, 'tracking_rmse_rad': 0., 'max_tracking_rmse_rad': 0.,
                      'note': 'Physics tracking; preserves physical state between clips and repeats. Completion is not a tracking-quality guarantee.'}
        self._enter()
        return self.state

    def _enter(self):
        entry = self.entries[self.index]
        initial = self.control_steps == 0
        self.runner.set_clip(self.clips[entry['motion'], entry['speed']], anchor_to_current=not initial)
        if initial:
            self.runner.reset_to_initial_pose()
        self.state.update(index=self.index, entry_id=entry['id'], motion=entry['motion'],
                          repeat_index=self.repeat + 1, repeats=entry['repeats'],
                          reset_count=self.state['reset_count'] + int(initial), frame=0,
                          root_height_m=float(self.data.qpos[2]), tracking_error_rad=0.)

    def step(self):
        if self.pending_advance:
            self.repeat += 1
            if self.repeat >= self.entries[self.index]['repeats']:
                self.repeat = 0
                self.index += 1
            if self.index == len(self.entries):
                self.index = 0
                self.cycle_steps = 0
                self.cycle += 1
                self.state['cycle'] = self.cycle
            self._enter()
            self.pending_advance = False
        holding = self.hold_remaining > 0
        self.state['phase'] = 'transition_hold' if holding else 'clip'
        clip = self.runner.clip
        reference = clip.joint_pos[min(self.runner.frame, clip.n_frames - 1)]
        self.runner.step_policy(hold=holding)
        if holding:
            self.hold_remaining -= 1
        self.state["transition_remaining"] = round(self.hold_remaining * self.dt, 6)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('Physics produced non-finite state; trial stopped.')
        error = float(np.mean((self.data.qpos[7:36] - reference) ** 2))
        self.error_sum += error
        self.max_error = max(self.max_error, error ** .5)
        self.control_steps += 1
        self.cycle_steps += 1
        root_height = float(self.data.qpos[2])
        upright = float(self.data.xmat[self.runner.root_body_id, 8])
        self.state.update(elapsed=round(self.cycle_steps * self.dt, 6),
                          simulation_seconds=round(self.control_steps * self.dt, 6),
                          control_steps=self.control_steps, frame=clip.n_frames - 1 if holding else self.runner.frame - 1,
                          root_height_m=root_height, contact_count=int(self.data.ncon),
                          tracking_error_rad=error ** .5,
                          tracking_rmse_rad=(self.error_sum / self.control_steps) ** .5,
                          max_tracking_rmse_rad=self.max_error)
        if root_height < .25 or upright < .2:
            reason = 'Pelvis fell below 0.25 m.' if root_height < .25 else 'Pelvis tilted more than 78 degrees.'
            self.state.update(state='fallen', fell=True, failure_reason=reason)
        elif holding:
            if self.hold_remaining == 0:
                self.pending_advance = True
        elif self.runner.frame >= clip.n_frames:
            self.state['completed_clips'] += 1
            final = self.index == len(self.entries) - 1 and self.repeat + 1 == self.entries[self.index]['repeats']
            if final and not self.loop:
                self.state['state'] = 'completed'
            else:
                self.hold_remaining = self.delays[self.index]
                self.state['transition_remaining'] = self.hold_remaining * self.dt
                self.pending_advance = self.hold_remaining == 0
        return self.state
