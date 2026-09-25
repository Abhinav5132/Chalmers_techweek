"""Run saved imitation controllers on the persistent viewer's model and data."""
from pathlib import Path
import json
import math
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def saved_policy(root=ROOT):
    runtime = root / '.robot-runtime/imitation'
    active = runtime / 'recovery/active.json'
    balance = runtime / 'balance/latest.json'
    if active.exists():
        path = Path(json.loads(active.read_text())['policy'])
    elif balance.exists():
        path = Path(json.loads(balance.read_text())['session']) / 'policy.pt'
    else:
        path = runtime / 'policy.pt'
    if not path.resolve().is_relative_to(runtime.resolve()) or not path.is_file():
        raise ValueError('Saved policy is unavailable. Train it locally or repair the saved session paths.')
    return path


def catalog(root=ROOT):
    try:
        checkpoint = str(saved_policy(root))
        error = None
    except (ValueError, OSError, KeyError) as exc:
        checkpoint, error = None, str(exc)
    teacher = (root / 'data/imitation/walk_teacher_bundle.onnx').is_file()
    return {'skills': ['stand', 'walk'], 'student_available': checkpoint is not None,
            'teacher_available': teacher, 'checkpoint': checkpoint, 'error': error,
            'mode': 'motor_driven_physics',
            'note': 'Reference-conditioned imitation controllers. Commands reset the trial; transitions and requested step lengths are not learned.'}


def validate(task, controller, seconds, push_force):
    if task not in ('stand', 'walk') or controller not in ('student', 'teacher'):
        raise ValueError('Choose stand/walk and student/teacher')
    if not math.isfinite(seconds) or not 1 <= seconds <= 10:
        raise ValueError('Trial duration must be 1–10 seconds')
    if not math.isfinite(push_force) or not 0 <= push_force <= 100:
        raise ValueError('Push force must be 0–100 N')


class PhysicsSession:
    def __init__(self, model, data):
        import torch
        from recovery_training import RecoverySimulation
        torch.set_num_threads(2)
        if not catalog()['teacher_available']:
            raise ValueError('Download the teacher with: uv run --group training python src/imitation_g1.py prepare')
        self.sim = RecoverySimulation()
        # Keep the same MuJoCo objects bound to the existing viewer window.
        self.sim.model, self.sim.data = model, data
        self.sim.model.opt.timestep = .002
        self.policy = None
        self.obs = None
        self.state: dict[str, Any] = {}

    def start(self, task, controller, seconds, push_force):
        from imitation_g1 import student
        validate(task, controller, seconds, push_force)
        checkpoint = saved_policy() if controller == 'student' else None
        self.policy = student(checkpoint) if checkpoint else self.sim.expert
        self.obs = self.sim.reset_scenario(task, 4300000, 'push' if push_force else 'clean', push_force=push_force)
        self.duration = seconds
        self.origin = self.sim.data.qpos[:2].copy()
        self.state = {'state': 'running', 'mode': 'motor_driven_physics', 'task': task,
                      'controller': controller, 'checkpoint': str(checkpoint) if checkpoint else None,
                      'simulation_seconds': 0., 'control_steps': 0, 'push_force_n': push_force,
                      'requested_seconds': seconds, 'fell': False, 'max_drift_m': 0.,
                      'simulation_paused': False, 'reset_at_start': True,
                      'note': 'Saved controller inference; no training occurs during this trial. Pushes, if requested, occur at 2 and 6 seconds.'}
        return self.state

    def step(self):
        import numpy as np
        if self.policy is None or self.obs is None:
            raise RuntimeError('Start a physical trial first')
        self.obs, fallen = self.sim.step(self.policy(self.obs))
        self.state.update(simulation_seconds=round(self.sim.t, 3), fell=fallen,
                          control_steps=self.state['control_steps'] + 1,
                          max_drift_m=max(self.state['max_drift_m'], float(np.linalg.norm(self.sim.data.qpos[:2] - self.origin))))
        if fallen or self.sim.t + 1e-9 >= self.duration:
            self.state.update(state='fallen' if fallen else 'completed', simulation_paused=True)
        return self.state
