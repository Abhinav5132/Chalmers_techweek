"""Incremental, headless task demos using the upstream physical controllers."""
from pathlib import Path
import importlib.util
import json
import math
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    'step_5cm': {'title': 'Climb a 5 cm step', 'height': .05, 'seconds': 23.},
    'step_35cm': {'title': 'Climb a 35 cm step', 'height': .35, 'seconds': 23.},
    'box_lift': {'title': 'Lift a box', 'seconds': 12.},
}


def checkpoint(task, root=ROOT):
    if task not in PRESETS:
        raise ValueError('Unknown task preset.')
    if task == 'box_lift':
        pointer = root / '.robot-runtime/imitation/grasp/qualified.json'
    else:
        suffix = '' if task == 'step_5cm' else '-0.350m'
        pointer = root / f'.robot-runtime/imitation/stair-curriculum/feedback-qualified{suffix}.json'
    if not pointer.is_file():
        raise ValueError('No qualified learned student is installed for this preset. Use Teacher.')
    path = Path(json.loads(pointer.read_text())['checkpoint'])
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise ValueError('The qualified student checkpoint is missing. Use Teacher.')
    return path


def catalog(root=ROOT):
    missing = [name for name in ('pinocchio', 'qpsolvers') if importlib.util.find_spec(name) is None]
    error = 'Missing task dependencies: ' + ', '.join(missing) if missing else None
    if not (root / 'unitree_mujoco/unitree_robots/g1/scene_29dof.xml').is_file():
        error = 'The G1 model is missing. Run just setup-model.'
    result = []
    for task, preset in PRESETS.items():
        student_error = None
        try:
            path = checkpoint(task, root)
            if task == 'box_lift':
                from grasp_learning import Student
                Student(path)
            else:
                from stair_feedback import FeedbackStudent
                student = FeedbackStudent(path)
                if not math.isclose(student.height, float(preset['height'])):
                    raise ValueError('The checkpoint is for a different step height.')
        except Exception as exc:
            student_error = str(exc)
        result.append({'id': task, **preset, 'available': error is None, 'error': error,
                       'student_available': student_error is None and error is None,
                       'student_error': student_error})
    return result


def make_model(task):
    if task not in PRESETS:
        raise ValueError('Unknown task preset.')
    if task == 'box_lift':
        from grasp_learning import make_model as grasp_model
        return grasp_model()
    from stair_curriculum import model
    return model(PRESETS[task]['height'])


class TaskSession:
    dt = .01

    def __init__(self, task, controller='teacher', root=ROOT, running=False):
        if task not in PRESETS or controller not in ('teacher', 'student'):
            raise ValueError('Choose a supported task and controller.')
        self.task, self.controller = task, controller
        self.duration = float(PRESETS[task]['seconds'])
        self.student = None
        self.teacher = None
        if task == 'box_lift':
            from grasp_learning import Simulation, GraspTeacher, Student
            if controller == 'student':
                self.student = Student(checkpoint(task, root))
            self.sim = Simulation(101)
            if controller == 'teacher':
                self.teacher = GraspTeacher(101)
        else:
            from stair_curriculum import Teacher
            height = float(PRESETS[task]['height'])
            if controller == 'student':
                from stair_feedback import FeedbackStudent
                self.student = FeedbackStudent(checkpoint(task, root))
                if not math.isclose(self.student.height, height):
                    raise ValueError('The checkpoint is for a different step height.')
            self.sim = Teacher(height, 101)
        self.model, self.data = self.sim.m, self.sim.d
        self.state: dict = {'state': 'running' if running else 'idle', 'mode': 'task_physics',
                      'model_id': task, 'task_id': task, 'controller': controller,
                      'elapsed': 0., 'total': self.duration, 'passed': False,
                      'simulation_seconds': 0., 'reset_count': 1}
        self.update()

    def update(self):
        s = self.sim
        t = float(self.data.time)
        upright = float(self.data.xmat[1, 8])
        if self.task == 'box_lift':
            hands, table, other = s.contacts()
            lift = float(self.data.xpos[s.object_id, 2] - s.initial_z)
            stages = ['Reach', 'Squeeze', 'Lift', 'Hold']
            # These are measured milestones, not a claim that timed targets succeeded.
            contact = bool(np.all(hands > .1))
            stage = 3 if s.hold > 0 else 2 if lift > .01 and contact else 1 if contact else 0
            passed = bool(not s.fail and t + 1e-8 >= self.duration and s.hold >= 2.)
            metrics = {'lift_m': lift, 'hand_contacts': hands.tolist(), 'table_contact_n': float(table),
                       'other_contact_n': float(other), 'hold_seconds': s.hold,
                       'hold_required': 2., 'object_attached': False}
        else:
            loads, top = s.contacts()
            on_top = [bool(self.data.xpos[bid, 0] > .25 and
                           abs(self.data.xpos[bid, 2] - (s.feet0[k, 2] + s.height)) < .025 and top[k] > 20)
                      for k, bid in enumerate(s.ids)]
            stages = ['Balance', 'First foot', 'Second foot', 'Hold']
            stage = 3 if s.top_hold_seconds > 0 else 2 if all(on_top) else 1 if any(on_top) else 0
            passed = bool(not s.fail and t + 1e-8 >= self.duration and all(on_top) and s.top_hold_seconds >= 2.9)
            metrics = {'height_m': s.height, 'foot_contacts': loads.tolist(), 'top_contacts': top.tolist(),
                       'feet_on_top': on_top, 'hold_seconds': s.top_hold_seconds, 'hold_required': 2.9}
        self.state.update(elapsed=round(t, 6), simulation_seconds=round(t, 6),
                          stages=stages, stage_index=stage, upright=upright, passed=passed,
                          base_anchored=False, **metrics)
        if s.fail:
            self.state.update(state='failed', failure_reason=s.fail)
        elif t + 1e-8 >= self.duration:
            self.state.update(state='completed' if passed else 'failed',
                              failure_reason=None if passed else 'Required contact and hold were not achieved.')

    def step(self):
        if self.state['state'] != 'running':
            return self.state
        try:
            if self.task == 'box_lift':
                if self.student is not None:
                    action = self.student.action(self.sim)
                else:
                    self.sim.sync_teacher(self.teacher)
                    action = self.teacher.action()[0]
                self.sim.step(action)
            else:
                self.sim.step(self.student.action(self.sim) if self.student is not None else None)
            mujoco.mj_forward(self.model, self.data)
            self.update()
        except Exception as exc:
            self.state.update(state='failed', failure_reason=str(exc))
        return self.state
