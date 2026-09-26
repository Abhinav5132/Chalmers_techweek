"""Private JSON-lines worker for the local web studio (no desktop/OpenGL needed).

MuJoCo owns the clock and poses; the browser only renders body transforms.
Recording routines reset at clip boundaries and do not claim physical balance.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import mujoco
import numpy as np

from robot_mcp import MotionPlayer
from wbc_session import catalog as tracking_catalog, clip_error as tracking_clip_error
from physics_session import catalog as physics_catalog, validate as validate_physics

ROOT = Path(__file__).resolve().parents[1]


class Studio:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self.library = MotionPlayer(root)
        self.model: Any = mujoco.MjModel.from_xml_path(str(root / 'unitree_mujoco/unitree_robots/g1/scene_29dof.xml'))
        self.data: Any = mujoco.MjData(self.model)
        self.original_friction = self.model.geom_friction.copy()
        self.physics = None
        self.tracking = None
        self.task_session = None
        self.task_models = {}
        self.entries: list[dict] = []
        self.clips: dict = {}
        self.loop = False
        self.elapsed = 0.0
        self.last_tick = time.monotonic()
        self.physics_budget = 0.0
        self.state: dict = {}
        self.reset()

    def reset(self):
        self.model.geom_friction[:] = self.original_friction
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = .79
        mujoco.mj_forward(self.model, self.data)
        self.entries = []
        self.elapsed = 0.0
        self.state = {'state': 'idle', 'mode': 'kinematic_playback', 'elapsed': 0, 'total': 0}
        return self.status()

    def catalog(self):
        result = self.library.list()
        try:
            result['physics'] = physics_catalog(self.root)
        except Exception as exc:
            result['physics'] = {'teacher_available': False, 'student_available': False, 'error': str(exc)}
        result['tracking'] = tracking_catalog(self.root)
        from task_session import catalog as task_catalog
        result['tasks'] = task_catalog(self.root)
        for motion in result['motions']:
            error = tracking_clip_error(self.library.path(motion['id']))
            motion.update(tracking_available=error is None, tracking_error=error)
        return result

    def _tracking(self):
        if self.tracking is None:
            readiness = tracking_catalog(self.root)
            if not readiness['available']:
                raise ValueError(readiness['error'])
            from wbc_session import WbcSession
            self.tracking = WbcSession(self.root)
        return self.tracking

    def geometry(self, scene='g1'):
        from task_session import PRESETS, make_model
        if scene in PRESETS:
            if scene not in self.task_models:
                self.task_models[scene] = make_model(scene)
            model = self.task_models[scene]
        elif scene in ('g1', 'wbc'):
            model = self._tracking().model if scene == 'wbc' else self.model
        else:
            raise ValueError('Unknown robot scene.')
        meshes = []
        for i in range(model.nmesh):
            va, vn = int(model.mesh_vertadr[i]), int(model.mesh_vertnum[i])
            fa, fn = int(model.mesh_faceadr[i]), int(model.mesh_facenum[i])
            meshes.append({'vertices': model.mesh_vert[va:va+vn].astype(float).round(6).reshape(-1).tolist(),
                           'indices': model.mesh_face[fa:fa+fn].reshape(-1).tolist()})
        geoms = []
        for i in range(model.ngeom):
            # This G1 model puts visual meshes in group 1, collision duplicates in 0.
            task_geom = scene in PRESETS and model.geom(i).name in ('curriculum_step', 'grasp_table', 'object_box')
            if not task_geom and (model.geom_bodyid[i] == 0 or model.geom_group[i] != (2 if scene == 'wbc' else 1)):
                continue
            rgba = model.geom_rgba[i].copy()
            material = int(model.geom_matid[i])
            if material >= 0:
                rgba = model.mat_rgba[material].copy()
            geoms.append({'body': int(model.geom_bodyid[i]), 'type': int(model.geom_type[i]),
                          'mesh': int(model.geom_dataid[i]), 'size': model.geom_size[i].tolist(),
                          'position': model.geom_pos[i].tolist(), 'quaternion': model.geom_quat[i].tolist(),
                          'color': rgba.tolist()})
        return {'id': scene, 'root_body': 2 if scene == 'wbc' else 1, 'meshes': meshes, 'geoms': geoms}

    def run(self, entries, loop=False, mode="kinematic_playback"):
        if mode not in ("kinematic_playback", "wbc_tracking"):
            raise ValueError("Choose recording playback or WBC physics tracking.")
        if not isinstance(entries, list) or not 1 <= len(entries) <= 100 or not isinstance(loop, bool):
            raise ValueError('Choose between 1 and 100 clips.')
        prepared, clips = [], {}
        for entry in entries:
            motion, speed, repeats = entry['motion'], entry['speed'], entry['repeats']
            if not isinstance(speed, (int, float)) or not math.isfinite(speed) or not .25 <= speed <= 2:
                raise ValueError('Speed must be 0.25–2×.')
            if not isinstance(repeats, int) or isinstance(repeats, bool) or not 1 <= repeats <= 20:
                raise ValueError('Repeats must be 1–20.')
            info = self.library.describe(motion)
            if motion not in clips:
                with np.load(self.library.path(motion), allow_pickle=False) as clip:
                    clips[motion] = {k: np.array(clip[k], copy=True) for k in ('joint_pos', 'body_pos_w', 'body_quat_w', 'fps')}
            prepared.append({**entry, 'duration': info['duration_seconds'] / speed * repeats})
        if mode == 'wbc_tracking':
            tracking = self._tracking()
            paths = {e['motion']: self.library.path(e['motion']) for e in prepared}
            state = tracking.start(prepared, paths, loop)
            self.entries, self.loop = prepared, loop
            self.state = dict(state)
            self.physics_budget = 0.
            self.last_tick = time.monotonic()
            return self.status()
        self.reset()
        self.entries, self.clips, self.loop = prepared, clips, loop
        self.state = {'state': 'running', 'mode': 'kinematic_playback', 'total': sum(e['duration'] for e in prepared), 'loop': loop}
        self.last_tick = time.monotonic()
        self.apply_pose()
        return self.status()

    def apply_pose(self):
        remaining = min(self.elapsed, self.state['total'])
        index = len(self.entries) - 1
        for i, entry in enumerate(self.entries):
            if remaining < entry['duration'] or i == len(self.entries) - 1:
                index = i
                break
            remaining -= entry['duration']
        entry = self.entries[index]
        clip = self.clips[entry['motion']]
        count = len(clip['joint_pos'])
        fps = float(clip['fps'].reshape(-1)[0])
        frame = min(count - 1, int(remaining * fps * entry['speed']) % count)
        if self.elapsed >= self.state['total']:
            frame = count - 1
        self.data.qpos[:3] = clip['body_pos_w'][frame, 0]
        self.data.qpos[3:7] = clip['body_quat_w'][frame, 0]
        self.data.qpos[7:36] = clip['joint_pos'][frame]
        mujoco.mj_forward(self.model, self.data)
        self.state.update(elapsed=self.elapsed, index=index, entry_id=entry['id'], motion=entry['motion'], frame=frame)

    def tick(self, dt: float):
        if self.state['state'] != 'running':
            return
        if self.state['mode'] == 'task_physics':
            self.physics_budget = min(self.physics_budget + dt, .05)
            for _ in range(min(2, int((self.physics_budget + 1e-9) / self.task_session.dt))):
                self.physics_budget -= self.task_session.dt
                self.state = dict(self.task_session.step())
                if self.state['state'] != 'running':
                    break
        elif self.state['mode'] == 'wbc_tracking':
            self.physics_budget += dt
            for _ in range(min(5, int((self.physics_budget + 1e-9) / self.tracking.dt))):
                self.physics_budget -= self.tracking.dt
                self.state = dict(self.tracking.step())
                if self.state['state'] != 'running':
                    break
            self.physics_budget = min(self.physics_budget, .1)
        elif self.state['mode'] == 'motor_driven_physics':
            self.physics_budget += dt
            # Preserve simulated time; cap catch-up after a stalled client/startup.
            for _ in range(min(5, int(self.physics_budget / .02))):
                self.physics_budget -= .02
                self.state = dict(self.physics.step())
                self.state.update(elapsed=self.state['simulation_seconds'], total=self.state['requested_seconds'])
                if self.state['state'] != 'running':
                    break
            self.physics_budget = min(self.physics_budget, .1)
        else:
            self.elapsed += dt
            if self.elapsed >= self.state['total']:
                if self.loop:
                    self.elapsed %= self.state['total']
                else:
                    self.elapsed = self.state['total']
                    self.state['state'] = 'completed'
            self.apply_pose()

    def status(self):
        is_tracking = self.state['mode'] == 'wbc_tracking'
        is_task = self.state['mode'] == 'task_physics'
        data = self.task_session.data if is_task else self.tracking.data if is_tracking else self.data
        return {**self.state, 'model_id': self.task_session.task if is_task else 'wbc' if is_tracking else 'g1',
                'positions': data.xpos.reshape(-1).round(6).tolist(),
                'quaternions': data.xquat.reshape(-1).round(6).tolist()}

    def command(self, action, payload):
        if action == 'catalog':
            return self.catalog()
        if action == 'model':
            return self.geometry(payload.get('scene', 'g1'))
        if action == 'status':
            return self.status()
        if action == 'run':
            return self.run(payload['entries'], payload.get('loop', False), payload.get('mode', 'kinematic_playback'))
        if action in ('task_prepare', 'task_run'):
            from task_session import TaskSession
            session = TaskSession(payload['task'], payload.get('controller', 'teacher'), self.root, running=action == 'task_run')
            self.task_session = session
            self.state = dict(session.state)
            self.physics_budget = 0.
            self.last_tick = time.monotonic()
            return self.status()
        if action == 'task_exit':
            self.task_session = None
            self.physics_budget = 0.
            self.last_tick = time.monotonic()
            return self.reset()
        if action == 'reset':
            if self.state['mode'] == 'task_physics':
                return self.command('task_prepare', {'task': self.task_session.task, 'controller': self.task_session.controller})
            return self.reset()
        if action == 'pause':
            if self.state['state'] == 'running':
                self.state['state'] = 'paused'
                if self.state['mode'] == 'task_physics':
                    self.task_session.state['state'] = 'paused'
            return self.status()
        if action == 'resume':
            if self.state['state'] == 'paused':
                self.state['state'] = 'running'
                if self.state['mode'] == 'task_physics':
                    self.task_session.state['state'] = 'running'
                self.last_tick = time.monotonic()
            return self.status()
        if action == 'stop':
            self.state['state'] = 'stopped'
            return self.status()
        if action == 'seek':
            seconds = payload['seconds']
            if self.state['mode'] != 'kinematic_playback' or not self.entries:
                raise ValueError('Start a recording routine before seeking.')
            if not math.isfinite(seconds) or not 0 <= seconds <= self.state['total']:
                raise ValueError('Seek position is outside the routine.')
            self.elapsed = seconds
            self.state['state'] = 'paused'
            self.apply_pose()
            return self.status()
        if action == 'physics':
            validate_physics(payload['task'], payload['controller'], payload['seconds'], payload['push_force'])
            self.state['state'] = 'stopped'
            try:
                if self.physics is None:
                    from physics_session import PhysicsSession
                    self.physics = PhysicsSession(self.model, self.data)
                self.state = dict(self.physics.start(payload['task'], payload['controller'], payload['seconds'], payload['push_force']))
            except Exception as exc:
                self.state.update(state='failed', error=str(exc))
                raise
            self.state.update(elapsed=0, total=payload['seconds'])
            self.physics_budget = 0
            self.last_tick = time.monotonic()
            return self.status()
        raise ValueError('Unknown action.')


def main():
    commands: queue.Queue = queue.Queue()
    def read():
        for line in sys.stdin:
            try:
                commands.put(json.loads(line))
            except ValueError:
                pass
        commands.put(None)
    threading.Thread(target=read, daemon=True).start()
    with redirect_stdout(sys.stderr):
        studio = Studio()
    while True:
        now = time.monotonic()
        dt, studio.last_tick = now - studio.last_tick, now
        try:
            with redirect_stdout(sys.stderr):
                studio.tick(dt)
        except Exception as exc:
            studio.state.update(state='failed', error=str(exc))
        try:
            request = commands.get(timeout=.01)
        except queue.Empty:
            continue
        if request is None:
            break
        try:
            with redirect_stdout(sys.stderr):
                result = studio.command(request['action'], request.get('payload', {}))
            response = {'id': request['id'], 'result': result}
        except Exception as exc:
            response = {'id': request['id'], 'error': str(exc)}
        print(json.dumps(response, allow_nan=False, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
