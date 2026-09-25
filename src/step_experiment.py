"""Measure recorded G1 foot placements; requested targets are comparisons, not controls."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import time
from pathlib import Path
import uuid
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIMITATION = ('Recorded-pose baseline only: requested step lengths do not change the motion. '
              'No walking controller, training, or dynamic balance is evaluated. Repetitions are deterministic replays.')


class TouchdownDetector:
    """Debounce contacts and emit only after a confirmed swing and landing.

    Initial ground contact never counts as a step. Snapshot the first contact
    frame but require consecutive contact frames to reject collision chatter.
    """
    def __init__(self, confirm_frames: int = 2):
        self.confirm = confirm_frames
        self.air = {'left': 0, 'right': 0}
        self.ground = {'left': 0, 'right': 0}
        self.armed = {'left': False, 'right': False}
        self.pending: dict[str, Any] = {}

    def update(self, frame: int, contacts: dict[str, bool], positions: dict[str, np.ndarray],
               direction: np.ndarray) -> list[dict[str, Any]]:
        events = []
        for side, other in (('left', 'right'), ('right', 'left')):
            if not contacts[side]:
                self.ground[side] = 0
                self.air[side] += 1
                if self.air[side] >= self.confirm:
                    self.armed[side] = True
                self.pending.pop(side, None)
                continue
            self.air[side] = 0
            self.ground[side] += 1
            if self.armed[side] and self.ground[side] == 1:
                distance = float(np.dot(positions[side][:2] - positions[other][:2], direction))
                self.pending[side] = {'frame': frame, 'foot': side, 'achieved_m': distance,
                                      'other_foot_in_contact': contacts[other],
                                      'landing_x_m': float(positions[side][0]),
                                      'landing_y_m': float(positions[side][1])}
            if self.armed[side] and self.ground[side] >= self.confirm:
                event = self.pending.pop(side)
                event['valid_forward_step'] = event['other_foot_in_contact'] and event['achieved_m'] > 0
                events.append(event)
                self.armed[side] = False
        return events


def load_clip(motion: str, root: Path) -> dict[str, np.ndarray]:
    if not motion or Path(motion).name != motion or motion in ('.', '..'):
        raise ValueError('Use a motion ID such as walk, not a file path.')
    directory = (root / 'data/motions').resolve()
    path = (directory / (motion + '.npz')).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f'Motion {motion!r} is unavailable. Run just setup-motions.')
    with np.load(path, allow_pickle=False) as archive:
        clip = {key: np.array(archive[key], dtype=float) for key in
                ('fps', 'joint_pos', 'body_pos_w', 'body_quat_w')}
    fps = clip['fps'].reshape(-1)
    joints, pos, quat = clip['joint_pos'], clip['body_pos_w'], clip['body_quat_w']
    if fps.size != 1 or not math.isfinite(float(fps[0])) or fps[0] <= 0:
        raise ValueError('Clip needs a positive finite frame rate.')
    if joints.ndim != 2 or joints.shape[1] != 29 or not len(joints) or len(joints) > 100000:
        raise ValueError('Clip must have 1–100000 frames and 29 joints per frame.')
    if (pos.ndim != 3 or quat.ndim != 3 or pos.shape[0] != len(joints)
            or quat.shape[0] != len(joints) or pos.shape[1] < 1 or quat.shape[1] < 1
            or pos.shape[2] != 3 or quat.shape[2] != 4):
        raise ValueError('Invalid body position or quaternion arrays.')
    if not all(np.isfinite(a).all() for a in clip.values()):
        raise ValueError('Non-finite clip values.')
    norms = np.linalg.norm(quat[:, 0], axis=1)
    if np.any(norms < 1e-6):
        raise ValueError('Zero root quaternion.')
    quat[:, 0] /= norms[:, None]
    return clip


def measure_trial(model: Any, clip: dict[str, np.ndarray], target: float, trial: int,
                  max_steps: int, heading_degrees: float, confirm_frames: int) -> tuple[list, dict]:
    data: Any = mujoco.MjData(model)
    detector = TouchdownDetector(confirm_frames)
    foot_ids = {side: model.body(side + '_ankle_roll_link').id for side in ('left', 'right')}
    floor = model.geom('floor').id
    angle = math.radians(heading_degrees)
    direction = np.array([math.cos(angle), math.sin(angle)])
    fps = float(clip['fps'].reshape(-1)[0])
    rows: list[dict[str, Any]] = []
    contact_counts = {'left': 0, 'right': 0}
    previous: dict[str, np.ndarray] = {}
    previous_contact = {'left': False, 'right': False}
    drift = {'left': 0.0, 'right': 0.0}
    fallen = False
    valid = 0
    for frame, joints in enumerate(clip['joint_pos']):
        data.qpos[:3] = clip['body_pos_w'][frame, 0]
        data.qpos[3:7] = clip['body_quat_w'][frame, 0]
        data.qpos[7:36] = joints
        mujoco.mj_forward(model, data)
        contacts = {'left': False, 'right': False}
        for contact in data.contact:
            if contact.dist > 0:
                continue
            geom = contact.geom2 if contact.geom1 == floor else contact.geom1 if contact.geom2 == floor else -1
            if geom < 0:
                continue
            for side, body in foot_ids.items():
                if model.geom_bodyid[geom] == body:
                    contacts[side] = True
        positions = {side: np.array(data.xpos[body]) for side, body in foot_ids.items()}
        for side in contacts:
            contact_counts[side] += int(contacts[side])
            if contacts[side] and previous_contact[side]:
                drift[side] += float(np.linalg.norm(positions[side][:2] - previous[side][:2]))
        previous, previous_contact = positions, contacts.copy()
        for event in detector.update(frame, contacts, positions, direction):
            valid += int(event['valid_forward_step'])
            rows.append(dict(trial=trial, step=len(rows) + 1, time_s=event['frame'] / fps,
                             requested_m=target, error_m=event['achieved_m'] - target,
                             mode='recorded_pose_baseline', **event))
        # A pose-level warning only; a kinematic replay cannot demonstrate dynamic falling.
        pelvis = model.body('pelvis').id
        upright = float(data.xmat[pelvis].reshape(3, 3)[2, 2])
        if data.qpos[2] < 0.4 or upright < math.cos(math.radians(60)):
            fallen = True
            break
        if valid >= max_steps:
            break
    usable = [r for r in rows if r['valid_forward_step']]
    errors = [abs(r['error_m']) for r in usable]
    distances = [r['achieved_m'] for r in usable]
    status = 'fall_pose_detected' if fallen else 'complete' if valid >= max_steps else 'insufficient_valid_steps'
    summary = {
        'trial': trial, 'requested_m': target, 'status': status,
        'touchdowns': len(rows), 'valid_steps': valid, 'required_steps': max_steps,
        'mean_step_m': float(np.mean(distances)) if distances else None,
        'mean_absolute_error_m': float(np.mean(errors)) if errors else None,
        'step_std_m': float(np.std(distances)) if distances else None,
        'recorded_fall_pose_detected': fallen, 'contact_frames': contact_counts,
        'stance_foot_origin_drift_m': drift, 'frames_processed': frame + 1,
    }
    return rows, summary


def run_experiment(motion: str = 'walk', targets: list[float] | None = None,
                   repetitions: int = 3, max_steps: int = 10, heading_degrees: float = 0,
                   confirm_frames: int = 2, root: Path = ROOT) -> dict[str, Any]:
    targets = [0.15, 0.20, 0.25] if targets is None else targets
    if not targets or len(targets) > 10 or any(not math.isfinite(t) or not 0 < t <= 1 for t in targets):
        raise ValueError('Provide 1–10 positive targets no greater than 1 metre.')
    if len(set(targets)) != len(targets):
        raise ValueError('Targets must be distinct; use repetitions for repeated trials.')
    if not 1 <= repetitions <= 10 or not 1 <= max_steps <= 100 or not 1 <= confirm_frames <= 30:
        raise ValueError('Use 1–10 repetitions, 1–100 steps, and 1–30 confirmation frames.')
    if not math.isfinite(heading_degrees):
        raise ValueError('Heading must be finite.')
    clip = load_clip(motion, root)
    if len(clip['joint_pos']) * len(targets) * repetitions > 500000:
        raise ValueError('Experiment exceeds 500000 frames; reduce targets or repetitions.')
    model = mujoco.MjModel.from_xml_path(str(root / 'unitree_mujoco/unitree_robots/g1/scene_29dof.xml'))
    all_rows, trials = [], []
    for target in targets:
        for repetition in range(1, repetitions + 1):
            rows, summary = measure_trial(model, clip, target, repetition, max_steps, heading_degrees, confirm_frames)
            all_rows.extend(rows)
            trials.append(summary)
    comparisons = []
    for target in targets:
        selected = [t for t in trials if t['requested_m'] == target]
        valid_rows = [r for r in all_rows if r['requested_m'] == target and r['valid_forward_step']]
        comparisons.append({'requested_m': target, 'trials': len(selected),
            'completed_trials': sum(t['status'] == 'complete' for t in selected),
            'valid_steps': len(valid_rows), 'touchdowns': sum(t['touchdowns'] for t in selected),
            'mean_achieved_m': float(np.mean([r['achieved_m'] for r in valid_rows])) if valid_rows else None,
            'mean_absolute_error_m': float(np.mean([abs(r['error_m']) for r in valid_rows])) if valid_rows else None})
    experiment_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    directory = root / '.robot-runtime/experiments' / experiment_id
    directory.mkdir(parents=True)
    warnings = []
    if any(t['status'] != 'complete' for t in trials):
        warnings.append('One or more trials did not achieve the requested number of valid forward steps. Do not report them as successes.')
    if any(t['contact_frames']['right'] == 0 or t['contact_frames']['left'] == 0 for t in trials):
        warnings.append('A foot never contacts the floor. Check recording retargeting, joint order, floor clearance, and the suitability of this clip before using it as a walking baseline.')
    report = {'experiment_id': experiment_id, 'mode': 'recorded_pose_baseline',
              'target_applied_to_controller': False, 'limitation': LIMITATION,
              'motion': motion, 'heading_degrees': heading_degrees, 'confirm_frames': confirm_frames,
              'max_steps': max_steps, 'repetitions': repetitions,
              'measurement': 'Signed foot-origin separation along fixed heading at first contact, confirmed over consecutive frames. Valid forward steps require the other foot in floor contact.',
              'fall_pose_thresholds': {'pelvis_height_below_m': 0.4, 'pelvis_tilt_above_degrees': 60},
              'warnings': warnings, 'comparisons': comparisons, 'trials': trials,
              'files': {name: str(directory / name) for name in ('steps.csv', 'trials.csv', 'results.json')}}
    fields = ['trial', 'step', 'time_s', 'requested_m', 'achieved_m', 'error_m', 'foot', 'frame',
              'other_foot_in_contact', 'valid_forward_step', 'landing_x_m', 'landing_y_m', 'mode']
    with (directory / 'steps.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    with (directory / 'trials.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
    (directory / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def read_results(experiment_id: str = 'latest', root: Path = ROOT) -> dict[str, Any]:
    directory = root / '.robot-runtime/experiments'
    if experiment_id == 'latest':
        paths = list(directory.glob('*/results.json'))
        if not paths:
            raise ValueError('No results yet. Run a step-length experiment first.')
        path = max(paths, key=lambda candidate: candidate.stat().st_mtime_ns)
    else:
        if Path(experiment_id).name != experiment_id or experiment_id in ('', '.', '..'):
            raise ValueError('Use an experiment ID returned by the experiment tool.')
        path = directory / experiment_id / 'results.json'
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError('Invalid results path.')
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', default='walk')
    parser.add_argument('--targets', nargs='+', type=float, default=[0.15, 0.20, 0.25])
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--heading-degrees', type=float, default=0)
    parser.add_argument('--confirm-frames', type=int, default=2)
    parser.add_argument('--show-simulation', action='store_true',
                        help='Also show the measured recording in MuJoCo; close the window to exit.')
    args = parser.parse_args()
    try:
        report = run_experiment(args.motion, args.targets, args.repetitions, args.steps,
                                args.heading_degrees, args.confirm_frames)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(LIMITATION)
    for row in report['comparisons']:
        print(json.dumps(row))
    for warning in report['warnings']:
        print('WARNING:', warning)
    print('Results:', report['files']['results.json'])
    if args.show_simulation:
        from robot_mcp import MotionPlayer
        player = MotionPlayer()
        print('Showing the same recording at real-time speed. Targets do not alter it. Close the window or press Ctrl+C to exit.')
        try:
            player.play(args.motion)
            while player.process is not None and player.process.poll() is None:
                time.sleep(0.1)
            status = player.status()
            if status['state'] == 'failed':
                raise RuntimeError(str(status))
        except KeyboardInterrupt:
            pass
        finally:
            player.close()


if __name__ == '__main__':
    main()
