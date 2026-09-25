"""Create a longer-stride kinematic recording using constrained leg inverse kinematics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from step_experiment import ROOT, load_clip


def make_longer_walk(scale: float = 1.25, root: Path = ROOT) -> dict[str, Any]:
    if not math.isfinite(scale) or not 1.05 <= scale <= 1.5:
        raise ValueError('Use a stride scale between 1.05 and 1.5 (e.g. 1.25 for 25% larger targets).')
    clip = load_clip('walk', root)
    model: Any = mujoco.MjModel.from_xml_path(str(root / 'unitree_mujoco/unitree_robots/g1/scene_29dof.xml'))
    data: Any = mujoco.MjData(model)
    bodies = [model.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    joint_names = [side + '_' + name + '_joint' for side in ('left', 'right')
                   for name in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
    joints = [model.joint(name).id for name in joint_names]
    qidx = model.jnt_qposadr[joints]
    vidx = model.jnt_dofadr[joints]
    lower, upper = model.jnt_range[joints].T
    out_joints, out_pos, out_quat = [], [], []
    original_separation, new_separation, residuals = [], [], []
    start = clip['body_pos_w'][0, 0, :2]
    for frame, pose in enumerate(clip['joint_pos']):
        data.qpos[:3] = clip['body_pos_w'][frame, 0]
        data.qpos[3:7] = clip['body_quat_w'][frame, 0]
        data.qpos[7:36] = pose
        mujoco.mj_forward(model, data)
        pelvis = model.body('pelvis').id
        forward = np.array(data.xmat[pelvis].reshape(3, 3)[:, 0])
        forward[2] = 0
        forward /= max(np.linalg.norm(forward), 1e-9)
        reference = np.array(data.xpos[bodies])
        rotations = np.array(data.xmat[bodies]).reshape(2, 3, 3)
        original_separation.append(float(np.dot(reference[0] - reference[1], forward)))
        root_delta = np.zeros(3)
        root_delta[:2] = (scale - 1) * (data.qpos[:2] - start)
        targets = reference + root_delta
        for i in range(2):
            targets[i] += (scale - 1) * np.dot(reference[i] - data.qpos[:3], forward) * forward
        data.qpos[:3] += root_delta
        data.qpos[qidx] = np.clip(data.qpos[qidx], lower, upper)
        for _ in range(60):
            mujoco.mj_forward(model, data)
            errors, jacobians = [], []
            for i, body in enumerate(bodies):
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jp, jr, body)
                rotation = data.xmat[body].reshape(3, 3)
                orientation_error = 0.5 * sum(np.cross(rotation[:, k], rotations[i, :, k]) for k in range(3))
                errors.extend([targets[i] - data.xpos[body], 0.15 * orientation_error])
                jacobians.extend([jp[:, vidx], 0.15 * jr[:, vidx]])
            error = np.concatenate(errors)
            if np.linalg.norm(error) < 0.0002:
                break
            jacobian = np.vstack(jacobians)
            delta = np.linalg.solve(jacobian.T @ jacobian + 1e-5 * np.eye(12), jacobian.T @ error)
            previous = data.qpos[qidx].copy()
            improved = False
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                data.qpos[qidx] = np.clip(previous + fraction * np.clip(delta, -0.08, 0.08), lower, upper)
                mujoco.mj_forward(model, data)
                candidate = []
                for i, body in enumerate(bodies):
                    rotation = data.xmat[body].reshape(3, 3)
                    orientation_error = 0.5 * sum(np.cross(rotation[:, k], rotations[i, :, k]) for k in range(3))
                    candidate.extend([targets[i] - data.xpos[body], 0.15 * orientation_error])
                if np.linalg.norm(np.concatenate(candidate)) < np.linalg.norm(error):
                    improved = True
                    break
            if not improved:
                data.qpos[qidx] = previous
                break
        mujoco.mj_forward(model, data)
        residuals.extend(np.linalg.norm(data.xpos[bodies] - targets, axis=1).tolist())
        new_separation.append(float(np.dot(data.xpos[bodies[0]] - data.xpos[bodies[1]], forward)))
        out_joints.append(data.qpos[7:36].copy())
        out_pos.append(data.xpos[1:].copy())
        out_quat.append(data.xquat[1:].copy())
    before = float(np.sqrt(np.mean(np.square(original_separation))))
    after = float(np.sqrt(np.mean(np.square(new_separation))))
    if not np.isfinite(out_joints).all() or after <= before:
        raise RuntimeError('Could not generate a valid recording with larger foot separation.')
    motion = f'walk_longer_{round(scale * 100)}'
    path = root / 'data/motions' / (motion + '.npz')
    np.savez_compressed(path, fps=clip['fps'], joint_pos=out_joints,
                        body_pos_w=out_pos, body_quat_w=out_quat)
    report = {'motion': motion, 'requested_stride_scale': scale,
              'foot_separation_rms_before_m': before, 'foot_separation_rms_after_m': after,
              'achieved_separation_ratio': after / before, 'max_foot_target_error_m': max(residuals),
              'mode': 'modified_recording_kinematic_ik',
              'limitation': 'Modified recording with IK-constrained legs; not dynamic balance or measured ground-contact step lengths.'}
    path.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', type=float, default=1.25)
    args = parser.parse_args()
    print(json.dumps(make_longer_walk(args.scale), indent=2))


if __name__ == '__main__':
    main()
