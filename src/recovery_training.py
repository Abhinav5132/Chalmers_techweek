"""Teacher-qualified disturbance recovery through supervised DAgger, on CPU."""
from contextlib import nullcontext
from datetime import datetime, timezone
import argparse
import base64
import csv
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import webbrowser
from typing import Any

import mujoco
import numpy as np
import torch
from balance_teacher import BalanceSimulation
from balance_training import BALANCE, save_data
from imitation_g1 import RUN, DT, student, train

RECOVERY = RUN / 'recovery'
KINDS = ('clean', 'push', 'pose', 'friction', 'combined')


class RecoverySimulation(BalanceSimulation):
    def __init__(self):
        super().__init__()
        self.original_friction = self.model.geom_friction.copy()
        feet = [self.model.body(s + '_ankle_roll_link').id for s in ('left', 'right')]
        self.contact_geoms = [i for i in range(self.model.ngeom)
                              if self.model.geom_bodyid[i] in feet or i == self.model.geom('floor').id]
        self.pelvis = self.model.body('pelvis').id
        self.scenario: dict[str, Any] = {}

    def reset_scenario(self, mode, seed, kind, push_force=8.):
        if not np.isfinite(push_force) or not 0 <= push_force <= 100:
            raise ValueError('Push force must be between 0 and 100 N')
        if kind not in KINDS:
            raise ValueError('Unknown disturbance condition')
        self.model.geom_friction[:] = self.original_friction
        self.data.xfrc_applied[:] = 0
        rng = np.random.default_rng(seed)
        self.reset_task(mode, seed, int(rng.integers(0, 36)), disturbance=.04)
        multiplier = float(rng.uniform(.9, 1.1)) if kind in ('friction', 'combined') else 1.
        # Scale floor AND foot friction: contact mixing can otherwise mask changes.
        self.model.geom_friction[self.contact_geoms, 0] *= multiplier
        joint_noise = np.zeros(29)
        if kind in ('pose', 'combined'):
            joint_noise = rng.uniform(-.008, .008, 29)
            self.data.qpos[7:] = np.clip(self.data.qpos[7:] + joint_noise,
                                        self.model.jnt_range[1:, 0], self.model.jnt_range[1:, 1])
            axis = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 0.])
            axis /= max(np.linalg.norm(axis), 1e-8)
            angle = float(rng.uniform(-.015, .015))
            delta = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
            original = self.data.qpos[3:7].copy()
            mujoco.mju_mulQuat(self.data.qpos[3:7], delta, original)
        pushes = []
        if kind in ('push', 'combined'):
            for onset in (2., 6.):
                angle = rng.uniform(-np.pi, np.pi)
                pushes.append({'start_s': onset, 'duration_s': .2,
                               'force_xy_n': [float(push_force * np.cos(angle)), float(push_force * np.sin(angle))]})
        self.scenario = {'seed': seed, 'mode': mode, 'kind': kind, 'friction_multiplier': multiplier,
                         'additional_joint_noise_max_rad': float(np.max(np.abs(joint_noise))), 'pushes': pushes}
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def step(self, action):
        self.data.xfrc_applied[:] = 0
        for push in self.scenario.get('pushes', []):
            if push['start_s'] - 1e-9 <= self.t < push['start_s'] + push['duration_s'] - 1e-9:
                self.data.xfrc_applied[self.pelvis, :2] = push['force_xy_n']
        return super().step(action)


def rollout(sim, policy, mode, seed, kind, record=False, view=False, push_force=8.):
    import mujoco.viewer
    obs = sim.reset_scenario(mode, seed, kind, push_force=push_force)
    origin = sim.data.qpos[:2].copy()
    observations, labels, drift, error = [], [], [], []
    context = mujoco.viewer.launch_passive(sim.model, sim.data) if view else nullcontext(None)
    with context as viewer:
        for _ in range(500):
            tick = time.monotonic()
            if record:
                observations.append(obs.copy()); labels.append(sim.expert(obs).copy())
            obs, fallen = sim.step(policy(obs))
            drift.append(float(np.linalg.norm(sim.data.qpos[:2] - origin)))
            error.append(float(np.linalg.norm(sim.data.xpos[sim.anchor] - sim.reference[2][sim.frame(), sim.anchor_ref])))
            if viewer is not None:
                if not viewer.is_running(): break
                viewer.sync(); time.sleep(max(0., DT - (time.monotonic() - tick)))
            if fallen: break
        if viewer is not None:
            print('Trial complete. Close the window to exit.', flush=True)
            while viewer.is_running(): viewer.sync(); time.sleep(.03)
    result = {**sim.scenario, 'seconds': sim.t, 'fell': fallen, 'max_drift_m': max(drift),
              'tracking_rmse_m': float(np.sqrt(np.mean(np.square(error)))),
              'last_second_tracking_m': float(np.mean(error[-50:]))}
    result['passed'] = bool(not fallen and sim.t >= 9.99 and
                            (max(drift) < .15 if mode == 'stand' else result['tracking_rmse_m'] < .3)
                            and result['last_second_tracking_m'] < (.15 if mode == 'stand' else .3))
    return result, observations, labels


def evaluate(sim, policy, seed_base):
    return [rollout(sim, policy, mode, seed_base + m * 100 + k * 10 + i, kind)[0]
            for m, mode in enumerate(('stand', 'walk')) for k, kind in enumerate(KINDS) for i in range(3)]


def promotion_allowed(before, after):
    # Predeclared validation gate, not a choice among checkpoints using test data.
    for mode in ('stand', 'walk'):
        for kind in KINDS:
            old = [r for r in before if r['mode'] == mode and r['kind'] == kind]
            new = [r for r in after if r['mode'] == mode and r['kind'] == kind]
            if sum(r['passed'] for r in new) < sum(r['passed'] for r in old): return False
            metric = 'max_drift_m' if mode == 'stand' else 'tracking_rmse_m'
            if np.mean([r[metric] for r in new]) > np.mean([r[metric] for r in old]) + .01: return False
    return True


def report(session, results):
    os.environ.setdefault('MPLCONFIGDIR', str(RUN.parent / 'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(RUN.parent / 'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for row_index, split in enumerate(('validation', 'test')):
        for ax, mode in zip(axes[row_index], ('stand', 'walk')):
            for offset, name in enumerate(('teacher', 'before', 'candidate')):
                rows = results['teacher_validation'] if split == 'validation' and name == 'teacher' else results[split][name]
                values = [sum(r['passed'] for r in rows if r['mode'] == mode and r['kind'] == kind) for kind in KINDS]
                ax.bar(np.arange(5) + (offset - 1) * .25, values, .25, label=name)
            ax.set(title=f'{mode.capitalize()} — {split}', ylabel='Passed trials / 3', ylim=(0, 4.1))
            ax.set_xticks(range(5), KINDS, rotation=20); ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.savefig(session / 'recovery_graphs.png', dpi=160); plt.close(fig)
    image = base64.b64encode((session / 'recovery_graphs.png').read_bytes()).decode()
    with (session / 'test_results.csv').open('w', newline='') as stream:
        columns = ['controller', 'mode', 'kind', 'seed', 'seconds', 'fell', 'passed', 'friction_multiplier', 'max_drift_m', 'tracking_rmse_m', 'last_second_tracking_m']
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for name, rows in results['test'].items():
            for row in rows: writer.writerow({k: name if k == 'controller' else row[k] for k in columns})
    (session / 'report.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Recovery training</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}}img{{width:100%}}</style><h1>Disturbance recovery</h1><p>Candidate promoted: <b>{results['promoted']}</b>. Promotion uses a separate validation set; the top graphs show validation and the bottom graphs show the separate final test set.</p><img src="data:image/png;base64,{image}"><p>Ten-second trials. Two 8 N horizontal pushes, each 0.2 seconds, at 2 and 6 seconds; up to 0.008 rad additional joint noise and 0.015 rad base tilt; floor and foot sliding friction multiplied by 0.9–1.1. Conditions are tested separately and together.</p><p>Standing passes with no fall, max drift below 0.15 m and final-second tracking error below 0.15 m. Walking passes with no fall, torso tracking RMSE below 0.30 m and final-second error below 0.30 m. All runs share one standing pose and one walking reference; this is not general recovery certification.</p><a href="results.json">Full results and teacher qualification</a> · <a href="test_results.csv">Test data CSV</a></html>''')


def run(rounds, episodes, epochs):
    torch.set_num_threads(2)
    source = Path(json.loads((BALANCE / 'latest.json').read_text())['session'])
    checkpoint, dataset = source / 'policy.pt', source / 'demonstrations.npz'
    if (RECOVERY / 'active.json').exists():
        active = json.loads((RECOVERY / 'active.json').read_text())
        checkpoint, dataset = Path(active['policy']), Path(active['dataset'])
    session = RECOVERY / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])
    session.mkdir(parents=True)
    shutil.copy2(checkpoint, session / 'initial_policy.pt'); shutil.copy2(checkpoint, session / 'policy.pt')
    sim = RecoverySimulation()
    teacher_validation = evaluate(sim, sim.expert, 4100000)
    (session / 'teacher_validation.json').write_text(json.dumps(teacher_validation, indent=2))
    # Reject entire mode/condition groups whose teacher fails qualification.
    qualified = {(r['mode'], r['kind']) for r in teacher_validation
                 if all(x['passed'] for x in teacher_validation if (x['mode'], x['kind']) == (r['mode'], r['kind']))}
    if not qualified: raise ValueError('Teacher failed every condition; see teacher_validation.json')
    print(f'Teacher qualified {len(qualified)}/10 task/condition groups', flush=True)
    before_validation = evaluate(sim, student(session / 'initial_policy.pt'), 4100000)
    with np.load(dataset, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in ('observations', 'actions', 'episode_ids')}
    qualification = []
    for iteration in range(rounds):
        policy = student(session / 'policy.pt')
        states, actions, ids = [], [], []
        first_id = int(arrays['episode_ids'].max()) + 1
        for episode in range(episodes):
            mode = ('stand', 'walk')[episode % 2]
            kind = KINDS[(episode // 2) % 5]
            if (mode, kind) not in qualified: continue
            seed = 5000000 + first_id * 10 + episode
            # Verify the teacher on the exact starting pose, friction and pushes.
            teacher_result, teacher_obs, teacher_actions = rollout(sim, sim.expert, mode, seed, kind, record=True)
            qualification.append(teacher_result)
            if not teacher_result['passed']: continue
            student_result, student_obs, student_actions = rollout(sim, policy, mode, seed, kind, record=True)
            for observations, labels in ((teacher_obs, teacher_actions), (student_obs, student_actions)):
                states.extend(observations); actions.extend(labels)
                # Pair teacher and student from one scenario in the same split.
                ids.extend([first_id + episode] * len(observations))
        if not states: raise ValueError('No teacher-qualified corrective data collected')
        added = {'observations': np.asarray(states, np.float32), 'actions': np.asarray(actions, np.float32), 'episode_ids': np.asarray(ids)}
        arrays = {k: np.concatenate((arrays[k], added[k])) for k in arrays}
        save_data(session / 'demonstrations.npz', arrays)
        (session / 'training_qualification.json').write_text(json.dumps(qualification, indent=2))
        print(f'Round {iteration + 1}: {len(states)} new examples', flush=True)
        train(epochs, resume=True, dataset=session / 'demonstrations.npz', output=session)
        shutil.copy2(session / 'policy.pt', session / f'round-{iteration + 1}.pt')
    candidate_validation = evaluate(sim, student(session / 'policy.pt'), 4100000)
    promoted = promotion_allowed(before_validation, candidate_validation)
    # Final test seeds are not used for training, qualification or promotion.
    results = {'promoted': promoted, 'teacher_validation': teacher_validation,
               'validation': {'before': before_validation, 'candidate': candidate_validation},
               'test': {'teacher': evaluate(sim, sim.expert, 4200000),
                        'before': evaluate(sim, student(session / 'initial_policy.pt'), 4200000),
                        'candidate': evaluate(sim, student(session / 'policy.pt'), 4200000)}}
    (session / 'results.json').write_text(json.dumps(results, indent=2))
    report(session, results)
    (RECOVERY / 'latest.json').write_text(json.dumps({'session': str(session)}))
    if promoted:
        (RECOVERY / 'active.json').write_text(json.dumps({'policy': str(session / 'policy.pt'), 'dataset': str(session / 'demonstrations.npz')}))
    elif not (RECOVERY / 'active.json').exists():
        (RECOVERY / 'active.json').write_text(json.dumps({'policy': str(checkpoint), 'dataset': str(dataset)}))
    print(json.dumps({'promoted': promoted, 'test_passed': {k: sum(r['passed'] for r in v) for k, v in results['test'].items()}, 'report': str(session / 'report.html')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'view', 'results'])
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--candidate', action='store_true', help='Preview the latest candidate, even if rejected')
    parser.add_argument('--mode', choices=['stand', 'walk'], default='stand')
    parser.add_argument('--push-force', type=float, default=8., help='Preview push strength in newtons (0–100); training remains at 8 N')
    parser.add_argument('--condition', choices=KINDS, default='combined')
    args = parser.parse_args()
    if args.rounds < 1 or args.episodes < 10 or args.epochs < 1: parser.error('Positive rounds/epochs and at least 10 episodes required')
    if not np.isfinite(args.push_force) or not 0 <= args.push_force <= 100:
        parser.error('Push force must be between 0 and 100 N')
    if args.command == 'train': run(args.rounds, args.episodes, args.epochs)
    elif args.command == 'results':
        session = Path(json.loads((RECOVERY / 'latest.json').read_text())['session'])
        webbrowser.open((session / 'report.html').as_uri())
    else:
        torch.set_num_threads(2)
        active = json.loads((RECOVERY / 'active.json').read_text())
        path = Path(active['policy'])
        if args.candidate:
            latest = json.loads((RECOVERY / 'latest.json').read_text())
            path = Path(latest['session']) / 'policy.pt'
        print(f'Preview policy: {path}', flush=True)
        rollout(RecoverySimulation(), student(path), args.mode, 4300000, args.condition, view=True, push_force=args.push_force)


if __name__ == '__main__': main()
