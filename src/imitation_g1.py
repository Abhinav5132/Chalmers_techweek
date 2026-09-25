"""CPU behavioural cloning from a public G1 teacher; no reward-based training.

Teacher metadata defines observations/actions. All rollouts use motor torques and
mj_step; qpos is assigned only at reset. Source teacher itself was RL-trained.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import mujoco
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/imitation'
RUN = ROOT / '.robot-runtime/imitation'
URL = 'https://huggingface.co/datasets/exptech/g1-moves/resolve/main/dance/J_ShortDance16_JazzWalk/policy/J_ShortDance16_JazzWalk.onnx'
SHA = '854784c9be5ec1b0a7c83d1ce14d319b1a562a92a3e3868005e631a2f74d8461'
DT = .02


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / 'walk_teacher_bundle.onnx'
    if not path.exists():
        temporary = path.with_suffix('.download')
        urllib.request.urlretrieve(URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA:
            raise ValueError('Teacher checksum mismatch; source changed. Review before using.')
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA:
        raise ValueError('Teacher checksum mismatch')
    return path


def rotation(quat):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


class WalkingSimulation:
    def __init__(self):
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.teacher = ort.InferenceSession(str(prepare()), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.meta = self.teacher.get_modelmeta().custom_metadata_map
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / 'unitree_mujoco/unitree_robots/g1/scene_29dof.xml'))
        self.model.opt.timestep = .002
        self.data = mujoco.MjData(self.model)
        names = self.meta['joint_names'].split(',')
        assert names == [self.model.joint(i).name for i in range(1, 30)]
        assert self.meta['observation_names'] == 'command,motion_anchor_pos_b,motion_anchor_ori_b,base_lin_vel,base_ang_vel,joint_pos,joint_vel,actions'
        for key in ('joint_stiffness', 'joint_damping', 'default_joint_pos', 'action_scale'):
            setattr(self, key, np.fromstring(self.meta[key], sep=','))
        self.anchor = self.model.body(self.meta['anchor_body_name']).id
        self.anchor_ref = self.meta['body_names'].split(',').index(self.meta['anchor_body_name'])
        # The bundled export contains the reference; avoid a separate joint/body mapping.
        references = []
        for frame in range(636):
            references.append(self.teacher.run(None, {'obs': np.zeros((1, 160), np.float32),
                'time_step': np.array([[frame]], np.float32)})[1:])
        self.reference = [np.concatenate([r[i] for r in references]) for i in range(6)]
        self.fps = 60.0
        self.last = np.zeros(29)
        self.t = 0.0
        self.start = 0

    def reset(self, seed=0, start=0):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.start, self.t = int(start), 0.0
        jp, jv, pos, quat, lin, ang = self.reference
        self.data.qpos[:3] = pos[start, 0]
        self.data.qpos[3:7] = quat[start, 0]
        self.data.qpos[7:] = jp[start] + rng.normal(0, .003, 29)
        self.data.qvel[:3] = lin[start, 0]
        self.data.qvel[3:6] = rotation(quat[start, 0]).T @ ang[start, 0]
        self.data.qvel[6:] = jv[start]
        self.last[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def frame(self):
        return min(635, self.start + int(round(self.t * self.fps)))

    def observation(self):
        d = self.data
        jp, jv, pos, quat, _, _ = self.reference
        f = self.frame()
        R = d.xmat[self.anchor].reshape(3, 3)
        anchor_pos = R.T @ (pos[f, self.anchor_ref] - d.xpos[self.anchor])
        anchor_ori = (R.T @ rotation(quat[f, self.anchor_ref]))[:, :2].reshape(-1)
        # Match IMU site-local velocities, rather than indexing unrelated joint sensors.
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_SITE,
                                self.model.site('imu').id, velocity, 1)
        obs = np.concatenate((jp[f], jv[f], anchor_pos, anchor_ori,
                              velocity[3:], velocity[:3],
                              d.qpos[7:] - self.default_joint_pos, d.qvel[6:], self.last))
        if not np.isfinite(obs).all():
            raise ValueError('Nonfinite simulator observation')
        return obs.astype(np.float32)

    def expert(self, obs):
        return np.asarray(self.teacher.run(['actions'], {'obs': obs[None],
            'time_step': np.array([[self.frame()]], np.float32)})[0])[0]

    def step(self, action):
        if np.shape(action) != (29,) or not np.isfinite(action).all():
            raise ValueError('Expected 29 finite action values')
        target = self.default_joint_pos + self.action_scale * action
        target = np.clip(target, self.model.jnt_range[1:, 0], self.model.jnt_range[1:, 1])
        for _ in range(10):
            torque = self.joint_stiffness * (target - self.data.qpos[7:]) - self.joint_damping * self.data.qvel[6:]
            self.data.ctrl[:] = np.clip(torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.last = np.array(action, copy=True)
        self.t += DT
        fallen = bool(self.data.qpos[2] < .45 or self.data.xmat[1].reshape(3, 3)[2, 2] < .5)
        return self.observation(), fallen


def collect(episodes):
    sim = WalkingSimulation()
    observations, actions, episode_ids, reports = [], [], [], []
    for episode in range(episodes):
        start = int(np.random.default_rng(episode).integers(0, 250))
        obs = sim.reset(episode, start)
        initial = sim.data.qpos[:2].copy()
        for _ in range(250):
            action = sim.expert(obs)
            observations.append(obs); actions.append(action); episode_ids.append(episode)
            obs, fallen = sim.step(action)
            if fallen:
                break
        reports.append({'episode': episode, 'seconds': sim.t, 'fell': fallen,
                        'displacement_m': float(np.linalg.norm(sim.data.qpos[:2] - initial))})
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA / 'demonstrations.npz', observations=np.array(observations),
                        actions=np.array(actions), episode_ids=np.array(episode_ids))
    report = {'source_url': URL, 'sha256': SHA, 'license': 'CC-BY-4.0', 'credit': 'Experiential Technologies / G1 Moves contributors', 'simulation_timestep': .002, 'control_timestep': DT, 'reference_fps': 60, 'method': 'Supervised distillation from an RL-trained public teacher',
              'samples': len(actions), 'episodes': reports,
              'limitation': 'Jazz-walk reference, not straight commanded-step demonstrations. Failed teacher rollouts retained and explicitly reported; not certified expert successes.'}
    (DATA / 'demonstrations.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'samples': len(actions), 'teacher_falls': sum(r['fell'] for r in reports), 'episodes': episodes}), flush=True)


def network():
    return nn.Sequential(nn.Linear(160, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 29))


def train(epochs, resume=False, dataset=None, output=RUN):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    with np.load(dataset or DATA / 'demonstrations.npz', allow_pickle=False) as d:
        x, y, ids = [torch.tensor(d[k]) for k in ('observations', 'actions', 'episode_ids')]
    validation = ids % 5 == 0
    if not validation.any() or validation.all():
        raise ValueError('Collect at least 5 episodes for separate training and validation episodes')
    mean, std = x[~validation].mean(0), x[~validation].std(0).clamp_min(.05)
    model = network()
    checkpoint = None
    if resume:
        checkpoint = torch.load(output / 'policy.pt', map_location='cpu', weights_only=True)
        if checkpoint['teacher_sha256'] != SHA:
            raise ValueError('Checkpoint teacher version mismatch')
        model.load_state_dict(checkpoint['weights'])
        mean, std = checkpoint['mean'], checkpoint['std']
    x = (x - mean) / std
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    if checkpoint is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    output.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    started = time.monotonic()
    with (output / 'training.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch', 'train_mse', 'validation_mse'])
        writer.writeheader()
        for epoch in range(epochs):
            model.train()
            train_ids = torch.where(~validation)[0]
            permutation = train_ids[torch.randperm(len(train_ids))]
            total = 0.
            for batch in permutation.split(256):
                loss = (model(x[batch]) - y[batch]).square().mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += loss.item() * len(batch)
            model.eval()
            with torch.no_grad():
                val = (model(x[validation]) - y[validation]).square().mean().item()
            row = {'epoch': epoch + 1, 'train_mse': total / len(train_ids), 'validation_mse': val}
            writer.writerow(row); file.flush()
            if val < best:
                best = val
                torch.save({'weights': model.state_dict(), 'mean': mean, 'std': std,
                            'epoch': epoch + 1, 'validation_mse': val, 'teacher_sha256': SHA, 'optimizer': optimizer.state_dict()}, output / 'policy.pt')
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(json.dumps(row), flush=True)
    summary = {'method': 'behavioural_cloning', 'resumed': resume, 'epochs': epochs, 'seconds': time.monotonic() - started,
               'best_validation_mse': best, 'training_samples': int((~validation).sum()),
               'validation_samples': int(validation.sum()), 'split': 'whole episodes; same reference motion',
               'limitation': 'Action prediction error does not establish balance or step-length control.'}
    (output / 'training.json').write_text(json.dumps(summary, indent=2))


def student(path=RUN / 'policy.pt'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['teacher_sha256'] != SHA:
        raise ValueError('Checkpoint teacher version mismatch')
    model = network(); model.load_state_dict(checkpoint['weights']); model.eval()
    def predict(obs):
        with torch.no_grad():
            return model((torch.from_numpy(obs) - checkpoint['mean']) / checkpoint['std']).numpy()
    return predict


def evaluate(view=False, output=RUN):
    from contextlib import nullcontext
    from step_experiment import TouchdownDetector
    torch.set_num_threads(2)
    sim = WalkingSimulation()
    predict = student(output / 'policy.pt')
    results = []
    for name, policy in [('teacher', sim.expert), ('student', predict)]:
        if view and name == 'teacher':
            continue
        for trial, start in enumerate((0,) if view else (0, 100, 200)):
            obs = sim.reset(10000 + trial, start)
            initial = sim.data.qpos[:2].copy()
            detector = TouchdownDetector()
            events = []
            if view:
                import mujoco.viewer
                context = mujoco.viewer.launch_passive(sim.model, sim.data)
            else:
                context = nullcontext(None)
            with context as viewer:
                for frame in range(250):
                    tick = time.monotonic()
                    obs, fallen = sim.step(policy(obs))
                    ids = {s: sim.model.body(s + '_ankle_roll_link').id for s in ('left', 'right')}
                    contacts = dict.fromkeys(ids, False)
                    floor = sim.model.geom('floor').id
                    for c in sim.data.contact:
                        if c.dist <= 0 and floor in (c.geom1, c.geom2):
                            other = c.geom2 if c.geom1 == floor else c.geom1
                            for side, body in ids.items():
                                contacts[side] |= sim.model.geom_bodyid[other] == body
                    events.extend(detector.update(frame, contacts, {s: sim.data.xpos[b].copy() for s, b in ids.items()}, np.array([1., 0.])))
                    if viewer is not None:
                        if not viewer.is_running():
                            return
                        viewer.sync(); time.sleep(max(0, DT - (time.monotonic() - tick)))
                    if fallen:
                        break
                if viewer is not None:
                    print('Trial finished. Close the viewer to exit.', flush=True)
                    while viewer.is_running():
                        viewer.sync(); time.sleep(.03)
            results.append({'controller': name, 'seed': 10000 + trial, 'start_frame': start,
                            'seconds': sim.t, 'fell': fallen, 'displacement_m': float(np.linalg.norm(sim.data.qpos[:2] - initial)),
                            'valid_forward_steps_fixed_x': int(sum(e['valid_forward_step'] for e in events))})
    if not view:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'evaluation.json').write_text(json.dumps({'trials': results,
            'limitation': '5 second trials on the same jazz-walk reference; not generalization to commanded step lengths.'}, indent=2))
        print(json.dumps(results, indent=2))
        from imitation_report import build_report
        print(f'Graphs and data: {build_report(output)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'collect', 'train', 'evaluate', 'view', 'all'])
    parser.add_argument('--resume', action='store_true', help='Continue saved weights and optimizer (if available)')
    parser.add_argument('--episodes', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    if args.episodes < 5 or args.epochs < 1:
        parser.error('Need at least 5 episodes and 1 epoch')
    if args.command == 'prepare': prepare()
    if args.command in ('collect', 'all'): collect(args.episodes)
    if args.command in ('train', 'all'): train(args.epochs, resume=args.resume)
    if args.command in ('evaluate', 'all'): evaluate()
    if args.command == 'view': evaluate(view=True)


if __name__ == '__main__':
    main()
