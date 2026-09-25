"""Supervised teacher/student standing and ten-second walking, with DAgger."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import webbrowser

import numpy as np
import torch
from balance_teacher import BalanceSimulation
from imitation_g1 import DATA, RUN, student, train, DT

BALANCE = RUN / 'balance'


def rollout(sim, policy, mode, seed, start=0, record=False, view=False):
    from contextlib import nullcontext
    import mujoco.viewer
    obs = sim.reset_task(mode, seed, start, disturbance=.04)
    origin = sim.data.qpos[:2].copy()
    observations, labels, drift, tracking = [], [], [], []
    context = mujoco.viewer.launch_passive(sim.model, sim.data) if view else nullcontext(None)
    with context as viewer:
        for _ in range(500):
            tick = time.monotonic()
            if record:
                observations.append(obs.copy()); labels.append(sim.expert(obs).copy())
            obs, fallen = sim.step(policy(obs))
            drift.append(float(np.linalg.norm(sim.data.qpos[:2] - origin)))
            tracking.append(float(np.linalg.norm(sim.data.xpos[sim.anchor] - sim.reference[2][sim.frame(), sim.anchor_ref])))
            if viewer is not None:
                if not viewer.is_running(): break
                viewer.sync(); time.sleep(max(0., DT - (time.monotonic() - tick)))
            if fallen: break
        if viewer is not None:
            print('Trial complete; close the window to exit.', flush=True)
            while viewer.is_running():
                viewer.sync(); time.sleep(.03)
    result = {'mode': mode, 'seed': seed, 'start_frame': start, 'seconds': sim.t, 'fell': fallen,
              'max_horizontal_drift_m': max(drift), 'torso_tracking_rmse_m': float(np.sqrt(np.mean(np.square(tracking))))}
    # Stationary success includes drift; walking includes tracking, not just survival.
    result['passed'] = bool(not fallen and sim.t >= 9.99 and
                            (max(drift) < .15 if mode == 'stand' else result['torso_tracking_rmse_m'] < .3))
    return result, observations, labels


def evaluate(sim, policy, output):
    results = []
    for name, controller in [('teacher', sim.expert), ('student', policy)]:
        for mode in ('stand', 'walk'):
            for trial in range(6):
                result, _, _ = rollout(sim, controller, mode, 900000 + trial, start=trial * 6)
                results.append({'controller': name, **result})
    (output / 'balance_evaluation.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({f'{name}_{mode}_passed': sum(bool(r['passed']) for r in results if r['controller'] == name and r['mode'] == mode)
                      for name in ('teacher', 'student') for mode in ('stand', 'walk')}), flush=True)
    return results


def save_data(path, arrays):
    np.savez_compressed(path, observations=arrays['observations'], actions=arrays['actions'], episode_ids=arrays['episode_ids'])


def gather(sim, policy, arrays, episodes, seed_base, successful_only=False):
    first = int(arrays['episode_ids'].max()) + 1
    obs, acts, ids, reports = [], [], [], []
    for episode in range(episodes):
        mode = 'stand' if episode % 2 == 0 else 'walk'
        start = int(np.random.default_rng(seed_base + episode).integers(0, 36))
        result, states, actions = rollout(sim, policy, mode, seed_base + episode, start, record=True)
        reports.append(result)
        if not successful_only or result['passed']:
            obs.extend(states); acts.extend(actions); ids.extend([first + episode] * len(states))
    if not obs:
        raise ValueError('No usable demonstrations; teacher validation failed')
    new = {'observations': np.array(obs, np.float32), 'actions': np.array(acts, np.float32), 'episode_ids': np.array(ids)}
    return {k: np.concatenate((arrays[k], new[k])) for k in arrays}, reports


def report(session):
    os.environ.setdefault('MPLCONFIGDIR', str(RUN.parent / 'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(RUN.parent / 'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import base64
    history = json.loads((session / 'progress.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for ax, mode in zip(axes[0], ('stand', 'walk')):
        ax.plot(range(len(history)), [sum(r['passed'] for r in rows if r['controller'] == 'student' and r['mode'] == mode) for rows in history], marker='o', label='Student')
        ax.plot(range(len(history)), [sum(r['passed'] for r in rows if r['controller'] == 'teacher' and r['mode'] == mode) for rows in history], linestyle='--', label='Teacher')
        ax.set(title='Standing' if mode == 'stand' else 'Walking', xlabel='Stage (0 = original student)', ylabel='Passed ten-second trials / 6', ylim=(-.2, 6.5))
        ax.set_xticks(range(len(history))); ax.legend(); ax.grid(alpha=.2)
    for ax, mode, metric, title in [(axes[1, 0], 'stand', 'max_horizontal_drift_m', 'Mean maximum standing drift'), (axes[1, 1], 'walk', 'torso_tracking_rmse_m', 'Mean walking tracking RMSE')]:
        for name in ('student', 'teacher'):
            ax.plot(range(len(history)), [float(np.mean([r[metric] for r in rows if r['controller'] == name and r['mode'] == mode])) for rows in history], marker='o', label=name.capitalize())
        ax.set(title=title, xlabel='Stage', ylabel='Metres (lower is better)')
        ax.set_xticks(range(len(history))); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(session / 'balance_graphs.png', dpi=160); plt.close(fig)
    image = base64.b64encode((session / 'balance_graphs.png').read_bytes()).decode()
    rows = ''.join(f'<tr><td>{r["controller"]}</td><td>{r["mode"]}</td><td>{r["seed"]}</td><td>{r["seconds"]:.2f}</td><td>{r["fell"]}</td><td>{r["max_horizontal_drift_m"]:.3f}</td><td>{r["torso_tracking_rmse_m"]:.3f}</td><td>{r["passed"]}</td></tr>' for r in history[-1])
    (session / 'report.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Standing and walking training</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}}img{{width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}</style><h1>Standing and walking — teacher/student results</h1><p>Supervised learning and DAgger. Each trial lasts ten seconds. Standing passes with no fall and maximum drift below 0.15 m; walking passes with no fall and torso tracking RMSE below 0.30 m.</p><img src="data:image/png;base64,{image}"><p>Stage 0 is the incoming student; stage 1 adds teacher demonstrations; later stages add DAgger corrections. One fixed standing pose and one jazz-walk reference. Small initial velocity perturbations only. No stand/walk transitions, arbitrary destinations, step-length commands or real-world validation. Evaluation seeds are excluded from training, but repeatedly checked across stages.</p><a href="progress.json">All stage data (JSON)</a><table><tr><th>Controller</th><th>Task</th><th>Seed</th><th>Seconds</th><th>Fell</th><th>Max drift m</th><th>Tracking RMSE m</th><th>Passed</th></tr>{rows}</table>''')


def run(rounds, epochs, episodes):
    torch.set_num_threads(2)
    session = BALANCE / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])
    session.mkdir(parents=True)
    sim = BalanceSimulation()
    source = DATA / 'dagger_demonstrations.npz'
    checkpoint = RUN / 'policy.pt'
    if (BALANCE / 'latest.json').exists():
        previous = Path(json.loads((BALANCE / 'latest.json').read_text())['session'])
        source, checkpoint = previous / 'demonstrations.npz', previous / 'policy.pt'
    if not source.exists(): source = DATA / 'demonstrations.npz'
    with np.load(source, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in ('observations', 'actions', 'episode_ids')}
    shutil.copy2(checkpoint, session / 'policy.pt')
    shutil.copy2(checkpoint, session / 'initial_policy.pt')
    history = [evaluate(sim, student(session / 'policy.pt'), session)]
    # Teacher validation gate: do not train either new task from a failing teacher.
    if not all(r['passed'] for r in history[0] if r['controller'] == 'teacher'):
        raise ValueError(f'Teacher failed qualification; see {session}')
    arrays, collected = gather(sim, sim.expert, arrays, episodes, 2000000 + int(arrays['episode_ids'].max()) * 10, successful_only=True)
    (session / 'teacher_collection.json').write_text(json.dumps(collected, indent=2))
    dataset = session / 'demonstrations.npz'; save_data(dataset, arrays)
    train(epochs, resume=True, dataset=dataset, output=session)
    history.append(evaluate(sim, student(session / 'policy.pt'), session))
    for iteration in range(rounds):
        output = session / f'round-{iteration + 1:02d}'; output.mkdir()
        arrays, collected = gather(sim, student(session / 'policy.pt'), arrays, episodes, 3000000 + int(arrays['episode_ids'].max()) * 10)
        (output / 'collection.json').write_text(json.dumps(collected, indent=2))
        save_data(dataset, arrays)
        train(epochs, resume=True, dataset=dataset, output=session)
        history.append(evaluate(sim, student(session / 'policy.pt'), output))
        shutil.copy2(session / 'policy.pt', output / 'policy.pt')
        (session / 'progress.json').write_text(json.dumps(history, indent=2))
        print(f'Balance DAgger round {iteration + 1}: {len(arrays["actions"])} examples', flush=True)
    (session / 'balance_evaluation.json').write_text(json.dumps(history[-1], indent=2))
    report(session)
    (BALANCE / 'latest.json').write_text(json.dumps({'session': str(session)}))
    print(f'Results: {session / "report.html"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'view', 'results'])
    parser.add_argument('--mode', choices=['stand', 'walk'], default='stand')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    if min(args.rounds, args.epochs) < 1 or args.episodes < 10:
        parser.error('Need positive rounds/epochs and at least 10 episodes')
    if args.command == 'train': run(args.rounds, args.epochs, args.episodes)
    else:
        session = Path(json.loads((BALANCE / 'latest.json').read_text())['session'])
        if args.command == 'results': webbrowser.open((session / 'report.html').as_uri())
        else:
            torch.set_num_threads(2)
            rollout(BalanceSimulation(), student(session / 'policy.pt'), args.mode, 900000, view=True)


if __name__ == '__main__': main()
