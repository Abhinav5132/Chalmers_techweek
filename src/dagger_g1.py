"""DAgger: execute the student, label visited states with teacher actions, aggregate.

No reward optimization. Teacher labels on recovery states are not guaranteed to
be recoverable. Evaluation seeds are excluded from collection; source motion is
shared, so results do not demonstrate new-motion or step-command generalization.
"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch
from imitation_g1 import DATA, RUN, WalkingSimulation, student, train, evaluate


def collect_corrections(sim, policy, episodes, first_id, seed_base):
    observations, actions, ids, reports = [], [], [], []
    for episode in range(episodes):
        seed = seed_base + episode
        start = int(np.random.default_rng(seed).integers(0, 250))
        obs = sim.reset(seed, start)
        for _ in range(250):
            # Label the exact observation visited, but execute ONLY the student.
            label = sim.expert(obs)
            action = policy(obs)
            observations.append(obs.copy()); actions.append(label.copy())
            ids.append(first_id + episode)
            obs, fallen = sim.step(action)
            if fallen:
                break
        reports.append({'seed': seed, 'start_frame': start, 'fell': fallen, 'seconds': sim.t})
    return {'observations': np.array(observations, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'episode_ids': np.array(ids)}, reports


def run(rounds, episodes, epochs):
    torch.set_num_threads(2)
    session = RUN / 'dagger' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])
    session.mkdir(parents=True)
    # Preserve the complete pre-update experiment before changing the active policy.
    baseline = session / 'baseline'
    baseline.mkdir()
    for name in ('policy.pt', 'training.csv', 'training.json'):
        shutil.copy2(RUN / name, baseline / name)
    evaluate(output=baseline)
    aggregate = DATA / 'dagger_demonstrations.npz'
    source = aggregate if aggregate.exists() else DATA / 'demonstrations.npz'
    with np.load(source, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in ('observations', 'actions', 'episode_ids')}
    shutil.copy2(source, session / 'initial_demonstrations.npz')
    sim = WalkingSimulation()
    history = []
    previous = baseline
    for iteration in range(1, rounds + 1):
        output = session / f'round-{iteration:02d}'
        output.mkdir()
        first_id = int(arrays['episode_ids'].max()) + 1
        added, reports = collect_corrections(sim, student(previous / 'policy.pt'), episodes,
                                             first_id, 100000 + first_id)
        np.savez_compressed(output / 'corrections.npz', **added)
        (output / 'collection.json').write_text(json.dumps(reports, indent=2))
        arrays = {k: np.concatenate((arrays[k], added[k])) for k in arrays}
        np.savez_compressed(aggregate, observations=arrays['observations'], actions=arrays['actions'], episode_ids=arrays['episode_ids'])
        shutil.copy2(previous / 'policy.pt', output / 'policy.pt')
        print(f'DAgger round {iteration}: collected {len(added["actions"])} corrections; {len(arrays["actions"])} total examples', flush=True)
        train(epochs, resume=True, dataset=aggregate, output=output)
        evaluate(output=output)
        trials = json.loads((output / 'evaluation.json').read_text())['trials']
        rows = [t for t in trials if t['controller'] == 'student']
        history.append({'round': iteration, 'new_examples': len(added['actions']),
                        'total_examples': len(arrays['actions']), 'falls': sum(t['fell'] for t in rows),
                        'trials': len(rows), 'mean_seconds': float(np.mean([t['seconds'] for t in rows])),
                        'mean_steps': float(np.mean([t['valid_forward_steps_fixed_x'] for t in rows]))})
        (session / 'progress.json').write_text(json.dumps(history, indent=2))
        previous = output
    # Activate the last round, not a policy selected using evaluation outcomes.
    for name in ('policy.pt', 'training.csv', 'training.json', 'evaluation.json', 'evaluation.csv', 'report.html', 'graphs.png', 'graphs.svg'):
        shutil.copy2(previous / name, RUN / name)
    (RUN / 'dagger_latest.json').write_text(json.dumps({'session': str(session), 'rounds': history,
        'method': 'Pure student rollouts with teacher action labels; supervised DAgger',
        'limitation': 'Same reference and three 5-second evaluation trials; not a general balance or standing policy.'}, indent=2))
    print(f'DAgger complete. History: {session}\nActive report: {RUN / "report.html"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    if args.rounds < 1 or args.episodes < 5 or args.epochs < 1:
        parser.error('Use at least 1 round, 5 episodes and 1 epoch')
    run(args.rounds, args.episodes, args.epochs)


if __name__ == '__main__':
    main()
