"""Create an offline, shareable report from saved imitation results."""
from __future__ import annotations
import argparse
import base64
import csv
import html
import json
import os
from pathlib import Path
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '.robot-runtime/imitation'


def build_report(directory: Path = RUN) -> Path:
    # Keep the plotting cache in the project, including on sandboxed machines.
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.robot-runtime/matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.robot-runtime/cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    evaluation = json.loads((directory / 'evaluation.json').read_text())
    trials = evaluation['trials']
    with (directory / 'training.csv').open(newline='') as stream:
        training = list(csv.DictReader(stream))
    summary = json.loads((directory / 'training.json').read_text())
    if not trials or not training:
        raise ValueError('Training and evaluation must contain data')
    groups = sorted({t['controller'] for t in trials}, key=lambda name: name != 'teacher')
    keys = sorted({(t['seed'], t['start_frame']) for t in trials})
    colors = {'teacher': '#2563eb', 'student': '#e07a24'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    fig.suptitle('G1 behavioural cloning — training and physical evaluation', fontsize=16)
    ax = axes[0, 0]
    for column, label, color in [('train_mse', 'Training', '#2563eb'), ('validation_mse', 'Validation', '#e07a24')]:
        ax.plot([int(r['epoch']) for r in training], [float(r[column]) for r in training], label=label, color=color)
    ax.set(title='Action prediction error (lower is better)', xlabel='Epoch', ylabel='Mean squared error')
    ax.set_yscale('log'); ax.legend()
    width = .8 / len(groups)
    for ax, metric, title, ylabel in [
        (axes[0, 1], 'seconds', 'Time before fall or trial end', 'Seconds'),
        (axes[1, 0], 'displacement_m', 'Horizontal displacement (not step length)', 'Metres'),
        (axes[1, 1], 'valid_forward_steps_fixed_x', 'Supported forward touchdowns along world +X', 'Count')]:
        for index, group in enumerate(groups):
            rows = {(t['seed'], t['start_frame']): t for t in trials if t['controller'] == group}
            values = [rows[k][metric] if k in rows else np.nan for k in keys]
            bars = ax.bar(np.arange(len(keys)) - .4 + width / 2 + index * width, values, width, label=group.capitalize(), color=colors.get(group))
            if metric == 'seconds':
                for bar, key in zip(bars, keys):
                    if key in rows:
                        ax.annotate('Fell' if rows[key]['fell'] else 'Trial end', (bar.get_x() + width / 2, bar.get_height()), ha='center', va='bottom', fontsize=8)
                ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 6)
        ax.set(title=title, ylabel=ylabel)
        ax.set_xticks(range(len(keys)), [f'Trial {i + 1}' for i in range(len(keys))])
        ax.legend()
    # Both groups determine the duration axis, even if the student falls early.
    axes[0, 1].set_ylim(0, max(t['seconds'] for t in trials) * 1.35)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.savefig(directory / 'graphs.png', dpi=170)
    fig.savefig(directory / 'graphs.svg')
    plt.close(fig)
    columns = ['controller', 'seed', 'start_frame', 'seconds', 'fell', 'displacement_m', 'valid_forward_steps_fixed_x']
    with (directory / 'evaluation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows({k: t[k] for k in columns} for t in trials)
    def download(name, label, mime):
        data = base64.b64encode((directory / name).read_bytes()).decode()
        return f'<a download="{name}" href="data:{mime};base64,{data}">{label}</a>'
    image = base64.b64encode((directory / 'graphs.png').read_bytes()).decode()
    cards = ''.join(f'<div class="card"><b>{html.escape(group.capitalize())}</b><strong>{sum(bool(t["fell"]) for t in trials if t["controller"] == group)} / {sum(t["controller"] == group for t in trials)} falls</strong></div>' for group in groups)
    rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(f"{t[k]:.3f}" if isinstance(t[k], float) else str(t[k]))}</td>' for k in columns) + '</tr>' for t in trials)
    links = ' '.join(download(*item) for item in [('graphs.png', 'Save graph image', 'image/png'), ('graphs.svg', 'Save vector graph', 'image/svg+xml'), ('evaluation.csv', 'Evaluation data (CSV)', 'text/csv'), ('training.csv', 'Training data (CSV)', 'text/csv'), ('evaluation.json', 'Evaluation JSON', 'application/json')])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>G1 learning results</title>
<style>body{{font:16px system-ui,sans-serif;background:#f5f7fb;color:#17253b;max-width:1200px;margin:40px auto;padding:0 24px}}h1{{margin-bottom:8px}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:white;border:1px solid #dce2ec;border-radius:12px;padding:20px;min-width:180px}}strong{{display:block;font-size:26px;margin-top:8px}}img{{width:100%;background:white;border-radius:12px;margin:20px 0}}a{{display:inline-block;background:#e2eafa;color:#174481;padding:10px;margin:4px;border-radius:6px}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}.table{{overflow:auto}}.note{{background:#fff0d9;padding:18px;border-radius:10px;line-height:1.6}}</style>
<h1>G1 learning results</h1><p>Behavioural cloning · CPU training · {summary['epochs']} epochs · {summary['training_samples']:,} training examples · {summary['validation_samples']:,} validation examples</p>
<div class="cards">{cards}<div class="card"><b>Best validation error</b><strong>{summary['best_validation_mse']:.6f}</strong></div></div>
<p class="note">{html.escape(evaluation['limitation'])} Low prediction error does not prove stable walking. Displacement includes motion before a fall and is not achieved step length. Training and validation share one source motion; only episodes differ. The teacher was RL-trained; the student uses supervised learning.</p>
<img alt="Training error and teacher versus student evaluation graphs" src="data:image/png;base64,{image}">
<h2>Save graphs and data</h2>{links}<h2>Individual trials</h2><p>Trial numbers in the graphs match the ordered seed/start-frame pairs below. All trials have a five-second limit.</p><div class="table"><table><thead><tr>{''.join(f'<th>{html.escape(k)}</th>' for k in columns)}</tr></thead><tbody>{rows}</tbody></table></div>
<p>This page includes its graph and downloadable data. It works offline and can be shared as a single HTML file.</p></html>'''
    target = directory / 'report.html'
    target.write_text(page)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open', action='store_true', help='Open the report in your browser')
    args = parser.parse_args()
    try:
        path = build_report()
    except FileNotFoundError as error:
        parser.exit(1, f'Missing results: {error.filename}. Run just imitation-train first.\n')
    print(path)
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == '__main__':
    main()
