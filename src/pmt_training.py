"""Supervised PMT terrain distillation on CPU, with teacher qualification.

A saved candidate is never installed as the active walking/robot-agent policy.
Evaluation uses held-out seeds on the selected clips, not unseen stair geometry.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any

import mujoco
import numpy as np
import torch
from torch import nn

from pmt_assets import REVISION, prepare
from pmt_simulation import OUT, ROBOT_ID, Simulation, Teacher, catalog

FEATURES = 1926


def features(obs):
    # Policy layout: 10x58 command, 10x58 future command, 10x3 anchor
    # position, 10x6 anchor orientation; histories are oldest -> newest.
    result = np.concatenate([obs['proprio_history'].ravel(),
        obs['command_window'].ravel(), obs['motion_anchor_delta_window'].ravel(),
        obs['anchor_body_pose'], obs['policy'][1187:1190], obs['policy'][1244:1250]])
    if result.shape != (FEATURES,) or not np.isfinite(result).all():
        raise ValueError('Invalid student feature vector')
    return result


def network():
    return nn.Sequential(nn.Linear(FEATURES,256),nn.ELU(),nn.Linear(256,256),
                         nn.ELU(),nn.Linear(256,29))


def support_heights(sim):
    ground = sim.model.geom('pmt_ground').id
    feet = {sim.model.body(f'{side}_ankle_roll_link').id for side in ['left','right']}
    heights = []
    for i, c in enumerate(sim.data.contact):
        if ground not in (c.geom1,c.geom2):
            continue
        other = c.geom2 if c.geom1 == ground else c.geom1
        if sim.model.geom_bodyid[other] not in feet or abs(c.frame[2]) < .8:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(sim.model,sim.data,i,force)
        if force[0] > 5:
            heights.append(float(c.pos[2]))
    return heights


def rollout(sim, controller, seed, teacher=None, viewer=None):
    obs = sim.reset(seed)
    errors, supports, xs, ys = [], [], [], []
    fallen = False
    for tick in range(sim.row['frames']):
        if viewer is not None and not viewer.is_running():
            break
        start = time.monotonic()
        label = teacher(obs) if teacher is not None else None
        if label is not None:
            xs.append(features(obs));ys.append(label)
        action = label if controller is teacher and label is not None else controller(obs)
        obs, fallen, error = sim.step(action)
        errors.append(error)
        supports.extend((tick,z) for z in support_heights(sim))
        if viewer is not None:
            viewer.cam.lookat[:] = sim.data.qpos[:3]
            viewer.sync()
            time.sleep(max(0,.02-(time.monotonic()-start)))
        if fallen:
            break
    first = [z for tick,z in supports if tick<50]
    last = [z for tick,z in supports if tick>=sim.row['frames']-50]
    gain = float(np.median(last)-np.median(first)) if first and last else None
    levels = {}
    for tick,z in supports:
        levels.setdefault(round(z/.02)*.02,set()).add(tick)
    supported_levels = sorted(round(z,2) for z,ticks in levels.items() if len(ticks)>=10)
    mean = float(np.mean(errors)) if errors else None
    complete = sim.elapsed == sim.row['frames']
    success = bool(complete and not fallen and gain is not None and gain>=.12
                   and mean is not None and mean<.25 and max(errors)<.5)
    report = {'clip':sim.row['clip'],'seed':seed,'seconds':sim.elapsed/50,
        'fell':fallen,'completed_segment':complete,'qualified_ascent':success,
        'support_height_gain_m':gain,'supported_height_levels_m':supported_levels,
        'mean_anchor_error_m':mean,
        'max_anchor_error_m':max(errors) if errors else None}
    return report, xs, ys


class Student:
    def __init__(self, path):
        c = torch.load(path,map_location='cpu',weights_only=True)
        if c.get('robot_id') != ROBOT_ID:
            raise ValueError('This checkpoint was trained on a different robot. Train a new candidate for the original G1.')
        if c['assets_revision'] != REVISION or c['feature_version'] != 1:
            raise ValueError('Incompatible PMT student checkpoint')
        self.net=network();self.net.load_state_dict(c['weights']);self.net.eval()
        self.mean,self.std=c['mean'],c['std']

    def __call__(self, obs):
        with torch.inference_mode():
            x=torch.from_numpy(features(obs))
            return self.net((x-self.mean)/self.std).numpy().copy()


def fit(dataset, output, epochs, resume=False):
    d=np.load(dataset,allow_pickle=False)
    x=torch.tensor(d['observations']); y=torch.tensor(d['actions'])
    validation=torch.tensor(d['validation'].astype(bool))
    if x.ndim!=2 or x.shape[1]!=FEATURES or y.shape!=(len(x),29) or validation.shape!=(len(x),):
        raise ValueError('Invalid PMT training array shapes')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite PMT training data')
    if not validation.any() or validation.all():
        raise ValueError('Training requires disjoint training and validation episodes')
    net=network();optimizer=torch.optim.Adam(net.parameters(),lr=.0005)
    checkpoint=output/'policy.pt'
    if resume and checkpoint.exists():
        c=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net.load_state_dict(c['weights']);mean,std=c['mean'],c['std']
        optimizer.load_state_dict(c['optimizer'])
    else:
        mean=x[~validation].mean(0);std=x[~validation].std(0).clamp_min(.01)
    x=(x-mean)/std
    train=torch.where(~validation)[0]; val=torch.where(validation)[0]
    with torch.no_grad():best=float(((net(x[val])-y[val])**2).mean()) if resume else float('inf')
    history=[]
    for epoch in range(epochs):
        net.train();total=0.
        for ids in train[torch.randperm(len(train))].split(256):
            loss=((net(x[ids])-y[ids])**2).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            total+=float(loss.detach())*len(ids)
        net.eval()
        with torch.no_grad():vl=float(((net(x[val])-y[val])**2).mean())
        history.append({'epoch':epoch+1,'train_mse':total/len(train),'validation_mse':vl})
        if vl<best:
            best=vl
            torch.save({'weights':net.state_dict(),'mean':mean,'std':std,
                'optimizer':optimizer.state_dict(),'assets_revision':REVISION,
                'feature_version':1,'robot_id':ROBOT_ID,'purpose':'experimental_terrain_student'},checkpoint)
        if (epoch+1)%10==0:print(history[-1],flush=True)
    return history


def qualify(teacher, candidates=30, limit=3):
    selected,reports=[],[]
    # Fixed order by reference gain; selection is teacher-only, never student score.
    for rank,row in enumerate(catalog()[:candidates]):
        sim=Simulation(row)
        trials=[rollout(sim,teacher,0)[0]]
        if trials[0]['qualified_ascent']:
            trials.extend(rollout(sim,teacher,seed)[0] for seed in [1,2])
        reports.extend(trials)
        ok=all(r['qualified_ascent'] for r in trials)
        print(f"Teacher candidate {rank}: {'accepted' if ok else 'rejected'}; {trials[0]['seconds']} s",flush=True)
        if ok:selected.append(row)
        if len(selected)>=limit:break
    return selected,reports


def evaluate(rows, controller, seeds):
    reports=[]
    for row in rows:
        sim=Simulation(row)
        reports.extend(rollout(sim,controller,seed)[0] for seed in seeds)
    return reports


def write_report(output, result):
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (output/'training.csv').open('w', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['round','epoch','train_mse','validation_mse'])
        writer.writeheader()
        for i,history in enumerate(result.get('training',[])):
            writer.writerows(dict(round=i,**row) for row in history)
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,history in enumerate(result.get('training',[])):
        axes[0].plot([r['validation_mse'] for r in history],label=f'round {i}')
    axes[0].set(title='Held-out episode action error',xlabel='Epoch',ylabel='MSE')
    if result.get('training'):axes[0].legend()
    groups=[('teacher',result.get('teacher_test',[])),('student',result.get('student_test',[]))]
    axes[1].bar([g[0] for g in groups],[sum(r['qualified_ascent'] for r in g[1]) for g in groups])
    axes[1].set(title='Successful held-out-seed ascent segments',ylabel='Trials passed')
    fig.tight_layout();fig.savefig(output/'results.png');plt.close(fig)
    (output/'report.html').write_text('''<!doctype html><meta charset="utf-8"><title>PMT terrain imitation</title>
<h1>PMT terrain imitation — original repository G1</h1><p>CPU MuJoCo transfer; six-second terrain ascent segments.
Held-out seeds use the same selected clips. This does not establish unseen-stair generalization.
No candidate is automatically installed as the robot agent's active policy.</p>
<p>Each round adds corrective states to the dataset. Loss values across rounds
are measured on different, expanding datasets and are not a fixed-benchmark comparison.</p>
<img src="results.png" style="max-width:100%"><p><a href="results.json">Full results</a> · <a href="training.csv">Training CSV</a></p>''' + '<p>Status: ' + result['status'] + '</p>')


def train(args):
    torch.manual_seed(42);torch.set_num_threads(1)
    output=OUT/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6])
    output.mkdir(parents=True)
    teacher=Teacher()
    rows,qualification=qualify(teacher,args.candidates,args.clips)
    result: dict[str, Any]={'assets_revision':REVISION,'robot_id':ROBOT_ID,
            'terrain_sample_spacing_m':.01,'feature_version':1,'selected_clips':rows,'qualification':qualification,
            'training':[],'promoted':False,'evaluation_scope':'held-out seeds on selected reference clips'}
    (output/'config.json').write_text(json.dumps({'selected_clips':rows},indent=2))
    (OUT/'latest.json').write_text(json.dumps({'run':str(output)},indent=2)+'\n')
    if not rows:
        result['status']='blocked_no_qualified_teacher_demonstrations'
        write_report(output,result)
        print(f'No teacher ascent passed qualification. Training not started. Report: {output}',flush=True)
        return
    xs,ys,validation,episode_ids=[],[],[],[]
    episode=0
    for round_index in range(args.rounds+1):
        controller=teacher if round_index==0 else Student(output/'policy.pt')
        for row in rows:
            sim=Simulation(row)
            for index in range(args.episodes):
                seed=1000+index+round_index*100
                # Exact reset must be solvable by teacher before querying student states.
                accepted,_,_=rollout(sim,teacher,seed)
                if not accepted['qualified_ascent']:
                    print(f'Round {round_index}, reset {seed}: teacher rejected',flush=True)
                    continue
                report,x,y=rollout(sim,controller,seed,teacher=teacher)
                xs.extend(x);ys.extend(y)
                validation.extend([index%5==0]*len(x))
                episode_ids.extend([episode]*len(x));episode+=1
                print(f'Round {round_index}, episode {episode}: {len(x)} labels; student/teacher ascent={report["qualified_ascent"]}',flush=True)
        if not xs:
            result['status']='blocked_no_qualified_collection_episodes'
            break
        np.savez_compressed(output/'demonstrations.npz',observations=np.asarray(xs,dtype=np.float32),
            actions=np.asarray(ys,dtype=np.float32),validation=np.asarray(validation),
            episode_ids=np.asarray(episode_ids))
        print(f'Training round {round_index}: {len(xs)} teacher-labelled examples',flush=True)
        history=fit(output/'demonstrations.npz',output,args.epochs,resume=round_index>0)
        result['training'].append(history)
        torch.save(torch.load(output/'policy.pt',weights_only=True),output/f'round_{round_index}.pt')
        result['status']='training'
        write_report(output,result)
    if (output/'policy.pt').exists():
        result['teacher_test']=evaluate(rows,teacher,[90000,90001,90002])
        result['student_test']=evaluate(rows,Student(output/'policy.pt'),[90000,90001,90002])
        result['status']='candidate_passed_selected_segments' if all(
            r['qualified_ascent'] for r in result['student_test']) else 'candidate_failed_physics_evaluation'
        result['examples']=len(xs)
    write_report(output,result)
    (OUT/'latest.json').write_text(json.dumps({'run':str(output)},indent=2)+'\n')
    print(f'Results: {output}/report.html',flush=True)
    print('Student successes:',sum(r['qualified_ascent'] for r in result.get('student_test',[])),
          '/',len(result.get('student_test',[])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['setup','train','view-teacher','view-student','results'])
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--episodes',type=int,default=10)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--candidates',type=int,default=30)
    parser.add_argument('--clips',type=int,default=2)
    parser.add_argument('--rank',type=int,default=0)
    args=parser.parse_args()
    if args.epochs<1 or args.episodes<6 or args.rounds<0 or args.candidates<1 or args.clips<1 or args.rank<0:
        parser.error('epochs/candidates/clips >=1, episodes >=6, rounds/rank >=0 required')
    if args.command=='setup':prepare();return
    if args.command=='train':train(args);return
    if args.command=='view-teacher':
        rows=catalog()
        if args.rank>=len(rows):parser.error('rank is outside downloaded catalog')
        row=rows[args.rank];controller=Teacher()
    else:
        pointer=OUT/'latest.json'
        if not pointer.exists():parser.error('No PMT training run. Run just pmt-train first.')
        output=Path(json.loads(pointer.read_text())['run'])
        if args.command=='results':
            import webbrowser
            webbrowser.open((output/'report.html').as_uri());return
        selected=json.loads((output/'config.json').read_text())['selected_clips']
        if not selected or not (output/'policy.pt').exists():
            parser.error('Teacher qualification has not produced a trained PMT student. Open just pmt-results for details.')
        row=selected[0]
        controller=Student(output/'policy.pt')
    sim=Simulation(row);sim.reset(90000)
    import mujoco.viewer
    with mujoco.viewer.launch_passive(sim.model,sim.data) as viewer:
        viewer.cam.distance=3.5
        print(json.dumps(rollout(sim,controller,90000,viewer=viewer)[0],indent=2),flush=True)
        print('Trial ended; simulation frozen. Close the window to exit.',flush=True)
        while viewer.is_running():viewer.sync();time.sleep(.05)


if __name__=='__main__':main()
