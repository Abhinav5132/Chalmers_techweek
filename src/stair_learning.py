"""CPU supervised residual imitation and DAgger, gated at every stair height.

Student outputs residual motor torques around joint PD, not pose playback. The
Pinocchio planner remains in both teacher and student; QP is teacher-only.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from stair_curriculum import Teacher, OUT, DT

FEATURES=110
VERSION='original-g1-step-wbc-v2'
KP=np.r_[np.tile([60,60,40,80,40,40],2),[40]*3,[20]*14]
KD=np.r_[np.tile([8,8,5,10,4,4],2),[5]*3,[2]*14]


def inputs(sim):
    feet,com,stance,_=sim.targets(sim.d.time)
    q=sim.reference(feet,com)
    d=sim.d
    x=np.r_[(q-d.qpos[7:])/.05,d.qvel[6:]/1,d.qpos[3:7]/[1,.1,.1,.1],d.qvel[:6]/[.2,.2,.2,.5,.5,.5],
            (com-d.subtree_com[1])/.02,(feet-d.xpos[sim.ids]).ravel()/.02,
            stance,sim.height/.1,min(d.time/23,1),q]
    assert len(x)==FEATURES
    baseline=KP*(q-d.qpos[7:])-KD*d.qvel[6:]
    return x.astype(np.float32),baseline


def network():return nn.Sequential(nn.Linear(FEATURES,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,29))


class Student:
    def __init__(self,path):
        ck=torch.load(path,weights_only=True,map_location='cpu')
        if ck.get('version')!=VERSION:raise ValueError('Incompatible robot/controller checkpoint')
        self.height=ck['height'];self.net=network();self.net.load_state_dict(ck['weights']);self.net.eval()
    def action(self,sim):
        if not np.isclose(sim.height,self.height):raise ValueError('Student was trained for a different stair height')
        x,baseline=inputs(sim)
        with torch.no_grad(): residual=self.net(torch.from_numpy(x)).numpy()
        return baseline+np.clip(residual,-1,1)*sim.m.actuator_ctrlrange[:,1]


def run(height,seed,student=None,collect=False,view=False,beta=0.):
    sim=Teacher(height,seed);xs=[];ys=[];viewer=None
    rng=np.random.default_rng(seed+200000)
    if view:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(sim.m,sim.d)
        viewer.cam.distance=2.3+height;viewer.cam.lookat[:]=[.3,0,.65+height/2]
    try:
        while sim.d.time<23 and not sim.fail:
            start=time.monotonic()
            try:
                if collect or student is None:
                    x,base=inputs(sim);tau,_=sim.action()
                    if collect:xs.append(x);ys.append((tau-base)/sim.m.actuator_ctrlrange[:,1])
                action=tau if student is None else student.action(sim)
                if student is not None and collect:action=beta*tau+(1-beta)*action
                # Small external pushes diversify corrective demonstrations. No state teleportation.
                sim.d.xfrc_applied[:]=0
                if collect and int(sim.d.time*100)%100<20:
                    sim.d.xfrc_applied[1,:2]=rng.normal(0,2,2)
                sim.step(action)
            except (RuntimeError,ValueError) as exc:sim.fail=str(exc);break
            if viewer:
                if not viewer.is_running():sim.fail='viewer_closed';break
                viewer.sync();time.sleep(max(0,DT-time.monotonic()+start))
    finally:
        if viewer:viewer.close()
    passed=not sim.fail and sim.d.time>=23 and sim.top_hold_seconds>=2.9
    report={'seed':seed,'height_m':height,'passed':passed,'failure':sim.fail,
            'seconds':float(sim.d.time),'top_hold_seconds':sim.top_hold_seconds,
            'minimum_up_dot':sim.min_up,'controller':'teacher' if student is None else 'student',
            'max_torque_ratio':sim.max_torque_ratio,
            'teacher_mixing_beta':beta if collect else 0.}
    return report,xs,ys,sim


def fit(x,y,val,path,height,epochs,initial=None):
    torch.set_num_threads(1);torch.manual_seed(42)
    net=network()
    if initial:net.load_state_dict(torch.load(initial,weights_only=True)['weights'])
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    X=torch.as_tensor(np.array(x),dtype=torch.float32);Y=torch.as_tensor(np.array(y),dtype=torch.float32)
    V=torch.as_tensor(val,dtype=torch.bool);train=torch.where(~V)[0]
    if not V.any() or not len(train):raise ValueError('Need separate training and validation episodes')
    best=float('inf');hist=[]
    for epoch in range(epochs):
        net.train();losses=[]
        for ids in train[torch.randperm(len(train))].split(512):
            loss=(net(X[ids])-Y[ids]).square().mean();opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        net.eval()
        with torch.no_grad():vl=(net(X[V])-Y[V]).square().mean().item()
        hist.append({'epoch':epoch+1,'train_mse':float(np.mean(losses)),'validation_mse':vl})
        if vl<best:
            best=vl;torch.save({'version':VERSION,'height':height,'weights':net.state_dict()},path)
    return hist


def save_report(folder,report):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    with (folder/'evaluations.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['label','height_m','seed','passed','failure','seconds','top_hold_seconds'])
        writer.writeheader()
        for group in report.get('evaluations',[]):
            for trial in group['trials']:
                writer.writerow({'label':group['label'],**{k:trial[k] for k in writer.fieldnames if k!='label'}})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'training-latest.json').write_text(json.dumps({'directory':str(folder),'status':report['status']},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    groups=report.get('evaluations',[])
    axes[0].bar(range(len(groups)),[sum(r['passed'] for r in g['trials'])/len(g['trials']) for g in groups])
    axes[0].set_xticks(range(len(groups)),[g['label'] for g in groups],rotation=25,ha='right')
    axes[0].set_ylabel('Successful climbs / trials');axes[0].set_ylim(0,1.05)
    for h in report.get('fits',[]):axes[1].plot([r['validation_mse'] for r in h['history']],label=h['label'])
    axes[1].set_xlabel('Epoch');axes[1].set_ylabel('Held-out residual torque MSE')
    if not report.get('fits'):
        axes[1].bar(range(len(groups)),[np.mean([r['seconds'] for r in g['trials']]) for g in groups])
        axes[1].set_xticks(range(len(groups)),[g['label'] for g in groups],rotation=25,ha='right')
        axes[1].set_xlabel('Physical evaluation');axes[1].set_ylabel('Mean seconds completed');axes[1].axhline(23,color='gray',linestyle='--')
    if report.get('fits'):axes[1].legend()
    fig.tight_layout();fig.savefig(folder/'results.png',dpi=150);plt.close(fig)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>G1 stair curriculum</title><h1>G1 stair curriculum</h1><p>'+report['status']+'</p><img src="results.png" width="1000"><p>Only repeated physical climbs with loaded feet and a final hold count as successes. Training loss alone is not success.</p><a href="results.json">Full results</a>')


def train(args):
    folder=OUT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');folder.mkdir(parents=True)
    report={'status':'running','robot':'original_g1_29dof','method':'supervised residual imitation + DAgger',
            'base_anchored':False,'evaluations':[],'fits':[],'stages':[],'collection_episodes':[],
            'configuration':{'heights':args.heights,'epochs':args.epochs,'episodes':args.episodes,'rounds':args.rounds,'initial_joint_noise_rad':.0005},
            'limitations':'Single known step, fixed approach and schedule; no unseen-terrain or real-hardware claim.'}
    save_report(folder,report)
    previous_stage=None
    try:
        for height in args.heights:
            stage=folder/f'{height:.3f}m';stage.mkdir()
            teachers=[]
            for seed in [101,102,103,104,105]:
                r=run(height,seed)[0];teachers.append(r)
                print(f'{height*100:g}cm teacher seed {seed}: {r["passed"]}',flush=True)
            report['evaluations'].append({'label':f'{height*100:g}cm teacher','trials':teachers})
            if not all(r['passed'] for r in teachers):
                report['status']='teacher_failed_qualification';save_report(folder,report);break
            X=[];Y=[];V=[];episodes=[]
            for i in range(args.episodes):
                r,x,y,_=run(height,1000+i,collect=True)
                report['collection_episodes'].append({**r,'round':0,'examples':len(x)})
                print(f'Teacher demonstration {i+1}/{args.episodes}: {r["passed"]}',flush=True)
                if not r['passed']:
                    report['status']='teacher_failed_collection';save_report(folder,report);return
                X.extend(x);Y.extend(y);V.extend([i%4==0]*len(x));episodes.extend([i]*len(x))
            student=None;passed=False
            for rd in range(args.rounds+1):
                if rd:
                    for i in range(args.episodes):
                        r,x,y,_=run(height,2000+100*rd+i,student,collect=True,beta=max(0.,.9-.2*(rd-1)))
                        report['collection_episodes'].append({**r,'round':rd,'examples':len(x)})
                        X.extend(x);Y.extend(y);V.extend([False]*len(x));episodes.extend([rd*100+i]*len(x))
                np.savez_compressed(stage/'demonstrations.npz',observations=X,residuals=Y,validation=V,episode=episodes)
                path=stage/f'policy-round-{rd}.pt'
                history=fit(X,Y,V,path,height,args.epochs,stage/f'policy-round-{rd-1}.pt' if rd else previous_stage)
                report['fits'].append({'label':f'{height*100:g}cm round {rd}','history':history,'examples':len(X)})
                student=Student(path)
                evaluations=[run(height,seed,student)[0] for seed in [9001,9002,9003]]
                report['evaluations'].append({'label':f'{height*100:g}cm student {rd}','trials':evaluations})
                save_report(folder,report)
                print(f'{height*100:g}cm round {rd}: {sum(r["passed"] for r in evaluations)}/3 successful climbs',flush=True)
                if all(r['passed'] for r in evaluations):
                    # Separate final seeds are never used to decide which round to train next.
                    final=[run(height,seed,student)[0] for seed in [99001,99002,99003,99004,99005]]
                    report['evaluations'].append({'label':f'{height*100:g}cm final','trials':final})
                    if all(r['passed'] for r in final):
                        passed=True;previous_stage=path;report['stages'].append({'height_m':height,'checkpoint':str(path),'passed':True});break
                    report['status']='student_failed_final_evaluation';break
            if not passed:
                if report['status']=='running':report['status']='student_failed_physics_evaluation'
                break
        else:report['status']='requested_curriculum_passed'
    except KeyboardInterrupt:report['status']='interrupted'
    except Exception as exc:
        report['status']='error';report['error']=str(exc);raise
    finally:save_report(folder,report)
    print(str(folder/'index.html'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['train','view-student','results'])
    p.add_argument('--heights',type=float,nargs='+',default=[.05,.075,.10]);p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--episodes',type=int,default=8);p.add_argument('--rounds',type=int,default=4);p.add_argument('--checkpoint',type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    if a.command=='train':
        if not np.isfinite(a.heights).all() or a.heights[0]!=.05 or any(h<.05 or h>.20 for h in a.heights) or any(b<=x or b-x>.02501 for x,b in zip(a.heights,a.heights[1:])):p.error('Start at 0.05 m and increase by at most 0.025 m, up to 0.20 m')
        if a.episodes<4 or a.epochs<1 or a.rounds<0:p.error('Need at least 4 episodes, 1 epoch, and nonnegative DAgger rounds')
        train(a)
    elif a.command=='results':
        latest=json.loads((OUT/'training-latest.json').read_text());print(str(Path(latest['directory'])/'index.html'))
    else:
        if not a.checkpoint:p.error('Specify --checkpoint from a results.json stage or candidate round')
        if a.checkpoint.suffix=='.npz':
            from stair_feedback import FeedbackStudent
            student=FeedbackStudent(a.checkpoint)
        else:student=Student(a.checkpoint)
        print(json.dumps(run(student.height,99010,student,view=True)[0],indent=2))

if __name__=='__main__':main()
