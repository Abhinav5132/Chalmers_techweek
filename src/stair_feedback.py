"""Supervised local feedback distillation for the original G1's known single step.

Fit phase-local linear torque policies to teacher labels around a successful
physical trajectory. Synthetic nearby-state queries teach feedback, unlike
repeating nearly identical successful demonstrations. Runtime uses only the
saved policy, current state and clock; no teacher QP or pose writes.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import mujoco
from stair_curriculum import Teacher, OUT
from stair_learning import run, save_report

VERSION='original-g1-local-feedback-v1'
QUALIFIED=OUT/'feedback-qualified.json'

def qualified_path(height):
    return QUALIFIED if np.isclose(height,.05) else OUT/f'feedback-qualified-{height:.3f}m.json'


class FeedbackStudent:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            if str(z['version'])!=VERSION:raise ValueError('Incompatible feedback policy')
            self.height=float(z['height'])
            self.times=z['times'];self.poses=z['poses'];self.velocities=z['velocities']
            self.actions=z['actions'];self.gains=z['gains'];self.phases=z['phases']
        count=len(self.times)
        expected=[(count,), (count,36),(count,35),(count,29),(count,29,70),(count,)]
        if count<2 or any(a.shape!=shape for a,shape in zip([self.times,self.poses,self.velocities,self.actions,self.gains,self.phases],expected)):
            raise ValueError('Invalid feedback policy dimensions')
        if not np.isfinite(self.height) or self.height<=0 or np.any(np.diff(self.times)<=0):raise ValueError('Invalid policy height or timing')
        if not all(np.isfinite(x).all() for x in [self.times,self.poses,self.velocities,self.actions,self.gains]):
            raise ValueError('Nonfinite policy')

    def action(self,sim):
        if not np.isclose(sim.height,self.height):raise ValueError('Policy is for a different step height')
        phase=sim.targets(sim.d.time)[3]
        ids=np.flatnonzero(self.phases==phase)
        if not len(ids):raise ValueError('Policy has no samples for this phase')
        j=int(np.searchsorted(self.times[ids],sim.d.time))
        left=ids[max(0,j-1)];right=ids[min(j,len(ids)-1)]
        w=0. if left==right else np.clip((sim.d.time-self.times[left])/(self.times[right]-self.times[left]),0,1)
        def predict(i):
            dq=np.zeros(sim.m.nv)
            mujoco.mj_differentiatePos(sim.m,dq,1.,self.poses[i],sim.d.qpos)
            error=np.r_[dq,sim.d.qvel-self.velocities[i]]
            return self.actions[i]+self.gains[i]@error
        return (1-w)*predict(left)+w*predict(right)


def collect(height,seed=300,spacing=10):
    sim=Teacher(height,seed);probe=Teacher(height,seed)
    times=[];poses=[];velocities=[];actions=[];gains=[];phases=[];labels=[]
    scale=np.r_[np.full(sim.m.nv,.002),np.full(sim.m.nv,.02)]
    # Full-rank supervised design: small positive/negative state deviations.
    design=np.vstack([np.diag(scale),-np.diag(scale)])
    step=0;boundaries=np.array([2,5,9,13,17])
    while sim.d.time<23 and not sim.fail:
        tau,phase=sim.action()
        if step%spacing==0 or np.min(np.abs(boundaries-sim.d.time))<.021:
            q=sim.d.qpos.copy();v=sim.d.qvel.copy()
            probe.d.time=sim.d.time;probe.last_qref=sim.reference().copy();probe.reference_time=sim.d.time
            local=[]
            for delta in design:
                probe.d.qpos[:]=q
                mujoco.mj_integratePos(probe.m,probe.d.qpos,delta[:sim.m.nv],1.)
                probe.d.qvel[:]=v+delta[sim.m.nv:]
                mujoco.mj_forward(probe.m,probe.d)
                # Nearby-state queries have no previous trajectory sample.
                probe.last_j={}
                local.append(probe.action()[0])
            y=np.array(local)
            # Symmetric least squares with independent pose/velocity coordinates.
            gain=((y[:len(scale)]-y[len(scale):])/(2*scale[:,None])).T
            times.append(sim.d.time);poses.append(q);velocities.append(v)
            actions.append(tau);gains.append(gain);phases.append(phase);labels.append(y)
            if len(times)%25==0:print(f'Feedback labels: {sim.d.time:.1f}/23 seconds',flush=True)
        sim.step(tau);step+=1
    if sim.fail or sim.top_hold_seconds<2.9:raise RuntimeError('Demonstration failed qualification: '+str(sim.fail))
    return dict(version=VERSION,height=height,times=times,poses=poses,velocities=velocities,
                actions=actions,gains=gains,phases=phases),dict(design=design,labels=labels)


def qualify_candidates(data,folder,report):
    """Select feedback damping on selection seeds; test the winner once on final seeds.

    The high-step landing can produce overly stiff velocity feedback at 100 Hz.
    Regularize that learned term, keeping the fitted pose feedback and nominal
    teacher torques. This is supervised controller selection, not RL.
    """
    height=float(data['height'])
    for damping in ([1.,.75,.5,.25] if height>.05001 else [1.]):
        candidate=dict(data)
        gains=np.array(data['gains'],copy=True)
        gains[np.asarray(data['phases'])=='shift_left',:,35:]*=damping
        candidate['gains']=gains;candidate['landing_velocity_scale']=damping
        policy=folder/f'feedback-policy-damping-{damping:.2f}.npz'
        np.savez_compressed(policy,**candidate)
        student=FeedbackStudent(policy)
        trials=[run(height,seed,student)[0] for seed in [9001,9002,9003]]
        report['evaluations'].append({'label':f'student damping {damping:g}','trials':trials})
        print(f'Damping {damping:g}: {sum(t["passed"] for t in trials)}/3 selection trials',flush=True)
        report['status']='student_failed_physics_evaluation'
        if not all(t['passed'] for t in trials):continue
        final=[run(height,seed,student)[0] for seed in [99001,99002,99003,99004,99005]]
        report['evaluations'].append({'label':'student final','trials':final})
        report['checkpoint']=str(policy);report['landing_velocity_scale']=damping
        report['status']='student_failed_final_evaluation'
        if all(t['passed'] for t in final):
            report['status']='student_passed_known_step'
            qualified_path(height).write_text(json.dumps({'checkpoint':str(policy),'report':str(folder/'results.json')},indent=2)+'\n')
        # Do not tune further against the final held-out seeds.
        return


def train(height=.05):
    folder=OUT/(f'feedback-{height:.3f}m-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));folder.mkdir(parents=True)
    report={'status':'running','method':'supervised local linear feedback distillation',
            'height_m':height,'robot':'original_g1_29dof','base_anchored':False,'evaluations':[],'fits':[],
            'limitations':'Known single step, fixed timing and approach. Nearby-state labels are synthetic teacher queries; successful demonstration and all evaluations use free-base physics. No general terrain claim.'}
    try:
        teachers=[run(height,s)[0] for s in [101,102,103,104,105]]
        report['evaluations'].append({'label':'teacher','trials':teachers})
        if not all(r['passed'] for r in teachers):report['status']='teacher_failed_qualification';return
        data,labels=collect(height)
        policy=folder/'feedback-policy.npz';np.savez_compressed(policy,**data)
        np.savez_compressed(folder/'teacher-labels.npz',**labels)
        report['labelled_states']=int(np.size(labels['labels'])/29)
        qualify_candidates(data,folder,report)
    except BaseException as exc:
        report['status']='error';report['error']=str(exc);raise
    finally:
        save_report(folder,report);print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['train','view']);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--height',type=float,default=.05)
    a=p.parse_args()
    if not np.isfinite(a.height) or not .05<=a.height<=.35:p.error('Height must be between 0.05 and 0.35 metres')
    if a.command=='train':train(a.height)
    else:
        if not a.checkpoint:
            selected=qualified_path(a.height)
            if not selected.exists():p.error(f'No qualified student for {a.height:g} m. Run just step-feedback-train {a.height:g} first.')
            a.checkpoint=Path(json.loads(selected.read_text())['checkpoint'])
        s=FeedbackStudent(a.checkpoint);print(json.dumps(run(s.height,99010,s,view=True)[0],indent=2))

if __name__=='__main__':main()
