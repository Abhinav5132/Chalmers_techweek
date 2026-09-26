"""CPU supervised two-hand box lifting with a free-base G1 and free object.

The original rubber-hand visual meshes become collision surfaces in this scene.
No finger actuators, object weld, root anchor or runtime pose playback is used.
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import mujoco
import pinocchio as pin
from stair_curriculum import Teacher, SCENE, ROOT, DT, smooth

OUT = ROOT / '.robot-runtime/imitation/grasp'
HANDS = ('left_wrist_yaw_link', 'right_wrist_yaw_link')
OFFSET = np.array([.10, 0., 0.])
DURATION = 12.


def make_model():
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.option.timestep = .002
    for geom in spec.geoms:
        if geom.meshname in ('left_rubber_hand', 'right_rubber_hand'):
            geom.contype = 1
            geom.conaffinity = 1
    spec.worldbody.add_geom(name='grasp_table', type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=[.32, 0, .36], size=[.055, .04, .36], rgba=[.4,.3,.2,1])
    obj = spec.worldbody.add_body(name='grasp_object', pos=[.32,0,.782])
    obj.add_freejoint(name='object_free')
    obj.add_geom(name='object_box', type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[.055,.055,.06], mass=.15, friction=[1,.005,.0001], rgba=[.95,.5,.1,1])
    return spec.compile()


class GraspTeacher(Teacher):
    def __init__(self, seed=0):
        super().__init__(.05, seed, stand_only=True)
        self.hand_ids = [self.m.body(n).id for n in HANDS]
        self.initial_hands = np.array([self.d.xpos[b]+self.d.xmat[b].reshape(3,3)@OFFSET for b in self.hand_ids])

    def hand_targets(self):
        t=self.d.time
        # Reach beside the box, close both palms, then lift and hold.
        width = .13 - .11*smooth((t-3)/2)
        z = .79 + .12*smooth((t-6)/3)
        target=np.array([[.32,width,z],[.32,-width,z]])
        s=smooth((t-1)/2)
        return (1-s)*self.initial_hands+s*target

    def extra_tasks(self, task):
        for k,bid in enumerate(self.hand_ids):
            R=self.d.xmat[bid].reshape(3,3)
            pt=self.d.xpos[bid]+R@OFFSET
            J,jdv=self.jac(pt,bid,('hand',k))
            err=np.r_[self.hand_targets()[k]-pt,pin.log3(R.T)]
            task(J,200*err-28*(J@self.d.qvel)-jdv,15)


class Simulation:
    def __init__(self, seed=0):
        self.m=make_model();self.d=mujoco.MjData(self.m)
        reset=GraspTeacher(seed)
        self.d.qpos[:36]=reset.d.qpos
        rng=np.random.default_rng(seed)
        self.d.qpos[36:39]=[.32+rng.uniform(-.001,.001),rng.uniform(-.001,.001),.782]
        self.d.qpos[39:43]=[1,0,0,0]
        mujoco.mj_forward(self.m,self.d)
        self.object_id=self.m.body('grasp_object').id
        self.object_geom=self.m.geom('object_box').id
        self.table=self.m.geom('grasp_table').id
        self.hands=[self.m.body(n).id for n in HANDS]
        self.fail=None;self.hold=0.;self.max_torque=0.;self.min_up=1.;self.trace=[]
        self.initial_z=float(self.d.xpos[self.object_id,2])

    def sync_teacher(self, teacher):
        teacher.d.qpos[:]=self.d.qpos[:36]
        teacher.d.qvel[:]=self.d.qvel[:35]
        teacher.d.time=self.d.time
        mujoco.mj_forward(teacher.m,teacher.d)

    def contacts(self):
        hands=np.zeros(2);table=0.;other=0.
        for i in range(self.d.ncon):
            c=self.d.contact[i]
            if self.object_geom not in (c.geom1,c.geom2):continue
            other_geom=c.geom2 if c.geom1==self.object_geom else c.geom1
            f=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,i,f)
            load=max(0.,f[0]);bid=self.m.geom_bodyid[other_geom]
            if other_geom==self.table:table+=load
            elif bid in self.hands:hands[self.hands.index(bid)]+=load
            else:other+=load
        return hands,table,other

    def step(self, action):
        action=np.asarray(action)
        if action.shape!=(29,) or not np.isfinite(action).all():raise ValueError('Expected 29 finite motor torques')
        self.d.ctrl[:]=np.clip(action,self.m.actuator_ctrlrange[:,0],self.m.actuator_ctrlrange[:,1])
        self.max_torque=max(self.max_torque,float(np.max(np.abs(self.d.ctrl)/self.m.actuator_ctrlrange[:,1])))
        for _ in range(5):mujoco.mj_step(self.m,self.d)
        up=float(self.d.xmat[1,8]);self.min_up=min(self.min_up,up)
        pos=self.d.xpos[self.object_id].copy();hands,table,other=self.contacts()
        if up<.8 or self.d.qpos[2]<.6:self.fail='robot_fall'
        if pos[2]<.70:self.fail='object_dropped'
        if not np.isfinite(self.d.qpos).all():self.fail='nonfinite_state'
        good=pos[2]-self.initial_z>.08 and np.all(hands>.1) and table<.05 and other<.05 and up>.95
        self.hold=self.hold+DT if good and self.d.time>=9 else 0.
        self.trace.append([self.d.time,*pos,up,*hands,table,other,self.hold,self.max_torque])

    def result(self,seed,controller):
        passed=not self.fail and self.d.time>=DURATION and self.hold>=2.
        return dict(seed=seed,controller=controller,passed=bool(passed),failure=self.fail or (None if passed else 'lift_hold_not_achieved'),
                    seconds=float(self.d.time),hold_seconds=self.hold,lift_m=float(self.d.xpos[self.object_id,2]-self.initial_z),
                    minimum_up_dot=self.min_up,max_torque_ratio=self.max_torque,base_anchored=False,object_attached=False)


def rollout(seed=0, student=None, view=False):
    sim=Simulation(seed);teacher=GraspTeacher(seed) if student is None else None
    viewer=None
    if view:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(sim.m,sim.d)
        viewer.cam.lookat[:]=[.2,0,.85];viewer.cam.distance=2.;viewer.cam.azimuth=135
    try:
        while sim.d.time<DURATION and not sim.fail:
            start=time.monotonic()
            if teacher is not None:
                sim.sync_teacher(teacher)
                try:action=teacher.action()[0]
                except RuntimeError as exc:
                    sim.fail=str(exc);break
            else:action=student.action(sim)
            sim.step(action)
            if viewer:
                if not viewer.is_running():sim.fail='viewer_closed';break
                viewer.sync();time.sleep(max(0,DT-(time.monotonic()-start)))
    finally:
        if viewer:viewer.close()
    return sim.result(seed,'teacher' if student is None else 'student'),sim


VERSION='g1-two-hand-local-feedback-v1'
QUALIFIED=OUT/'qualified.json'
TRACE_HEADER='time,object_x,object_y,object_z,up,left_contact_N,right_contact_N,table_contact_N,other_contact_N,hold_seconds,max_torque_ratio'


class Student:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            if str(z['version'])!=VERSION:raise ValueError('Incompatible grasp policy')
            self.times=z['times'];self.poses=z['poses'];self.velocities=z['velocities']
            self.actions=z['actions'];self.gains=z['gains']
        n=len(self.times)
        expected=[(n,), (n,36), (n,35), (n,29), (n,29,70)]
        for value,shape in zip([self.times,self.poses,self.velocities,self.actions,self.gains],expected):
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid grasp policy arrays')
        if n<2 or np.any(np.diff(self.times)<=0):raise ValueError('Invalid policy times')

    def action(self,sim):
        j=int(np.searchsorted(self.times,sim.d.time))
        left=max(0,j-1);right=min(j,len(self.times)-1)
        w=0. if left==right else np.clip((sim.d.time-self.times[left])/(self.times[right]-self.times[left]),0,1)
        def predict(i):
            q=sim.d.qpos.copy();q[:36]=self.poses[i]
            dq=np.zeros(sim.m.nv)
            mujoco.mj_differentiatePos(sim.m,dq,1.,q,sim.d.qpos)
            return self.actions[i]+self.gains[i]@np.r_[dq[:35],sim.d.qvel[:35]-self.velocities[i]]
        return (1-w)*predict(left)+w*predict(right)


def collect(folder):
    sim=Simulation(300);teacher=GraspTeacher(300);probe=GraspTeacher(300)
    times=[];poses=[];velocities=[];actions=[];gains=[];labels=[]
    scale=np.r_[np.full(35,.002),np.full(35,.02)]
    design=np.vstack([np.diag(scale),-np.diag(scale)])
    step=0
    while sim.d.time<DURATION and not sim.fail:
        sim.sync_teacher(teacher);tau=teacher.action()[0]
        if step%10==0:
            q=teacher.d.qpos.copy();v=teacher.d.qvel.copy()
            probe.d.time=sim.d.time
            probe.last_qref=teacher.reference().copy();probe.reference_time=sim.d.time
            local=[]
            for delta in design:
                probe.d.qpos[:]=q
                mujoco.mj_integratePos(probe.m,probe.d.qpos,delta[:35],1.)
                probe.d.qvel[:]=v+delta[35:]
                mujoco.mj_forward(probe.m,probe.d)
                probe.last_j={}
                local.append(probe.action()[0])
            y=np.array(local)
            gains.append(((y[:70]-y[70:])/(2*scale[:,None])).T)
            labels.append(y);times.append(sim.d.time);poses.append(q);velocities.append(v);actions.append(tau)
            if len(times)%20==0:print(f'Teacher labels: {sim.d.time:.1f}/{DURATION:g} simulated seconds',flush=True)
        sim.step(tau);step+=1
    result=sim.result(300,'demonstration')
    np.savetxt(folder/'demonstration-trace.csv',sim.trace,delimiter=',',header=TRACE_HEADER,comments='')
    if not result['passed']:raise RuntimeError('Demonstration failed: '+str(result))
    np.savez_compressed(folder/'teacher-labels.npz',design=design,labels=labels)
    data: dict[str, Any]=dict(version=VERSION,times=times,poses=poses,velocities=velocities,actions=actions,gains=gains)
    np.savez_compressed(folder/'demonstration-policy.npz',**data)
    return data,result


def save_report(folder,report):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (folder/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    rows=report['trials']
    with (folder/'evaluations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['group','seed','controller','passed','failure','seconds','hold_seconds','lift_m','minimum_up_dot','max_torque_ratio','base_anchored','object_attached'])
        writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    groups=list(dict.fromkeys(r['group'] for r in rows))
    for ax,key,title in zip(axes,['passed','lift_m','hold_seconds'],['Success fraction','Final object lift (m)','Final continuous hold (s)']):
        ax.bar(groups,[np.mean([r[key] for r in rows if r['group']==g]) for g in groups])
        ax.set_title(title);ax.tick_params(axis='x',rotation=35)
    axes[0].set_ylim(0,1.05);fig.tight_layout();fig.savefig(folder/'results.png',dpi=150);plt.close(fig)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>G1 grasp evaluation</title><h1>G1 two-hand lift</h1><p>'+report['status']+'</p><img src="results.png" style="max-width:100%"><p><a href="results.json">Full results</a> · <a href="evaluations.csv">Trial CSV</a></p><p>Known 150 g box and pedestal, small initial variations. No finger grasp, vision or general object handling claim.</p>')
    (OUT/'latest.json').write_text(json.dumps({'directory':str(folder),'status':report['status']},indent=2)+'\n')


def train():
    folder=OUT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True)
    report=dict(status='qualifying_teacher',method='supervised local linear torque feedback distillation',
                robot='original G1 with rubber-hand mesh collisions enabled for this scene',object_mass_kg=.15,
                trials=[],limitations='Fixed two-hand box lift, no fingers, no vision, no RL or DAgger. Student observes robot state and time; it does not adapt to arbitrary object positions. Initial box XY +/-1 mm and joints sigma 0.0005 rad.')
    def evaluate(group,seeds,student=None):
        results=[]
        for seed in seeds:
            result,sim=rollout(seed,student)
            report['trials'].append(dict(group=group,**result));results.append(result)
            np.savetxt(folder/f'{group}-{seed}-trace.csv',sim.trace,delimiter=',',header=TRACE_HEADER,comments='')
            print(f'{group} seed {seed}: {result["passed"]}, lift {result["lift_m"]:.3f} m, hold {result["hold_seconds"]:.2f} s',flush=True)
        save_report(folder,report)
        return all(r['passed'] for r in results)
    try:
        if not evaluate('teacher',[101,102,103,104,105]):
            report['status']='teacher_failed_qualification';return folder
        data,demo=collect(folder);report['demonstration']=demo;report['labelled_states']=len(data['times'])*140
        for damping in [1.,.75,.5]:
            candidate=dict(data);gain=np.array(data['gains'],copy=True);gain[:,:,35:]*=damping;candidate['gains']=gain
            path=folder/f'policy-damping-{damping:.2f}.npz';np.savez_compressed(path,**candidate)
            student=Student(path)
            if not evaluate(f'selection-{damping:g}',[9001,9002,9003],student):continue
            report['checkpoint']=str(path);report['velocity_feedback_scale']=damping
            if evaluate('student-final',[99001,99002,99003,99004,99005],student):
                report['status']='student_passed_known_box_lift'
                QUALIFIED.write_text(json.dumps({'checkpoint':str(path),'report':str(folder/'results.json')},indent=2)+'\n')
            else:report['status']='student_failed_final_evaluation'
            return folder
        report['status']='student_failed_selection'
    except BaseException as exc:
        report['status']='error';report['error']=str(exc);raise
    finally:
        save_report(folder,report);print('Report:',folder/'index.html',flush=True)
    return folder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['evaluate','view-teacher','train','view-student','results'])
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--checkpoint',type=Path)
    a=p.parse_args()
    if a.command=='train':train();return
    if a.command=='results':
        if not (OUT/'latest.json').exists():p.error('No training report yet. Run just grasp-train.')
        print(Path(json.loads((OUT/'latest.json').read_text())['directory'])/'index.html');return
    student=None
    if a.command=='view-student':
        if a.checkpoint is None:
            if not QUALIFIED.exists():p.error('No qualified grasp student. Run just grasp-train and inspect its results.')
            a.checkpoint=Path(json.loads(QUALIFIED.read_text())['checkpoint'])
        student=Student(a.checkpoint)
    result,sim=rollout(a.seed,student,view=a.command.startswith('view-'))
    OUT.mkdir(parents=True,exist_ok=True)
    name='teacher' if student is None else 'student'
    (OUT/f'{name}-latest.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savetxt(OUT/f'{name}-trace.csv',sim.trace,delimiter=',',header=TRACE_HEADER,comments='')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
