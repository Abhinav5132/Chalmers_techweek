"""Original-G1 single-step teacher: Pinocchio planning, constrained inverse dynamics.

No base anchoring or model-parameter substitutions. Runtime pose writes occur only
at reset; IK uses its own Pinocchio state. Training is gated by physical evaluation.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import mujoco
import pinocchio as pin
from qpsolvers import solve_qp

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'unitree_mujoco/unitree_robots/g1/scene_29dof.xml'
ROBOT = SCENE.with_name('g1_29dof.xml')
OUT = ROOT / '.robot-runtime/imitation/stair-curriculum'
DT = .01
FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t*t*t*(10+t*(-15+6*t))


def model(height):
    if not np.isfinite(height) or not .01<=height<=.35:raise ValueError("Step height must be between 0.01 and 0.35 m")
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.option.timestep = .002
    spec.worldbody.add_geom(name='curriculum_step', type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[.7, 0, height/2], size=[.5, .65, height/2],
        rgba=[.3,.45,.55,1], friction=[1,.005,.0001], contype=1, conaffinity=1)
    return spec.compile()


class Planner:
    def __init__(self, m, initial):
        self.m = pin.buildModelFromMJCF(str(ROBOT))
        self.d = self.m.createData()
        self.frames = [self.m.getFrameId(x) for x in FEET]
        assert self.m.nq == m.nq and self.m.nv == m.nv
        self.q = self.from_mj(initial)
        self.nominal = self.q.copy()
        self.update()
        self.feet = [self.d.oMf[i].translation.copy() for i in self.frames]
        self.com = pin.centerOfMass(self.m,self.d,self.q).copy()

    @staticmethod
    def from_mj(q):
        return np.r_[q[:3],q[4:7],q[3],q[7:]]

    def update(self):
        pin.computeJointJacobians(self.m,self.d,self.q)
        pin.updateFramePlacements(self.m,self.d)

    def solve(self, feet, com):
        for _ in range(12):
            self.update()
            rows=[]; errs=[]
            for fid,target in zip(self.frames,feet):
                pose=self.d.oMf[fid]
                J=pin.getFrameJacobian(self.m,self.d,fid,pin.LOCAL_WORLD_ALIGNED)
                rows.append(J)
                errs.append(np.r_[target-pose.translation, pin.log3(pose.rotation.T)])
            Jc=pin.jacobianCenterOfMass(self.m,self.d,self.q)
            rows.append(Jc);errs.append(com-self.d.com[0])
            root=np.zeros((3,self.m.nv));root[:,3:6]=np.eye(3)
            rows.append(root);errs.append(pin.log3(pin.Quaternion(self.q[3:7]).matrix().T))
            posture=np.zeros((29,self.m.nv));posture[:,6:]=np.eye(29)*.025
            rows.append(posture);errs.append((self.nominal[7:]-self.q[7:])*.025)
            J=np.vstack(rows);e=np.concatenate(errs)
            dq=np.linalg.solve(J.T@J+np.eye(self.m.nv)*1e-6,J.T@e)
            self.q=pin.integrate(self.m,self.q,np.clip(dq,-.05,.05))
            self.q[7:]=np.clip(self.q[7:],self.m.lowerPositionLimit[7:]+.01,self.m.upperPositionLimit[7:]-.01)
            if np.max(np.abs(e[:12]))<1e-5: break
        return self.q[7:].copy()


class Teacher:
    """QP inverse dynamics with actual robot inertia, unilateral friction forces,
    original torque limits, and feedback tasks for COM, feet, torso and posture.
    """
    def __init__(self, height=.05, seed=0, stand_only=False):
        self.height=height;self.stand_only=stand_only
        self.m=model(height);self.d=mujoco.MjData(self.m)
        self.ids=[self.m.body(x).id for x in FEET]
        self.torso=self.m.body('torso_link').id
        self.d.qpos[:]=0;self.d.qpos[3]=1
        for offset in [7,13]: self.d.qpos[offset:offset+6]=[-.15,0,0,.3,-.15,0]
        self.d.qpos[7+16]=.15;self.d.qpos[7+23]=-.15
        self.d.qpos[7+18]=.4;self.d.qpos[7+25]=.4
        self.d.qpos[2]=.8;mujoco.mj_forward(self.m,self.d)
        self.d.qpos[2]-=min(self.d.xpos[i,2] for i in self.ids)-.035
        self.d.qpos[7:]+=np.random.default_rng(seed).normal(0,.0005,29)
        mujoco.mj_forward(self.m,self.d)
        self.planner=Planner(self.m,self.d.qpos.copy())
        self.feet0=np.array([self.d.xpos[i].copy() for i in self.ids])
        self.com0=self.d.subtree_com[1].copy()
        self.last_j={};self.last_qref=self.d.qpos[7:].copy();self.fail=None
        self.max_torque_ratio=0.;self.min_up=1.;self.trace=[]
        self.top_hold_seconds=0.; self.contact_trace=[]; self.support_loss_seconds=0.

    def targets(self,t):
        if self.height>.05001 and not self.stand_only:return self.high_targets(t)
        feet=self.feet0.copy();com=self.com0.copy();stance=[True,True]
        # Slow weight transfers, with both feet supported before each swing.
        if self.stand_only: return feet,com,stance,'stand'
        land=self.feet0.copy();land[:,0]+=.36;land[:,2]+=self.height
        if t<2: phase='settle'
        elif t<5:
            com[1]=(1-smooth((t-2)/3))*com[1]+smooth((t-2)/3)*feet[1,1];phase='shift_right'
        elif t<9:
            s=smooth((t-5)/4);feet[0]=(1-s)*feet[0]+s*land[0];feet[0,2]+=.10*np.sin(np.pi*s)
            com[1]=self.feet0[1,1];stance[0]=False;phase='left_swing'
        elif t<13:
            feet[0]=land[0];s=smooth((t-9)/4)
            com[:2]=(1-s)*np.array([self.com0[0],self.feet0[1,1]])+s*(land[0,:2]+[.02,0])
            com[2]+=(self.height)*s;phase='shift_left'
        elif t<17:
            feet[0]=land[0];s=smooth((t-13)/4);feet[1]=(1-s)*feet[1]+s*land[1];feet[1,2]+=.10*np.sin(np.pi*s)
            com[:2]=land[0,:2]+[.02,0];com[2]+=self.height;stance[1]=False;phase='right_swing'
        else:
            feet=land;s=smooth((t-17)/3)
            com[:2]=(1-s)*(land[0,:2]+[.02,0])+s*(np.mean(land[:,:2],axis=0)+[self.com0[0]-np.mean(self.feet0[:,0]),0])
            com[2]+=self.height;phase='top_hold'
        return feet,com,stance,phase

    def high_targets(self,t):
        feet=self.feet0.copy();com=self.com0.copy();stance=[True,True]
        land=self.feet0.copy();land[:,0]+=.30;land[:,2]+=self.height
        rise=self.height*.5
        def swing(k,u):
            # Lift before translating across the vertical riser, then lower.
            feet[k,0]+= .30*smooth((u-.3)/.5)
            feet[k,2]+= (self.height+.045)*smooth(u/.4)-.045*smooth((u-.8)/.2)
        if t<2:phase='settle'
        elif t<5:
            com[1]=(1-smooth((t-2)/3))*com[1]+smooth((t-2)/3)*feet[1,1];phase='shift_right'
        elif t<9:
            swing(0,(t-5)/4);com[1]=self.feet0[1,1];stance[0]=False;phase='left_swing'
        elif t<13:
            feet[0]=land[0];s=smooth((t-9)/4)
            com[:2]=(1-s)*np.array([self.com0[0],self.feet0[1,1]])+s*(land[0,:2]+[.02,0])
            com[2]+=rise*s;phase='shift_left'
        elif t<17:
            feet[0]=land[0];u=(t-13)/4;s=smooth(u)
            swing(1,u);com[:2]=land[0,:2]+[.02,0]
            com[2]+=rise+(self.height-rise)*s;stance[1]=False;phase='right_swing'
        else:
            feet=land;s=smooth((t-17)/3)
            com[:2]=(1-s)*(land[0,:2]+[.02,0])+s*(np.mean(land[:,:2],axis=0)+[self.com0[0]-np.mean(self.feet0[:,0]),0])
            com[2]+=self.height;phase='top_hold'
        return feet,com,stance,phase

    def jac(self, point, body, key):
        jp=np.zeros((3,self.m.nv));jr=jp.copy()
        mujoco.mj_jac(self.m,self.d,jp,jr,point,body)
        J=np.vstack([jp,jr]);old=self.last_j.get(key,J);self.last_j[key]=J.copy()
        return J,(J-old)@self.d.qvel/DT

    def reference(self, feet=None, com=None):
        if getattr(self,"reference_time",None)!=self.d.time:
            if feet is None:feet,com,_,_=self.targets(self.d.time)
            self.last_qref=self.planner.solve(feet,com)
            self.reference_time=self.d.time
        q=self.last_qref.copy()
        if self.height>.05001:
            envelope=min(1.,(self.height-.05)/.3)*smooth((self.d.time-9)/4)*(1-smooth((self.d.time-13)/4))
            q[13]=-.45*envelope
            q[14]=.4*envelope
        return q

    def extra_tasks(self, task):
        """Optional manipulation tasks, expressed in the robot's velocity space."""

    def action(self):
        m,d=self.m,self.d;n=m.nv
        feet,com,stance,phase=self.targets(d.time)
        qref=self.reference(feet,com)
        active=[k for k in range(2) if stance[k]]
        nf=12*len(active);N=n+nf
        M=np.zeros((n,n));mujoco.mj_fullM(m,d,M)
        # Each supporting foot has four actual collision-sphere contact points.
        Jforces=[]
        for k in active:
            bid=self.ids[k]
            for gid in range(m.ngeom):
                if m.geom_bodyid[gid]==bid and m.geom_contype[gid] and m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_SPHERE:
                    pt=d.geom_xpos[gid].copy();pt[2]-=m.geom_size[gid,0]
                    jp=np.zeros((3,n));jr=jp.copy();mujoco.mj_jac(m,d,jp,jr,pt,bid);Jforces.append(jp)
        assert len(Jforces)*3==nf
        Jf=np.vstack(Jforces)
        T=np.hstack([M[6:],-Jf[:,6:].T]);bias=d.qfrc_bias[6:]-d.qfrc_passive[6:]
        H=np.eye(N)*1e-7;g=np.zeros(N)
        def task(J,a,w):
            nonlocal H,g
            B=np.zeros((len(a),N));B[:,:n]=J
            H+=w*(B.T@B);g-=w*B.T@a
        Jcom=np.zeros((3,n));mujoco.mj_jacSubtreeCom(m,d,Jcom,1)
        old=self.last_j.get('com',Jcom);self.last_j['com']=Jcom.copy()
        ac=60*(com-d.subtree_com[1])-16*(Jcom@d.qvel)-(Jcom-old)@d.qvel/DT
        task(Jcom,ac,15)
        for k,bid in enumerate(self.ids):
            J,jdv=self.jac(d.xpos[bid],bid,('foot',k))
            R=d.xmat[bid].reshape(3,3)
            err=np.r_[feet[k]-d.xpos[bid],pin.log3(R.T)]
            acc=180*err-28*(J@d.qvel)-jdv
            task(J,acc,100 if stance[k] else 20)
        J,jdv=self.jac(d.xpos[1],1,'root')
        task(J[3:],100*pin.log3(d.xmat[1].reshape(3,3).T)-20*(J[3:]@d.qvel)-jdv[3:],10)
        Jp=np.zeros((29,n));Jp[:,6:]=np.eye(29)
        task(Jp,40*(qref-d.qpos[7:])-10*d.qvel[6:],.1)
        if self.height>.05001:
            task(Jp[13:15],60*(qref[13:15]-d.qpos[20:22])-16*d.qvel[19:21],5)
        self.extra_tasks(task)
        A=np.hstack([M[:6],-Jf[:,:6].T]);b=-d.qfrc_bias[:6]+d.qfrc_passive[:6]
        # Original actuator torque bounds, non-negative normal force and friction pyramid.
        G=[T,-T];h=[m.actuator_ctrlrange[:,1]-bias,-m.actuator_ctrlrange[:,0]+bias]
        F=[]
        for i in range(nf//3):
            for coefficients in ([1,0,-.5],[-1,0,-.5],[0,1,-.5],[0,-1,-.5],[0,0,-1]):
                row=np.zeros(N);row[n+3*i:n+3*i+3]=coefficients;F.append(row)
        G.append(np.array(F));h.append(np.zeros(len(F)))
        x=solve_qp(H,g,np.vstack(G),np.concatenate(h),A,b,solver='proxqp')
        if x is None or not np.isfinite(x).all(): raise RuntimeError('Balance QP failed')
        tau=T@x+bias
        if np.max(np.maximum(tau-m.actuator_ctrlrange[:,1],m.actuator_ctrlrange[:,0]-tau))>.1:
            raise RuntimeError('Balance QP exceeded motor limits')
        return np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]),phase

    def contacts(self):
        loads=np.zeros(2);top=np.zeros(2)
        for ci in range(self.d.ncon):
            c=self.d.contact[ci]
            for k,bid in enumerate(self.ids):
                bodies=[self.m.geom_bodyid[c.geom1],self.m.geom_bodyid[c.geom2]]
                if bid not in bodies or 0 not in bodies:continue
                f=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,ci,f)
                if abs(c.frame[2])>.8:
                    loads[k]+=max(0,f[0])
                    if self.m.geom('curriculum_step').id in (c.geom1,c.geom2):top[k]+=max(0,f[0])
        return loads,top

    def step(self, action=None):
        if action is None: action,phase=self.action()
        else: phase=self.targets(self.d.time)[3]
        if np.shape(action)!=(29,) or not np.isfinite(action).all():raise ValueError("Expected 29 finite motor torques")
        self.d.ctrl[:]=np.clip(action,self.m.actuator_ctrlrange[:,0],self.m.actuator_ctrlrange[:,1])
        self.max_torque_ratio=max(self.max_torque_ratio,float(np.max(np.abs(self.d.ctrl)/self.m.actuator_ctrlrange[:,1])))
        for _ in range(round(DT/self.m.opt.timestep)):mujoco.mj_step(self.m,self.d)
        up=float(self.d.xmat[1,8]);self.min_up=min(self.min_up,up)
        if up<.65 or self.d.qpos[2]<.45:self.fail='fall'
        if not np.isfinite(self.d.qpos).all():self.fail='nonfinite_state'
        loads,top=self.contacts()
        self.contact_trace.append([self.d.time,*loads,*top])
        stance=self.targets(self.d.time)[2]
        # Stop if a required single-support foot loses physical ground contact.
        if sum(stance)==1 and loads[np.flatnonzero(stance)[0]]<20:self.support_loss_seconds+=DT
        else:self.support_loss_seconds=0.
        if self.support_loss_seconds>.15:self.fail="support_contact_lost"
        if self.d.time>=20 and np.all(top>20) and up>.95:self.top_hold_seconds+=DT
        elif self.d.time>=20:self.top_hold_seconds=0.
        self.trace.append([self.d.time,*self.d.qpos[:3],up,*self.d.subtree_com[1],*self.d.xpos[self.ids[0]],*self.d.xpos[self.ids[1]]])
        return phase


def rollout(height=.05,seed=0,seconds=23,stand=False,view=False):
    sim=Teacher(height,seed,stand)
    viewer=None
    if view:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(sim.m,sim.d)
        viewer.cam.distance=2.3+height;viewer.cam.lookat[:]=[.25,0,.65+height/2]
    try:
        while sim.d.time<seconds and not sim.fail:
            start=time.monotonic()
            try:sim.step()
            except RuntimeError as exc:sim.fail=str(exc);break
            if viewer:
                if not viewer.is_running():sim.fail='viewer_closed';break
                viewer.sync();time.sleep(max(0,DT-(time.monotonic()-start)))
    finally:
        if viewer:viewer.close()
    top=all(sim.d.xpos[i,0]>.25 and abs(sim.d.xpos[i,2]-(sim.feet0[k,2]+height))<.025 for k,i in enumerate(sim.ids))
    result={'height_m':height,'seed':seed,'seconds':float(sim.d.time),'passed':not sim.fail and (stand or (seconds>=23 and top and sim.top_hold_seconds>=2.9)),
            'failure':sim.fail,'both_feet_on_top':top,'top_hold_seconds':sim.top_hold_seconds,'minimum_up_dot':sim.min_up,'max_torque_ratio':sim.max_torque_ratio,
            'robot':'original_g1_29dof','base_anchored':False}
    return sim,result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['evaluate','view','stand'])
    p.add_argument('--height',type=float,default=.05);p.add_argument('--seconds',type=float,default=23);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    if not .01<=a.height<=.35:p.error('height must be between 0.01 and 0.35 m')
    sim,result=rollout(a.height,a.seed,a.seconds,a.command=='stand',a.command=='view')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'latest.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savetxt(OUT/'contacts.csv',sim.contact_trace,delimiter=',',header='time,left_load_N,right_load_N,left_step_load_N,right_step_load_N',comments='')
    np.savetxt(OUT/'trace.csv',sim.trace,delimiter=',',header='time,root_x,root_y,root_z,up,com_x,com_y,com_z,left_x,left_y,left_z,right_x,right_y,right_z',comments='')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
