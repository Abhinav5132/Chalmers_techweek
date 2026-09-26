"""Experimental CPU MuJoCo transfer of the pinned PMT terrain teacher.

The upstream environment uses Isaac Lab. This adapter retains its observation
layout and residual joint targets, but simulator/contact transfer needs testing.
Terrain is a local 1 cm height-field sampling of the source triangle mesh.
"""
from collections import deque
import math

import mujoco
import numpy as np
import torch

from pmt_assets import DATA, MOTIONS, ROOT, TEACHER, TERRAIN, verify
from pmt_vendor.order import BFS_BODIES, BFS_JOINTS
from pmt_vendor.transformer import TransformerActorCritic

TRACKED = ['pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link',
           'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'torso_link',
           'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link',
           'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link']
ROBOT_ID = 'repository_g1_29dof_original'
SCENE = ROOT / 'unitree_mujoco/unitree_robots/g1/scene_29dof.xml'
OUT = ROOT / '.robot-runtime/imitation/pmt-original-g1'


def rotation(q):
    out = np.empty(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, dtype=float))
    return out.reshape(3, 3)


def load_clip(path):
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k].copy() for k in f.files}
    frames = len(d['joint_pos'])
    expected = {'joint_pos': (frames, 29), 'joint_vel': (frames, 29),
                'body_pos_w': (frames, 30, 3), 'body_quat_w': (frames, 30, 4),
                'body_lin_vel_w': (frames, 30, 3), 'body_ang_vel_w': (frames, 30, 3)}
    if frames < 300 or not np.isclose(float(d['fps'].item()), 50):
        raise ValueError('Expected at least 300 frames at 50 Hz')
    for k, shape in expected.items():
        if d[k].shape != shape or not np.isfinite(d[k]).all():
            raise ValueError(f'Invalid PMT array: {k}')
    if not np.allclose(np.linalg.norm(d['body_quat_w'], axis=2), 1, atol=.002):
        raise ValueError('Invalid reference quaternions')
    return d


def catalog():
    """Rank by reference foot-height gain; candidates, not verified stair skills."""
    rows = []
    for path in sorted((DATA / MOTIONS).glob('*.npz')):
        d = load_clip(path)
        z = d['body_pos_w'][:, [18, 19], 2].min(axis=1)
        gain = z[250:] - z[:-250]
        start = int(gain.argmax())
        rows.append({'clip': path.name, 'start': start, 'frames': 300,
                     'reference_foot_gain_m': float(gain[start])})
    if not rows:
        raise ValueError('No PMT motions found. Run just pmt-setup.')
    return sorted(rows, key=lambda row: row['reference_foot_gain_m'], reverse=True)


def height_grid(triangles, low, high, spacing=.01):
    """Rasterize triangle tops without replacing a nonconvex mesh by its hull."""
    x = np.linspace(low[0], high[0], int(np.ceil((high[0] - low[0]) / spacing)) + 1)
    y = np.linspace(low[1], high[1], int(np.ceil((high[1] - low[1]) / spacing)) + 1)
    height = np.full((len(y), len(x)), -np.inf)
    for v in triangles:
        if np.any(v[:, :2].max(0) < low) or np.any(v[:, :2].min(0) > high):
            continue
        a, b, c = v
        den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(den) < 1e-10:  # vertical faces have no area in a height map
            continue
        ix = np.flatnonzero((x >= v[:, 0].min()-1e-6) & (x <= v[:, 0].max()+1e-6))
        iy = np.flatnonzero((y >= v[:, 1].min()-1e-6) & (y <= v[:, 1].max()+1e-6))
        if not len(ix) or not len(iy):
            continue
        xx, yy = np.meshgrid(x[ix], y[iy])
        u = ((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        w = ((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        inside = (u >= -1e-5) & (w >= -1e-5) & (u+w <= 1+1e-5)
        values = np.where(inside, u*a[2]+w*b[2]+(1-u-w)*c[2], -np.inf)
        height[np.ix_(iy, ix)] = np.maximum(height[np.ix_(iy, ix)], values)
    if not np.isfinite(height).all():
        raise ValueError('Terrain crop contains unsupported holes; cannot approximate them as floor')
    return x, y, height


def terrain_model(clip):
    payload = verify(TERRAIN).read_bytes()
    n = int.from_bytes(payload[80:84], 'little')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
    if len(payload) != 84 + 50*n:
        raise ValueError('Invalid binary STL')
    triangles = np.frombuffer(payload, dtype=dtype, offset=84)['vertices']  # ty: ignore[invalid-argument-type]
    xy = clip['body_pos_w'][:, 0, :2]
    low, high = xy.min(0)-1.5, xy.max(0)+1.5
    x, y, height = height_grid(triangles, low, high)
    # Use the project's original G1, including its meshes, inertias, joints,
    # collision feet and actuator limits. Only replace the environment floor.
    spec = mujoco.MjSpec.from_file(str(SCENE))
    floor = spec.geom('floor')
    floor.contype = 0
    floor.conaffinity = 0
    floor.rgba = [0,0,0,0]
    zmin = float(height.min())
    zscale = max(.01, float(height.max()-zmin))
    spec.add_hfield(name='pmt_terrain', nrow=len(y), ncol=len(x),
                    userdata=((height-zmin)/zscale).ravel().tolist(),
                    size=[(x[-1]-x[0])/2, (y[-1]-y[0])/2, zscale, .1])
    spec.worldbody.add_geom(name='pmt_ground', type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname='pmt_terrain', pos=[(x[-1]+x[0])/2, (y[-1]+y[0])/2, zmin],
        rgba=[.45, .6, .7, 1], friction=[1, .005, .0001], contype=1, conaffinity=1)
    model = spec.compile()
    model.hfield_data[:] = ((height-zmin)/zscale).ravel()
    model.opt.timestep = .002
    return model


class Teacher:
    def __init__(self):
        torch.set_num_threads(1)
        ckpt = torch.load(verify(TEACHER), map_location='cpu', weights_only=True)
        schema = ckpt['policy_metadata']['obs_schema']
        self.shapes = {k: tuple(v['shape']) for k, v in schema['available_obs'].items()}
        zeros = {k: torch.zeros(1, *shape) for k, shape in self.shapes.items()}
        self.net = TransformerActorCritic(zeros, schema['obs_groups'], 29,
            actor_obs_normalization=True, critic_obs_normalization=True,
            history_obs_normalization=True, command_obs_normalization=True,
            actor_hidden_dims=[512,256,128], critic_hidden_dims=[512,256,128],
            use_vel_estimator=True, use_anchor_estimator=True,
            vel_gt_normalization=True, anchor_gt_normalization=True,
            vel_estimator_hidden_dims=(256,128), anchor_estimator_hidden_dims=(256,128))
        self.net.load_state_dict(ckpt['model_state_dict'], strict=True)
        self.net.eval()

    def __call__(self, obs):
        for key, value in obs.items():
            if value.shape != self.shapes[key] or not np.isfinite(value).all():
                raise ValueError(f'Invalid teacher observation: {key}, {value.shape}')
        tensors = {k: torch.as_tensor(v, dtype=torch.float32)[None] for k, v in obs.items()}
        with torch.inference_mode():
            return self.net.act_inference(tensors)[0].numpy().copy()


class Simulation:
    def __init__(self, row):
        self.row = row
        path = verify(MOTIONS + row['clip'])
        self.clip = load_clip(path)
        self.model = terrain_model(self.clip)
        self.data = mujoco.MjData(self.model)
        self.perm = np.array([BFS_JOINTS.index(self.model.joint(i).name) for i in range(1,30)])
        self.inverse = np.argsort(self.perm)
        self.ids = [self.model.body(name).id for name in TRACKED]
        self.anchor = self.model.body('torso_link').id
        self.kp, self.kd, self.default = [], [], []
        # PMT G1_CYLINDER_CFG: natural frequency 10 Hz, damping ratio 2.
        for i in range(1, 30):
            name = self.model.joint(i).name
            armature = .003609725
            if 'hip_roll' in name or 'knee' in name:
                armature = .025101925
            elif any(s in name for s in ('hip_pitch','hip_yaw','waist_yaw')):
                armature = .010177520
            elif any(s in name for s in ('ankle','waist_roll','waist_pitch')):
                armature = 2*.003609725
            elif 'wrist_pitch' in name or 'wrist_yaw' in name:
                armature = .00425
            self.kp.append(armature * (20*math.pi)**2)
            self.kd.append(4*armature*20*math.pi)
            q = 0.
            for suffix, value in [('hip_pitch_joint',-.312),('knee_joint',.669),
                                  ('ankle_pitch_joint',-.363),('elbow_joint',.6),
                                  ('shoulder_pitch_joint',.2),('shoulder_roll_joint',.2)]:
                if name.endswith(suffix):
                    q = -value if name == 'right_shoulder_roll_joint' else value
            self.default.append(q)
        self.kp, self.kd = np.array(self.kp), np.array(self.kd)
        self.default = np.array(self.default)[self.inverse]
        self.last = np.zeros(29)
        self.history = {}
        self.frame = row['start']
        self.elapsed = 0

    def reset(self, seed=0):
        self.frame = self.row['start']
        self.elapsed = 0
        self.history = {}
        self.last[:] = 0
        mujoco.mj_resetData(self.model, self.data)
        c, f, d = self.clip, self.frame, self.data
        d.qpos[:3] = c['body_pos_w'][f, 0]
        d.qpos[3:7] = c['body_quat_w'][f, 0]
        d.qpos[7:] = c['joint_pos'][f, self.perm] + np.random.default_rng(seed).normal(0,.001,29)
        # Isaac Lab root velocity is COM velocity; MuJoCo free-joint linear
        # velocity is the body origin. Match the source reset convention.
        base_rotation = rotation(d.qpos[3:7])
        d.qvel[:3] = c['body_lin_vel_w'][f, 0] - np.cross(
            c['body_ang_vel_w'][f, 0], base_rotation @ self.model.body_ipos[1])
        d.qvel[3:6] = rotation(d.qpos[3:7]).T @ c['body_ang_vel_w'][f, 0]
        d.qvel[6:] = c['joint_vel'][f, self.perm]
        mujoco.mj_forward(self.model, d)
        return self.observation()

    def observation(self):
        c, f, d = self.clip, self.frame, self.data
        R = d.xmat[self.anchor].reshape(3,3)
        base = rotation(d.qpos[3:7])
        bp = ((d.xpos[self.ids]-d.xpos[self.anchor]) @ R).ravel()
        bo = np.array([(R.T @ d.xmat[i].reshape(3,3))[:,:2] for i in self.ids]).ravel()
        q, dq = d.qpos[7:][self.inverse]-self.default, d.qvel[6:][self.inverse]
        w = d.qvel[3:6].copy()
        v = base.T @ d.qvel[:3] + np.cross(w, self.model.body_ipos[1])
        ap = R.T @ (c['body_pos_w'][f,9]-d.xpos[self.anchor])
        ao = (R.T @ rotation(c['body_quat_w'][f,9]))[:,:2].ravel()
        terms = [np.r_[c['joint_pos'][f], c['joint_vel'][f]], ap, ao, bp, bo, v, w, q, dq, self.last.copy()]
        hist = []
        for i, term in enumerate(terms):
            if i not in self.history:
                self.history[i] = deque([term.copy() for _ in range(10)], maxlen=10)
            else:
                self.history[i].append(term.copy())
            hist.append(np.asarray(self.history[i]).ravel())
        future = np.clip(f+np.arange(10)*5, 0, len(c['joint_pos'])-1)
        sonic = np.concatenate([c['joint_pos'][future],c['joint_vel'][future]],axis=1).ravel()
        policy = np.concatenate([hist[0], sonic, *hist[1:]])
        proprio = np.r_[base.T @ [0,0,-1], w, q, dq, self.last]
        if 'proprio' not in self.history:
            self.history['proprio'] = deque([proprio.copy() for _ in range(10)], maxlen=10)
        else:
            self.history['proprio'].append(proprio.copy())
        indices = np.clip(f+np.arange(-10,11), 0, len(c['joint_pos'])-1)
        window, delta = [], []
        centerR = rotation(c['body_quat_w'][f,9])
        for j in indices:
            r = rotation(c['body_quat_w'][j,9])
            window.append(np.r_[r.T @ c['body_lin_vel_w'][j,9],
                r.T @ c['body_ang_vel_w'][j,9], r.T @ [0,0,-1], c['joint_pos'][j]])
            delta.append(centerR.T @ (c['body_pos_w'][j,9]-c['body_pos_w'][f,9]))
        return {k: np.asarray(v,dtype=np.float32) for k,v in {
            'policy':policy, 'proprio':proprio,
            'proprio_history':np.asarray(self.history['proprio']),
            'command_window':window, 'motion_anchor_delta_window':delta,
            'anchor_body_pose':np.r_[bp,bo]}.items()}

    def step(self, action):
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError('Expected finite PMT residual action with 29 joints')
        target = (self.clip['joint_pos'][self.frame]+action)[self.perm]
        for _ in range(10):
            torque = self.kp*(target-self.data.qpos[7:])-self.kd*self.data.qvel[6:]
            self.data.ctrl[:] = np.clip(torque,self.model.actuator_ctrlrange[:,0],self.model.actuator_ctrlrange[:,1])
            mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)
        self.last = action.copy()
        self.frame = min(self.frame+1,len(self.clip['joint_pos'])-1)
        self.elapsed += 1
        ref = self.clip['body_pos_w'][self.frame,9]
        error = float(np.linalg.norm(ref-self.data.xpos[self.anchor]))
        # Relative reference height, rather than flat-world absolute height.
        fallen = bool(self.data.xmat[1].reshape(3,3)[2,2]<.5 or
                      self.data.qpos[2] < self.clip['body_pos_w'][self.frame,0,2]-.3)
        return self.observation(), fallen, error
