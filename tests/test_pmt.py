"""Verify geometry, joint mapping and supervised checkpoint behavior."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import mujoco

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pmt_assets import DATA, MOTIONS
from pmt_simulation import height_grid, load_clip, Simulation, Teacher, catalog, SCENE
from pmt_training import FEATURES, features, fit, Student, qualify


class TerrainTests(unittest.TestCase):
    def test_raster_preserves_step_and_rejects_holes(self):
        # Floor plus a raised plateau; taking a convex hull would fill the stair.
        floor=np.array([[[0,0,0],[2,0,0],[2,1,0]],[[0,0,0],[2,1,0],[0,1,0]]])
        top=np.array([[[1,0,.2],[2,0,.2],[2,1,.2]],[[1,0,.2],[2,1,.2],[1,1,.2]]])
        x,y,h=height_grid(np.concatenate([floor,top]),np.array([0,0]),np.array([2,1]),.1)
        self.assertAlmostEqual(h[5,5],0)
        self.assertAlmostEqual(h[5,15],.2)
        with self.assertRaises(ValueError):
            height_grid(top,np.array([0,0]),np.array([2,1]),.1)

    def test_failed_teachers_are_excluded(self):
        failed={'qualified_ascent':False,'seconds':1.2}
        with patch('pmt_training.catalog',return_value=[{'clip':'failed'}]), \
             patch('pmt_training.Simulation'), \
             patch('pmt_training.rollout',return_value=(failed,[],[])):
            selected,reports=qualify(object())
        self.assertEqual(selected,[])
        self.assertEqual(reports,[failed])

    def test_fit_learns_and_saves_candidate(self):
        torch.set_num_threads(1);torch.manual_seed(2)
        rng=np.random.default_rng(2)
        x=rng.normal(size=(100,FEATURES)).astype(np.float32)
        # A constant residual exercises fitting without mirroring model internals.
        y=np.full((100,29),.05,dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            np.savez(path/'data.npz',observations=x,actions=y,validation=np.arange(100)>=80)
            history=fit(path/'data.npz',path,20)
            self.assertLess(min(r['validation_mse'] for r in history),history[0]['validation_mse'])
            student=Student(path/'policy.pt')
            self.assertFalse(student.net.training)
            checkpoint=torch.load(path/'policy.pt',weights_only=True)
            checkpoint.pop('robot_id')
            torch.save(checkpoint,path/'legacy.pt')
            with self.assertRaisesRegex(ValueError,'different robot'):
                Student(path/'legacy.pt')


@unittest.skipUnless((DATA/MOTIONS).exists(),'Run just pmt-setup first')
class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim=Simulation(catalog()[20]);cls.teacher=Teacher()

    def test_original_robot_properties_are_preserved(self):
        original=mujoco.MjModel.from_xml_path(str(SCENE))
        actual=self.sim.model
        for name in ['body_mass','body_inertia','body_pos','body_quat',
                     'dof_armature','dof_damping','dof_frictionloss',
                     'actuator_ctrlrange','jnt_range']:
            np.testing.assert_array_equal(getattr(actual,name),getattr(original,name))
        # Terrain is appended; all original robot collision and visual geoms remain.
        robot=np.flatnonzero(original.geom_bodyid!=0)
        for name in ['geom_type','geom_pos','geom_size','geom_contype','geom_conaffinity','geom_rgba']:
            np.testing.assert_array_equal(getattr(actual,name)[actual.geom_bodyid!=0],getattr(original,name)[robot])

    def test_schema_and_physical_action(self):
        sim=self.sim
        obs=sim.reset(42)
        self.assertEqual(features(obs).shape,(FEATURES,))
        action=self.teacher(obs)
        self.assertEqual(action.shape,(29,))
        sim.step(action)
        first=sim.data.qpos.copy()
        sim.reset(42);sim.step(np.zeros(29))
        self.assertAlmostEqual(sim.data.time,.02)
        self.assertGreater(np.linalg.norm(first-sim.data.qpos),1e-7)
        self.assertEqual(sim.model.neq,0)  # floating root has no weld anchor
        with self.assertRaises(ValueError):sim.step(np.full(29,np.nan))

    def test_named_permutation_and_foot_reference(self):
        sim=self.sim;sim.reset(42)
        # Reset's 0.001 rad noise allows a small FK discrepancy.
        for name,index in [('left_ankle_roll_link',18),('right_ankle_roll_link',19)]:
            error=np.linalg.norm(sim.data.body(name).xpos-sim.clip['body_pos_w'][sim.frame,index])
            self.assertLess(error,.004)
        q=np.arange(29)
        np.testing.assert_array_equal(q[sim.perm][sim.inverse],q)


if __name__=='__main__':unittest.main()
