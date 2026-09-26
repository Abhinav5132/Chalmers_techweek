"""Physical invariants and qualification for the original G1 stair curriculum."""
import sys
from pathlib import Path
import unittest
import tempfile
import argparse
from unittest.mock import patch
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from stair_curriculum import Teacher, SCENE, rollout
from stair_learning import inputs, FEATURES, train


class StairTests(unittest.TestCase):
    def test_original_robot_and_pin_mapping(self):
        s=Teacher();m=mujoco.MjModel.from_xml_path(str(SCENE))
        for field in ['body_mass','body_inertia','dof_armature','dof_frictionloss',
                      'actuator_ctrlrange','jnt_range']:
            np.testing.assert_array_equal(getattr(s.m,field),getattr(m,field))
        for field in ['geom_size','geom_type','geom_pos','geom_friction','geom_contype']:
            np.testing.assert_array_equal(getattr(s.m,field)[s.m.geom_bodyid!=0],getattr(m,field)[m.geom_bodyid!=0])
        self.assertEqual(s.m.neq,0)
        for p,b in zip(s.planner.feet,s.ids):np.testing.assert_allclose(p,s.d.xpos[b],atol=1e-8)
        before=s.d.qpos.copy();x,base=inputs(s)
        self.assertEqual(x.shape,(FEATURES,));self.assertEqual(base.shape,(29,))
        np.testing.assert_array_equal(s.d.qpos,before)  # IK cannot move physical state.

    def test_failed_teacher_blocks_collection_and_training(self):
        failed={"passed":False,"failure":"fall","seconds":1,"height_m":.05,"seed":1,"top_hold_seconds":0}
        with tempfile.TemporaryDirectory() as directory, patch("stair_learning.OUT",Path(directory)), \
             patch("stair_learning.run",return_value=(failed,[],[],None)) as runner, \
             patch("stair_learning.fit") as fitter:
            args=argparse.Namespace(heights=[.05,.075],episodes=4,epochs=1,rounds=1)
            train(args)
            self.assertEqual(runner.call_count,5)
            fitter.assert_not_called()
            self.assertFalse(list(Path(directory).rglob("*.pt")))

    def test_physics_action_and_invalid_input(self):
        a=Teacher(stand_only=True);b=Teacher(stand_only=True)
        for _ in range(30):a.step();b.step(np.zeros(29))
        self.assertGreater(np.linalg.norm(a.d.qpos-b.d.qpos),1e-3)
        self.assertLessEqual(a.max_torque_ratio,1.001)
        self.assertIsNone(a.fail)
        with self.assertRaises(ValueError):a.step(np.full(29,np.nan))

    def test_climb_requires_loaded_final_hold(self):
        _,short=rollout(seconds=1)
        self.assertFalse(short['passed'])
        sim,full=rollout(seed=41)
        self.assertTrue(full['passed'],full)
        self.assertGreaterEqual(sim.top_hold_seconds,2.9)
        self.assertTrue(np.all(sim.contacts()[1]>20))
        self.assertLess(full['max_torque_ratio'],1)
        self.assertGreater(sim.d.qpos[0],.25)

if __name__=='__main__':unittest.main()
