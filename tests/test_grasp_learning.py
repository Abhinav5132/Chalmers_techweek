"""Grasping requires actual contact, free physics and independent student control."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import grasp_learning as grasp


class GraspPhysicsTests(unittest.TestCase):
    def test_model_preserves_robot_and_free_object(self):
        original=mujoco.MjModel.from_xml_path(str(grasp.SCENE))
        model=grasp.make_model()
        self.assertEqual((model.nq,model.nv,model.nu,model.neq),(43,41,29,0))
        for name in ['body_mass','body_inertia','body_pos','body_quat','dof_armature','dof_damping','dof_frictionloss','jnt_range','actuator_ctrlrange']:
            value=getattr(original,name)
            np.testing.assert_array_equal(getattr(model,name)[:len(value)],value)
        self.assertAlmostEqual(model.body('grasp_object').mass[0],.15)
        for name in ['left_rubber_hand','right_rubber_hand']:
            mesh=model.mesh(name).id
            ids=np.flatnonzero(model.geom_dataid==mesh)
            self.assertTrue(any(model.geom_contype[i] for i in ids))
        self.assertEqual(model.joint('object_free').type[0],mujoco.mjtJoint.mjJNT_FREE)

    def test_unheld_object_falls_without_attachment(self):
        sim=grasp.Simulation()
        sim.d.qpos[38]+=.2
        mujoco.mj_forward(sim.m,sim.d)
        start=float(sim.d.qpos[38])
        for _ in range(20):mujoco.mj_step(sim.m,sim.d)
        self.assertLess(sim.d.qpos[38],start-.005)
        self.assertEqual(sim.hold,0.)
        self.assertFalse(sim.result(0,'uncontrolled')['passed'])

    def test_failed_teacher_blocks_collection_and_promotion(self):
        failed=dict(seed=0,controller='teacher',passed=False,failure='object_dropped',seconds=1.,hold_seconds=0.,lift_m=-.1,minimum_up_dot=1.,max_torque_ratio=.2,base_anchored=False,object_attached=False)
        class Trace:
            trace=np.zeros((1,11))
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch.object(grasp,'OUT',out), patch.object(grasp,'QUALIFIED',out/'qualified.json'), patch.object(grasp,'rollout',return_value=(failed,Trace())), patch.object(grasp,'collect') as collect:
                folder=grasp.train()
                collect.assert_not_called()
                self.assertFalse((out/'qualified.json').exists())
                self.assertEqual(json.loads((folder/'results.json').read_text())['status'],'teacher_failed_qualification')


@unittest.skipUnless(grasp.QUALIFIED.exists(),'Run just grasp-train to create a qualified student')
class GraspStudentTests(unittest.TestCase):
    def test_student_lifts_without_teacher_and_without_pose_writes(self):
        path=Path(json.loads(grasp.QUALIFIED.read_text())['checkpoint'])
        student=grasp.Student(path)
        sim=grasp.Simulation(99010)
        q=sim.d.qpos.copy();v=sim.d.qvel.copy()
        student.action(sim)
        np.testing.assert_array_equal(sim.d.qpos,q)
        np.testing.assert_array_equal(sim.d.qvel,v)
        with patch.object(grasp.GraspTeacher,'action',side_effect=AssertionError('Hidden teacher assistance')):
            result,sim=grasp.rollout(99010,student)
        self.assertTrue(result['passed'],result)
        self.assertGreater(result['lift_m'],.08)
        self.assertGreaterEqual(result['hold_seconds'],2.)
        hands,table,other=sim.contacts()
        self.assertTrue(np.all(hands>.1))
        self.assertLess(table,.05);self.assertLess(other,.05)
        self.assertLessEqual(result['max_torque_ratio'],1.)


if __name__=='__main__':unittest.main()
