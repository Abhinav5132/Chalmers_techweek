"""A learned feedback student must climb without hidden teacher assistance."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from stair_feedback import FeedbackStudent, QUALIFIED, qualified_path
from stair_curriculum import Teacher
from stair_learning import run


@unittest.skipUnless(QUALIFIED.exists(),'Run just step-feedback-train to create a qualified policy')
class FeedbackTests(unittest.TestCase):
    def test_student_climbs_without_teacher_or_pose_writes(self):
        path=Path(json.loads(QUALIFIED.read_text())['checkpoint'])
        student=FeedbackStudent(path)
        sim=Teacher(student.height,99010)
        before=sim.d.qpos.copy();a=student.action(sim)
        np.testing.assert_array_equal(sim.d.qpos,before)
        sim.d.qvel[3]=.03
        self.assertGreater(np.linalg.norm(student.action(sim)-a),.01)
        # Entire physical rollout must succeed with the teacher QP disabled.
        with patch.object(Teacher,'action',side_effect=AssertionError('Teacher called during student evaluation')):
            report,_,_,sim=run(student.height,99010,student)
        self.assertTrue(report['passed'],report)
        self.assertEqual(report['teacher_mixing_beta'],0)
        self.assertLessEqual(report['max_torque_ratio'],1)
        self.assertGreaterEqual(sim.top_hold_seconds,2.9)

class HeightSelectionTests(unittest.TestCase):
    def test_height_specific_checkpoint_selection(self):
        self.assertEqual(qualified_path(.05),QUALIFIED)
        self.assertNotEqual(qualified_path(.35),QUALIFIED)
        s=Teacher(.35)
        g=s.m.geom('curriculum_step')
        self.assertAlmostEqual(float(g.pos[2]+g.size[2]),.35)
        self.assertEqual(s.m.neq,0)

    @unittest.skipUnless(qualified_path(.35).exists(),'Train the 35 cm student first')
    def test_35cm_student_without_teacher(self):
        student=FeedbackStudent(Path(json.loads(qualified_path(.35).read_text())['checkpoint']))
        with patch.object(Teacher,'action',side_effect=AssertionError('Teacher assistance')):
            report,_,_,_=run(.35,99010,student)
        self.assertTrue(report['passed'],report)
        self.assertEqual(report['teacher_mixing_beta'],0.)
        with self.assertRaisesRegex(ValueError,'different step height'):
            student.action(Teacher(.05))

if __name__=='__main__':unittest.main()
