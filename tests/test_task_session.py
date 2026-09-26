"""Task demos must stream genuine dynamics, independent scenes and measured outcomes."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import mujoco
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from web_bridge import Studio
from task_session import TaskSession, catalog


class TaskDemoTests(unittest.TestCase):
    def test_exit_task_restores_idle_flat_scene(self):
        studio = Studio()
        studio.command('task_run', {'task': 'box_lift', 'controller': 'teacher'})
        studio.tick(.02)
        state = studio.command('task_exit', {})
        self.assertIsNone(studio.task_session)
        self.assertEqual(state['model_id'], 'g1')
        self.assertEqual(state['state'], 'idle')
        self.assertEqual(state['elapsed'], 0)
        self.assertEqual(state['mode'], 'kinematic_playback')
        self.assertNotIn('task_id', state)
        studio.tick(1)
        self.assertEqual(studio.status(), state)

    def test_box_scene_streams_object_and_controls_pause_resume_reset(self):
        studio = Studio()
        state = studio.command('task_run', {'task': 'box_lift', 'controller': 'teacher'})
        self.assertEqual(state['model_id'], 'box_lift')
        task = studio.task_session
        assert task is not None
        self.assertEqual(task.model.nq, 43)
        geometry = studio.geometry('box_lift')
        self.assertTrue(any(g['body'] == 0 and g['type'] == 6 for g in geometry['geoms']))
        self.assertTrue(any(g['body'] == task.sim.object_id for g in geometry['geoms']))
        self.assertEqual(len(state['positions']), task.model.nbody * 3)
        studio.tick(.02)
        self.assertGreater(task.data.time, 0.)
        self.assertGreater(np.linalg.norm(task.data.ctrl), 0.)
        studio.command('pause', {})
        pose = task.data.qpos.copy()
        studio.tick(1)
        np.testing.assert_array_equal(task.data.qpos, pose)
        with self.assertRaises(ValueError): studio.command('seek', {'seconds': 1})
        studio.command('resume', {})
        studio.tick(.01)
        self.assertEqual(studio.state['state'], 'running')
        studio.command('stop', {})
        t = task.data.time
        studio.tick(1)
        self.assertEqual(task.data.time, t)
        reset = studio.command('reset', {})
        self.assertEqual(reset['model_id'], 'box_lift')
        self.assertEqual(reset['elapsed'], 0.)
        self.assertEqual(reset['state'], 'idle')

    def test_stair_presets_have_distinct_world_geometry(self):
        studio = Studio()
        low, high = studio.geometry('step_5cm'), studio.geometry('step_35cm')
        low_step = next(g for g in low['geoms'] if g['body'] == 0)
        high_step = next(g for g in high['geoms'] if g['body'] == 0)
        self.assertAlmostEqual(low_step['size'][2] * 2, .05)
        self.assertAlmostEqual(high_step['size'][2] * 2, .35)
        with self.assertRaises(ValueError): studio.geometry('arbitrary_scene')

    def test_unknown_or_missing_student_does_not_replace_active_demo(self):
        studio = Studio()
        studio.command('task_prepare', {'task': 'step_5cm', 'controller': 'teacher'})
        active = studio.task_session
        with tempfile.TemporaryDirectory() as folder:
            studio.root = Path(folder)
            with self.assertRaisesRegex(ValueError, 'No qualified'):
                studio.command('task_run', {'task': 'box_lift', 'controller': 'student'})
        with self.assertRaises(ValueError): studio.command('task_run', {'task': 'arbitrary', 'controller':'teacher'})
        self.assertIs(studio.task_session, active)

    def test_dropped_box_reports_failure_without_claiming_success(self):
        session = TaskSession('box_lift', running=True)
        session.data.qpos[38] = .65
        mujoco.mj_forward(session.model, session.data)
        state = session.step()
        self.assertEqual(state['state'], 'failed')
        self.assertEqual(state['failure_reason'], 'object_dropped')
        self.assertFalse(state['passed'])
        session.step()
        self.assertEqual(session.state['elapsed'], state['elapsed'])

    def test_teacher_demos_complete_measured_physical_holds(self):
        for name in ('step_5cm', 'step_35cm', 'box_lift'):
            with self.subTest(task=name):
                session = TaskSession(name, running=True)
                while session.state['state'] == 'running': session.step()
                self.assertEqual(session.state['state'], 'completed', session.state)
                self.assertTrue(session.state['passed'])
                self.assertGreaterEqual(session.state['hold_seconds'], session.state['hold_required'])
                self.assertFalse(session.state['base_anchored'])
                if name == 'box_lift':
                    self.assertGreater(session.state['lift_m'], .08)
                    self.assertFalse(session.state['object_attached'])


if __name__ == '__main__': unittest.main()
