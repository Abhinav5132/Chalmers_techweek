"""Real ONNX + MuJoCo tests for web physics tracking; no mocked dynamics."""
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from web_bridge import Studio
from wbc_session import catalog

READY = catalog()['available'] and (ROOT / 'data/motions/walk.npz').is_file()


@unittest.skipUnless(READY, 'Install tracking dependencies and download recordings first')
class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'models').symlink_to(ROOT / 'models', target_is_directory=True)
        (self.root / 'unitree_mujoco').symlink_to(ROOT / 'unitree_mujoco', target_is_directory=True)
        directory = self.root / 'data/motions'
        directory.mkdir(parents=True)
        with np.load(ROOT / 'data/motions/walk.npz') as source:
            clip = {k: v[:3].copy() if v.ndim > 1 else v.copy() for k, v in source.items()}
        for name in ('a', 'b'):
            np.savez(directory / f'{name}.npz', **clip)
        self.studio = Studio(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def entry(self, motion='a', repeats=1, speed=1, id='one'):
        return dict(id=id, motion=motion, speed=speed, repeats=repeats)

    def run_tracking(self, entries=None, loop=False):
        return self.studio.run(entries or [self.entry()], loop, 'wbc_tracking')

    def test_motor_dynamics_and_correct_angular_velocity_observation(self):
        self.run_tracking()
        s = self.studio
        runtime = s.tracking
        q0 = runtime.data.qpos.copy()
        s.tick(.02)
        self.assertAlmostEqual(runtime.data.time, .02)
        self.assertGreater(np.linalg.norm(runtime.data.ctrl), 0)
        self.assertFalse(np.allclose(q0, runtime.data.qpos))
        self.assertGreater(s.state['tracking_error_rad'], 0)
        self.assertEqual(s.status()['model_id'], 'wbc')
        self.assertEqual(len(s.status()['positions']), runtime.model.nbody * 3)
        self.assertLess(runtime.model.opt.gravity[2], 0)
        runtime.data.qpos[3:7] = [1, 0, 0, 0]
        runtime.data.qvel[:] = 0
        runtime.data.qvel[:6] = [4, 5, 6, .1, .2, .3]
        mujoco.mj_forward(runtime.model, runtime.data)
        np.testing.assert_allclose(runtime.runner._robot_obs()['base_ang_vel'], [.1, .2, .3], atol=1e-6)

    def test_policy_history_uses_raw_actions_and_clears_on_explicit_reset(self):
        self.run_tracking()
        runner = self.studio.tracking.runner
        scaled = runner.compute_action()
        np.testing.assert_allclose(scaled, runner.last_action * runner.action_scale)
        np.testing.assert_array_equal(runner._robot_obs()['actions'], runner.last_action)
        self.assertFalse(np.allclose(scaled, runner.last_action))
        runner.reset_to_initial_pose()
        np.testing.assert_array_equal(runner._robot_obs()['actions'], np.zeros(29))

    def test_gyro_and_fallback_use_body_frame_for_rotated_robot(self):
        self.run_tracking()
        runtime = self.studio.tracking
        runtime.data.qpos[3:7] = [2 ** -.5, 0, 0, 2 ** -.5]
        runtime.data.qvel[:] = 0
        runtime.data.qvel[:6] = [4, 5, 6, .1, .2, .3]
        mujoco.mj_forward(runtime.model, runtime.data)
        self.assertIsNotNone(runtime.runner.imu_sensor_adr)
        np.testing.assert_allclose(runtime.runner._robot_obs()['base_ang_vel'], [.1, .2, .3], atol=1e-6)
        runtime.runner.imu_sensor_adr = None
        np.testing.assert_allclose(runtime.runner._robot_obs()['base_ang_vel'], [.1, .2, .3], atol=1e-6)

    def test_sequence_duplicates_repeats_and_completion(self):
        self.run_tracking([self.entry(repeats=2), self.entry('b', id='two'), self.entry(id='three')])
        visited = []
        for _ in range(100):
            s = self.studio
            s.tick(.02)
            key = (s.state['entry_id'], s.state['repeat_index'])
            if not visited or key != visited[-1]:
                visited.append(key)
            if s.state['state'] != 'running':
                break
        self.assertEqual(visited, [('one', 1), ('one', 2), ('two', 1), ('three', 1)])
        self.assertEqual(s.state['state'], 'completed')
        self.assertEqual(s.state['completed_clips'], 4)
        self.assertEqual(s.state['reset_count'], 1)
        self.assertAlmostEqual(s.state['elapsed'], s.state['total'])

    def test_clip_switch_preserves_pose_velocity_time_and_action_history(self):
        self.run_tracking([self.entry(), self.entry('b', id='two')])
        runtime = self.studio.tracking
        assert runtime is not None
        runtime.step()
        # A displaced, rotated robot must not return to the recording origin.
        runtime.data.qpos[:2] = [3., -2.]
        runtime.data.qpos[3:7] = [2 ** -.5, 0, 0, 2 ** -.5]
        mujoco.mj_forward(runtime.model, runtime.data)
        pose, velocity = runtime.data.qpos.copy(), runtime.data.qvel.copy()
        previous_action = runtime.runner.last_action.copy()
        sim_time = runtime.data.time
        runtime.index = 1
        runtime._enter()
        np.testing.assert_array_equal(runtime.data.qpos, pose)
        np.testing.assert_array_equal(runtime.data.qvel, velocity)
        np.testing.assert_array_equal(runtime.runner.last_action, previous_action)
        self.assertEqual(runtime.data.time, sim_time)
        np.testing.assert_allclose(runtime.runner.clip_origin_pos[:2], [3., -2.])
        from controllers.wbc_runner import quat_multiply, quat_yaw
        ref_quat = runtime.runner.clip.anchor_frame(0, runtime.runner.npz_anchor_idx)[1]
        anchored = quat_yaw(quat_multiply(runtime.runner.clip_origin_quat, ref_quat))
        np.testing.assert_allclose(anchored, pose[3:7], atol=1e-7)
        self.assertEqual(runtime.state['reset_count'], 1)
        runtime.step()
        self.assertAlmostEqual(runtime.data.time, sim_time + runtime.dt)
        self.assertLess(np.linalg.norm(runtime.data.qpos[:3] - pose[:3]), .1)

    def test_transition_delay_advances_physics_and_counts_repeats_and_loop_gaps(self):
        entry = {**self.entry(repeats=2), 'delay_after': .1}
        self.run_tracking([entry])
        runtime = self.studio.tracking
        assert runtime is not None
        self.assertAlmostEqual(runtime.state['total'], .22)  # 2 x .06 + one .10 gap
        for _ in range(3): runtime.step()
        self.assertEqual(runtime.state['completed_clips'], 1)
        pose = runtime.data.qpos.copy()
        for _ in range(5):
            runtime.step()
            self.assertEqual(runtime.state['phase'], 'transition_hold')
            self.assertEqual(runtime.state['repeat_index'], 1)
        self.assertAlmostEqual(runtime.data.time, .16)
        self.assertFalse(np.array_equal(runtime.data.qpos, pose))
        self.assertEqual(runtime.state['reset_count'], 1)
        np.testing.assert_array_equal(runtime.runner._reference()['ref_base_lin_vel_b'], np.zeros(3))
        runtime.step()
        self.assertEqual(runtime.state['repeat_index'], 2)
        for _ in range(2): runtime.step()
        self.assertEqual(runtime.state['state'], 'completed')
        self.assertAlmostEqual(runtime.state['elapsed'], .22)
        self.run_tracking([entry], loop=True)
        self.assertAlmostEqual(runtime.state['total'], .32)
        for _ in range(17): runtime.step()
        self.assertEqual(runtime.state['cycle'], 2)
        self.assertEqual(runtime.state['reset_count'], 1)

    def test_transition_delay_validation_preserves_active_run(self):
        self.run_tracking()
        state = self.studio.state.copy()
        for invalid in (-1, 11, float('nan'), True):
            with self.assertRaisesRegex(ValueError, 'Transition delay'):
                self.run_tracking([{**self.entry(), 'delay_after': invalid}])
            self.assertEqual(self.studio.state, state)

    def test_pause_resume_stop_loop_and_switch_back(self):
        self.run_tracking(loop=True)
        s = self.studio
        s.tick(.02)
        s.command('pause', {})
        paused = s.tracking.data.qpos.copy()
        s.tick(20)
        np.testing.assert_array_equal(s.tracking.data.qpos, paused)
        with self.assertRaises(ValueError):
            s.command('seek', {'seconds': 0})
        s.command('resume', {})
        for _ in range(12): s.tick(.02)
        self.assertGreater(s.state['cycle'], 1)
        self.assertEqual(s.state['reset_count'], 1)
        self.assertAlmostEqual(s.tracking.data.time, s.state['simulation_seconds'])
        s.command('stop', {})
        steps = s.state['control_steps']
        s.tick(2)
        self.assertEqual(s.state['control_steps'], steps)
        s.run([self.entry()])
        self.assertEqual(s.status()['model_id'], 'g1')
        self.assertEqual(s.state['mode'], 'kinematic_playback')
        s.command('reset', {})
        self.assertEqual(s.state['state'], 'idle')

    def test_detected_fall_stops_without_teleporting_to_next_clip(self):
        self.run_tracking([self.entry(repeats=3)])
        s = self.studio
        s.tracking.data.qpos[2] = 1.
        s.tracking.data.qpos[3:7] = [2 ** -.5, 2 ** -.5, 0, 0]
        mujoco.mj_forward(s.tracking.model, s.tracking.data)
        s.tick(.02)
        self.assertEqual(s.state['state'], 'fallen')
        self.assertTrue(s.state['fell'])
        self.assertIn('failure_reason', s.state)
        self.assertEqual(s.state['reset_count'], 1)
        s.tick(5)
        self.assertEqual(s.state['control_steps'], 1)

    def test_retiming_scales_reference_velocities_and_preserves_policy_clock(self):
        from controllers.wbc_runner import ClipReference
        path = str(self.root / 'data/motions/a.npz')
        normal = ClipReference(path, target_fps=50)
        slow = ClipReference(path, target_fps=50, speed=.5)
        self.assertGreater(slow.n_frames, normal.n_frames)
        np.testing.assert_allclose(slow.body_lin_vel_w[0], normal.body_lin_vel_w[0] * .5)
        self.run_tracking([self.entry(speed=.5)])
        self.studio.tick(.02)
        self.assertAlmostEqual(self.studio.tracking.data.time, .02)

    def test_rejects_incompatible_clip_without_replacing_current_run(self):
        self.run_tracking()
        s = self.studio
        state = s.state.copy()
        path = self.root / 'data/motions/b.npz'
        with np.load(path) as source:
            bad = {k: v.copy() for k, v in source.items() if k != 'body_ang_vel_w'}
        np.savez(path, **bad)
        with self.assertRaisesRegex(ValueError, 'body_ang_vel_w'):
            self.run_tracking([self.entry('b')])
        self.assertEqual(s.state, state)
        self.assertFalse(next(m for m in s.catalog()['motions'] if m['id'] == 'b')['tracking_available'])


class MissingBundleTests(unittest.TestCase):
    def test_missing_bundle_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = catalog(Path(directory))
            self.assertFalse(result['available'])
            self.assertIn('policy.onnx', result['error'])


if __name__ == '__main__':
    unittest.main()
