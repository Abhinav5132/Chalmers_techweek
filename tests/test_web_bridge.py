"""Real MuJoCo sequencing tests using tiny deterministic recording fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from web_bridge import Studio


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / 'unitree_mujoco').symlink_to(ROOT / 'unitree_mujoco', target_is_directory=True)
        motions = self.root / 'data/motions'
        motions.mkdir(parents=True)
        for name, height in [('a', .8), ('b', .9)]:
            np.savez(motions / f'{name}.npz', fps=np.array([10.]), joint_pos=np.zeros((10, 29)),
                     body_pos_w=np.tile([[[0., 0., height]]], (10, 1, 1)),
                     body_quat_w=np.tile([[[1., 0., 0., 0.]]], (10, 1, 1)))
        self.studio = Studio(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def entry(self, motion='a', speed=1., repeats=1, id='one'):
        return dict(id=id, motion=motion, speed=speed, repeats=repeats)

    def test_order_repeats_speed_pause_seek_and_completion(self):
        s = self.studio
        s.run([self.entry(speed=2, repeats=2), self.entry('b', id='two')])
        self.assertEqual(s.state['total'], 2)
        s.tick(.75)
        self.assertEqual(s.state['index'], 0)
        s.command('pause', {})
        s.tick(10)
        self.assertEqual(s.elapsed, .75)
        s.command('resume', {})
        s.tick(.3)
        self.assertEqual(s.state['entry_id'], 'two')
        self.assertAlmostEqual(s.data.qpos[2], .9)
        s.command('seek', {'seconds': .2})
        self.assertEqual(s.state['state'], 'paused')
        self.assertEqual(s.state['index'], 0)
        s.command('resume', {})
        s.tick(10)
        self.assertEqual(s.state['state'], 'completed')
        self.assertEqual(s.state['frame'], 9)
        self.assertEqual(s.elapsed, 2)

    def test_loop_stop_reset_and_duplicate_clips(self):
        s = self.studio
        s.run([self.entry(), self.entry(id='two')], True)
        s.tick(2.25)
        self.assertEqual(s.state['state'], 'running')
        self.assertAlmostEqual(s.elapsed, .25)
        s.command('stop', {})
        s.tick(10)
        self.assertEqual(s.elapsed, .25)
        s.reset()
        self.assertEqual(s.status()['state'], 'idle')
        self.assertFalse(s.entries)
        self.assertEqual(len(s.status()['positions']), s.model.nbody * 3)

    def test_invalid_run_does_not_replace_existing_playback(self):
        s = self.studio
        s.run([self.entry()])
        for entry in [self.entry('../a'), self.entry(speed=float('nan')), self.entry(repeats=0)]:
            with self.assertRaises((ValueError, KeyError)):
                s.run([entry])
        self.assertEqual(s.state['motion'], 'a')
        self.assertEqual(s.state['state'], 'running')
        with self.assertRaises(ValueError):
            s.command('seek', {'seconds': 200})

    def test_json_lines_transport_and_error_recovery(self):
        commands = [dict(id=1, action='status'), dict(id=2, action='shell'), dict(id=3, action='reset')]
        result = subprocess.run([sys.executable, str(ROOT / 'src/web_bridge.py')],
                                input=''.join(json.dumps(c)+'\n' for c in commands),
                                capture_output=True, text=True, timeout=30, check=True)
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([r['id'] for r in responses], [1, 2, 3])
        self.assertEqual(responses[0]['result']['state'], 'idle')
        self.assertIn('error', responses[1])
        self.assertEqual(responses[2]['result']['state'], 'idle')


if __name__ == '__main__':
    unittest.main()
