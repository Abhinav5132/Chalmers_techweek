"""Physical smoke tests for the downloaded, checksum-verified teacher adapter."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from imitation_g1 import DATA, WalkingSimulation, rotation


@unittest.skipUnless((DATA / 'walk_teacher_bundle.onnx').exists(), 'Run imitation_g1.py prepare first')
class ImitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = WalkingSimulation()

    def test_seeded_reset_and_physical_action(self):
        sim = self.sim
        first = sim.reset(7)
        np.testing.assert_array_equal(first, sim.reset(7))
        self.assertEqual(first.shape, (160,))
        sim.step(np.zeros(29))
        zero_state = sim.data.qpos.copy()
        sim.reset(7)
        sim.step(np.ones(29) * .3)
        self.assertAlmostEqual(sim.data.time, .02)
        self.assertGreater(np.linalg.norm(sim.data.qpos - zero_state), 1e-5)
        with self.assertRaises(ValueError):
            sim.step(np.full(29, np.nan))

    def test_observation_orientation_and_teacher(self):
        np.testing.assert_array_equal(rotation([1., 0., 0., 0.])[:, :2].reshape(-1), [1, 0, 0, 1, 0, 0])
        sim = self.sim
        obs = sim.reset(10000)
        for _ in range(50):
            action = sim.expert(obs)
            self.assertEqual(action.shape, (29,))
            obs, fallen = sim.step(action)
            self.assertFalse(fallen)
            self.assertTrue(np.isfinite(obs).all())


if __name__ == '__main__':
    unittest.main()
