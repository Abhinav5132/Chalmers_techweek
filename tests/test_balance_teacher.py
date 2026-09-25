import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from balance_teacher import BalanceSimulation
from imitation_g1 import DATA


@unittest.skipUnless((DATA / 'walk_teacher_bundle.onnx').exists(), 'Teacher not downloaded')
class BalanceTests(unittest.TestCase):
    def test_standing_reference_and_task_switch(self):
        sim = BalanceSimulation()
        obs = sim.reset_task('stand', 12)
        target = obs[:58].copy()
        initial = sim.data.qpos.copy()
        for _ in range(50):
            obs, fallen = sim.step(sim.expert(obs))
        self.assertFalse(fallen)
        np.testing.assert_array_equal(obs[:58], target)
        np.testing.assert_array_equal(obs[29:58], np.zeros(29))
        self.assertEqual(sim.frame(), 0)
        self.assertGreater(np.linalg.norm(sim.data.qpos - initial), 0)
        sim.reset_task('walk', 12, start=10)
        sim.step(sim.expert(sim.observation()))
        self.assertGreater(sim.frame(), 10)
        with self.assertRaises(ValueError): sim.reset_task('invalid', 12)


if __name__ == '__main__': unittest.main()
