import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from recovery_training import RecoverySimulation, promotion_allowed, KINDS
from imitation_g1 import DATA


@unittest.skipUnless((DATA / 'walk_teacher_bundle.onnx').exists(), 'Teacher missing')
class RecoveryTests(unittest.TestCase):
    def test_disturbances_reset_and_push_timing(self):
        sim = RecoverySimulation()
        obs = sim.reset_scenario('stand', 17, 'combined')
        friction = sim.model.geom_friction.copy()
        np.testing.assert_array_equal(obs, sim.reset_scenario('stand', 17, 'combined'))
        np.testing.assert_array_equal(friction, sim.model.geom_friction)
        self.assertFalse(np.array_equal(friction, sim.original_friction))
        self.assertLessEqual(sim.scenario['additional_joint_noise_max_rad'], .008)
        sim.t = 2.
        sim.step(sim.expert(sim.observation()))
        self.assertAlmostEqual(np.linalg.norm(sim.data.xfrc_applied[sim.pelvis, :2]), 8.)
        sim.t = 2.2
        sim.step(sim.expert(sim.observation()))
        self.assertEqual(np.linalg.norm(sim.data.xfrc_applied), 0)
        sim.reset_scenario('walk', 17, 'clean')
        np.testing.assert_array_equal(sim.model.geom_friction, sim.original_friction)
        self.assertEqual(np.linalg.norm(sim.data.xfrc_applied), 0)


class PromotionTests(unittest.TestCase):
    def test_regression_is_rejected(self):
        before = [{'mode': mode, 'kind': kind, 'passed': True, 'max_drift_m': .03, 'tracking_rmse_m': .08}
                  for mode in ('stand', 'walk') for kind in KINDS]
        after = [r.copy() for r in before]
        self.assertTrue(promotion_allowed(before, after))
        after[0]['passed'] = False
        self.assertFalse(promotion_allowed(before, after))
        after = [r.copy() for r in before]
        after[0]['max_drift_m'] = .05
        self.assertFalse(promotion_allowed(before, after))


if __name__ == '__main__': unittest.main()
