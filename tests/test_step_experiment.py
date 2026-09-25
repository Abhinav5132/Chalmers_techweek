import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from step_experiment import TouchdownDetector, run_experiment, read_results


class MeasurementTests(unittest.TestCase):
    def test_touchdown_debounce_and_first_frame_measurement(self):
        detector = TouchdownDetector(2)
        direction = np.array([1., 0.])
        positions = {'left': np.array([0.41, 0., 0.]), 'right': np.array([0.60, 0., 0.])}
        def update(frame, right):
            return detector.update(frame, {'left': True, 'right': right}, positions, direction)
        self.assertEqual(update(0, True), [])  # Initial stance is not a step.
        self.assertEqual(update(1, True), [])
        update(2, False)
        update(3, False)
        self.assertEqual(update(4, True), [])
        positions['right'][0] = 0.62
        events = update(5, True)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]['achieved_m'], 0.19)
        self.assertEqual(events[0]['frame'], 4)
        self.assertTrue(events[0]['valid_forward_step'])
        self.assertEqual(update(6, True), [])
        update(7, False)  # Single-frame chatter must not re-arm the detector.
        self.assertEqual(update(8, True), [])
        self.assertEqual(update(9, True), [])

    def test_heading_and_unsupported_landing(self):
        detector = TouchdownDetector(2)
        direction = np.array([0., 1.])
        positions = {'left': np.array([0., 0.1, 0.]), 'right': np.array([0., 0.3, 0.])}
        for frame in (0, 1):
            detector.update(frame, {'left': False, 'right': False}, positions, direction)
        detector.update(2, {'left': False, 'right': True}, positions, direction)
        events = detector.update(3, {'left': False, 'right': True}, positions, direction)
        self.assertAlmostEqual(events[0]['achieved_m'], 0.2)
        self.assertFalse(events[0]['valid_forward_step'])

    def test_invalid_experiments(self):
        for targets in ([], [float('nan')], [-0.1], [2.]):
            with self.assertRaises(ValueError):
                run_experiment(targets=targets)
        with self.assertRaises(ValueError):
            read_results('../secrets')


if __name__ == '__main__':
    unittest.main()
