import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dagger_g1 import collect_corrections
from imitation_g1 import SHA, network, train


class DaggerTests(unittest.TestCase):
    def test_teacher_labels_student_actions(self):
        class Simulation:
            def reset(self, seed, start):
                self.t = 0
                return np.zeros(160, np.float32)
            def expert(self, obs):
                return np.ones(29, np.float32) * 7
            def step(self, action):
                np.testing.assert_array_equal(action, np.ones(29) * 2)
                self.t += .02
                return np.ones(160, np.float32), True
        data, reports = collect_corrections(Simulation(), lambda obs: np.ones(29) * 2, 5, 60, 100060)
        np.testing.assert_array_equal(data['actions'], np.ones((5, 29)) * 7)
        np.testing.assert_array_equal(data['episode_ids'], np.arange(60, 65))
        self.assertEqual(len(reports), 5)
        self.assertTrue(all(r['fell'] for r in reports))

    def test_resume_preserves_observation_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            model = network()
            mean, std = torch.ones(160) * 3, torch.ones(160) * 2
            torch.save({'weights': model.state_dict(), 'mean': mean, 'std': std,
                        'teacher_sha256': SHA}, path / 'policy.pt')
            np.savez(path / 'data.npz', observations=np.zeros((10, 160), np.float32),
                     actions=np.zeros((10, 29), np.float32), episode_ids=np.arange(10))
            train(1, resume=True, dataset=path / 'data.npz', output=path)
            checkpoint = torch.load(path / 'policy.pt', weights_only=True)
            torch.testing.assert_close(checkpoint['mean'], mean)
            torch.testing.assert_close(checkpoint['std'], std)
            self.assertIn('optimizer', checkpoint)


if __name__ == '__main__': unittest.main()
