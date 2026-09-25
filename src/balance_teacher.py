"""Stationary-reference and longer walking tasks for the G1 imitation teacher."""
import numpy as np
import mujoco
from imitation_g1 import WalkingSimulation


class BalanceSimulation(WalkingSimulation):
    def __init__(self):
        super().__init__()
        self.walk_reference = self.reference
        self.mode = 'walk'
        # A quiet end-of-clip pose, validated separately as a standing teacher.
        self.stand_reference = [a[600:601].copy() for a in self.walk_reference]
        for index in (1, 4, 5):
            self.stand_reference[index][:] = 0

    def reset_task(self, mode, seed, start=0, disturbance=.02):
        if mode not in ('stand', 'walk'):
            raise ValueError('Task must be stand or walk')
        self.mode = mode
        self.reference = self.stand_reference if mode == 'stand' else self.walk_reference
        super().reset(seed, 0 if mode == 'stand' else start)
        rng = np.random.default_rng(seed)
        self.data.qvel[:2] += rng.uniform(-disturbance, disturbance, 2)
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def frame(self):
        return 0 if self.mode == 'stand' else super().frame()
