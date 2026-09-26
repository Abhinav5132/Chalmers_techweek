"""Qualify a motion teacher on physical stairs before collecting imitation data.

This is an assessment, not a trained stair controller. Reference poses are never
lifted onto the stairs and the floating pelvis is driven only by physics.
"""
import argparse
import json
import time

import mujoco
import numpy as np

from imitation_g1 import ROOT, RUN, WalkingSimulation

SCENE = ROOT / 'unitree_mujoco/unitree_robots/g1/scene_29dof.xml'
OUTPUT = RUN / 'stairs'


def staircase_model():
    spec = mujoco.MjSpec.from_file(str(SCENE))
    # Three deliberately small risers: 5 cm high, 30 cm deep, 1.2 m wide.
    for i in range(3):
        height = .05 * (i + 1)
        spec.worldbody.add_geom(name=f'stair_{i + 1}',
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[.5 + .3 * i + .15, -.1, height / 2],
            size=[.15, .6, height / 2],
            rgba=[.35 + .15 * i, .5, .7, 1],
            friction=[1, .005, .0001], contype=1, conaffinity=1)
    model = spec.compile()
    model.opt.timestep = .002
    return model


class StairSimulation(WalkingSimulation):
    def __init__(self):
        super().__init__()
        self.model = staircase_model()
        self.data = mujoco.MjData(self.model)
        self.anchor = self.model.body(self.meta['anchor_body_name']).id
        self.feet = {self.model.body(f'{side}_ankle_roll_link').id: side
                     for side in ('left', 'right')}
        self.treads = {self.model.geom(f'stair_{i}').id: i for i in range(1, 4)}

    def supports(self):
        """Only loaded foot contacts near a tread top count; risers do not."""
        result = set()
        for index, c in enumerate(self.data.contact):
            for foot, tread in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                side = self.feet.get(int(self.model.geom_bodyid[foot]))
                level = self.treads.get(int(tread))
                if side is not None and level is not None:
                    if abs(float(c.pos[2]) - .05 * level) < .01 and abs(c.frame[2]) > .7:
                        # Contact force is expressed in its contact frame.
                        if c.efc_address >= 0:
                            force = np.zeros(6)
                            mujoco.mj_contactForce(self.model, self.data, index, force)
                            if force[0] > 1:
                                result.add((side, level))
        return result


def trial(sim, seed, viewer=None):
    obs = sim.reset(seed, 0)
    initial = sim.data.qpos[:3].copy()
    visited = 0
    top_time = 0.
    max_top_time = 0.
    fallen = False
    completed = False
    for _ in range(500):
        if viewer is not None and not viewer.is_running():
            break
        start = time.monotonic()
        obs, fallen = sim.step(sim.expert(obs))
        if isinstance(sim, StairSimulation):
            contacts = sim.supports()
            if any(level == visited + 1 for _, level in contacts):
                visited += 1
            both_top = {('left', 3), ('right', 3)} <= contacts
            top_time = top_time + .02 if both_top else 0.
            max_top_time = max(max_top_time, top_time)
            completed = visited == 3 and top_time >= .5 and not fallen
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0, .02 - (time.monotonic() - start)))
        if fallen or completed:
            break
    return {'seed': seed, 'seconds': round(sim.t, 2), 'fell': fallen,
            'stairs_completed': completed, 'highest_ordered_tread': visited,
            'both_feet_top_hold_s': round(max_top_time, 2),
            'forward_displacement_m': float(sim.data.qpos[0] - initial[0]),
            'pelvis_height_m': float(sim.data.qpos[2])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate', 'view'])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sim = StairSimulation()
    if args.command == 'view':
        import mujoco.viewer
        sim.reset(730000, 0)
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.lookat[:] = [.65, -.1, .6]
            viewer.cam.distance = 3.2
            viewer.cam.azimuth = 130
            print(json.dumps(trial(sim, 730000, viewer), indent=2), flush=True)
            print('Trial ended; the simulation is frozen. Close the window to exit.', flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(.05)
        return
    stairs = [trial(sim, seed) for seed in range(730000, 730003)]
    flat = WalkingSimulation()
    control = [trial(flat, seed) for seed in range(730000, 730003)]
    qualified = all(t['stairs_completed'] for t in stairs)
    results = {'assessment': 'existing_flat_ground_teacher_on_stairs',
        'teacher_qualified_for_stair_demonstrations': qualified,
        'training_started': False,
        'geometry': {'risers': 3, 'rise_m': .05, 'tread_m': .3,
                     'width_m': 1.2, 'first_riser_x_m': .5},
        'success_rule': 'Loaded foot contact on each tread in order, then both feet on the top for 0.5 seconds without falling.',
        'limitation': 'Small qualification test, not proof of robust stair climbing. Existing teacher has a flat-ground reference and no terrain observation. No failed demonstrations are used for training.',
        'stairs': stairs, 'flat_control': control}
    path = OUTPUT / 'results.json'
    path.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
    print(f'Results: {path}')


if __name__ == '__main__':
    main()
