"""Persistent MuJoCo viewer worker, controlled by JSON lines over private pipes."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    commands: queue.Queue = queue.Queue()

    def read_commands():
        for line in sys.stdin:
            try:
                commands.put(json.loads(line))
            except ValueError:
                continue
        commands.put({'action': 'close'})

    threading.Thread(target=read_commands, daemon=True).start()
    model: Any = mujoco.MjModel.from_xml_path(str(ROOT / 'unitree_mujoco/unitree_robots/g1/scene_29dof.xml'))
    data: Any = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)
    state: dict[str, Any] = {'state': 'idle', 'viewer_open': not args.headless}
    motion = None
    started = 0.0
    fps = 60.0
    speed = 1.0
    loop = False
    with context as viewer:
        print(json.dumps({'ready': True}), flush=True)
        while viewer is None or viewer.is_running():
            try:
                command = commands.get_nowait()
            except queue.Empty:
                command = None
            if command:
                action = command['action']
                if action == 'close':
                    break
                error = None
                if action == 'play':
                    try:
                        with np.load(command['path'], allow_pickle=False) as clip:
                            replacement = {key: np.array(clip[key], copy=True) for key in
                                           ('joint_pos', 'body_pos_w', 'body_quat_w')}
                            fps = float(np.asarray(clip['fps']).reshape(-1)[0])
                        motion = replacement
                        speed, loop = command['speed'], command['loop']
                        started = time.monotonic()
                        state = {'state': 'running', 'motion': command['motion'], 'speed': speed,
                                 'loop': loop, 'viewer_open': viewer is not None,
                                 'mode': 'headless_verification' if args.headless else 'kinematic_playback'}
                    except Exception as exc:
                        error = str(exc)
                elif action == 'stop':
                    state['state'] = 'stopped'
                print(json.dumps({'error': error} if error else state), flush=True)
            if motion is not None and state['state'] == 'running':
                count = len(motion['joint_pos'])
                elapsed = (time.monotonic() - started) * (50 if args.headless else 1)
                frame = int(elapsed * fps * speed)
                if frame >= count and not loop:
                    frame = count - 1
                    state['state'] = 'completed'
                    state['exit_code'] = 0
                else:
                    frame %= count
                data.qpos[:3] = motion['body_pos_w'][frame, 0]
                data.qpos[3:7] = motion['body_quat_w'][frame, 0]
                data.qpos[7:36] = motion['joint_pos'][frame]
                mujoco.mj_forward(model, data)
            if viewer is not None:
                viewer.sync()
            time.sleep(0.01)


if __name__ == '__main__':
    main()
