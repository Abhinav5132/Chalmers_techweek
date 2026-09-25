"""Exercise catalog validation and the real MCP transport without an LLM key."""
import asyncio
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_mcp import MotionPlayer


class CatalogTests(unittest.TestCase):
    def test_persistent_worker(self):
        if not (ROOT / "data/motions/bow.npz").exists():
            self.skipTest("Download motion clips first")
        player = MotionPlayer(headless=True)
        try:
            player.play("walk", loop=True)
            process = player.process
            self.assertIsNotNone(process)
            self.assertEqual(player.play("bow")["motion"], "bow")
            self.assertIs(player.process, process)
            deadline = time.monotonic() + 10
            while player.status()["state"] == "running" and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(player.status()["state"], "completed")
            assert process is not None
            self.assertIsNone(process.poll())
            player.play("walk", loop=True)
            self.assertEqual(player.stop()["state"], "stopped")
            self.assertIsNone(process.poll())
            self.assertIs(player.process, process)
            player.play("bow")
            self.assertIs(player.process, process)
        finally:
            player.close()

    def test_physics_uses_same_worker(self):
        from physics_session import catalog
        if not catalog()['student_available'] or not catalog()['teacher_available']:
            self.skipTest('Train the imitation model first')
        player = MotionPlayer(headless=True)
        try:
            player.play('walk', loop=True)
            process = player.process
            state = player.physics('stand', seconds=1)
            self.assertEqual(state['mode'], 'motor_driven_physics')
            self.assertIs(player.process, process)
            deadline = time.monotonic() + 15
            while player.status()['state'] == 'running' and time.monotonic() < deadline:
                time.sleep(.02)
            state = player.status()
            self.assertEqual(state['state'], 'completed')
            self.assertEqual(state['control_steps'], 50)
            self.assertAlmostEqual(state['simulation_seconds'], 1.)
            self.assertTrue(state['simulation_paused'])
            player.physics('walk', controller='teacher', seconds=10, push_force=12)
            self.assertEqual(player.stop()['state'], 'stopped')
            player.play('bow')
            self.assertIs(player.process, process)
            for task, force in [('invalid', 8), ('walk', float('nan'))]:
                with self.assertRaises(ValueError): player.physics(task, push_force=force)
        finally:
            player.close()

    def test_paths_and_invalid_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = root / "data/motions"
            clips.mkdir(parents=True)
            player = MotionPlayer(root)
            np.savez(clips / "bad.npz", fps=[0], joint_pos=np.zeros((1, 29)),
                     body_pos_w=np.zeros((1, 1, 3)), body_quat_w=np.zeros((1, 1, 4)))
            self.assertEqual(player.list()["motions"], [])
            self.assertEqual(len(player.list()["invalid"]), 1)
            for value in ("../secret", "/tmp/secret", "missing"):
                with self.assertRaises(ValueError):
                    player.path(value)
            outside = root / "outside.npz"
            outside.touch()
            (clips / "escape.npz").symlink_to(outside)
            with self.assertRaises(ValueError):
                player.path("escape")
            for speed in (float("nan"), 0, 3):
                with self.assertRaises(ValueError):
                    player.play("bad", speed)

    def test_real_mcp_playback(self):
        if not (ROOT / "data/motions/walk.npz").exists():
            self.skipTest("Download motion clips with just setup-motions first")
        asyncio.run(self.check_transport())

    async def check_transport(self):
        params = StdioServerParameters(command=str(ROOT / ".venv/bin/python"),
            args=[str(ROOT / "src/robot_mcp.py")], env={**os.environ, "ROBOT_MCP_HEADLESS": "1"})
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                names = {t.name for t in (await session.list_tools()).tools}
                self.assertEqual(names, {"list_motions", "play_motion", "playback_status", "stop_motion", "close_robot_window",
                                         "run_step_length_experiment", "get_step_length_results", "play_larger_steps", "list_physics_skills", "run_physics_motion"})
                physical = await session.call_tool('list_physics_skills', {})
                self.assertFalse(physical.isError)
                if physical.structuredContent and physical.structuredContent['student_available']:
                    trial = await session.call_tool('run_physics_motion', {'task': 'stand', 'seconds': 1})
                    self.assertFalse(trial.isError)
                    assert trial.structuredContent is not None
                    self.assertEqual(trial.structuredContent['mode'], 'motor_driven_physics')
                bad_physical = await session.call_tool('run_physics_motion', {'task': 'fly'})
                self.assertTrue(bad_physical.isError)
                catalog = await session.call_tool("list_motions", {})
                self.assertFalse(catalog.isError)
                invalid = await session.call_tool("play_motion", {"motion": "../secret"})
                self.assertTrue(invalid.isError)
                started = await session.call_tool("play_motion", {"motion": "walk"})
                self.assertFalse(started.isError)
                for _ in range(100):
                    result = await session.call_tool("playback_status", {})
                    assert result.structuredContent is not None
                    if result.structuredContent["state"] != "running":
                        break
                    await asyncio.sleep(0.1)
                assert result.structuredContent is not None
                self.assertEqual(result.structuredContent["state"], "completed")
                self.assertEqual(result.structuredContent["exit_code"], 0)
                for _ in range(2):
                    closed = await session.call_tool("close_robot_window", {})
                    self.assertFalse(closed.isError)
                    assert closed.structuredContent is not None
                    self.assertEqual(closed.structuredContent["state"], "closed")
                    self.assertFalse(closed.structuredContent["viewer_open"])
                reopened = await session.call_tool("play_motion", {"motion": "walk", "loop": True})
                self.assertFalse(reopened.isError)
                assert reopened.structuredContent is not None
                self.assertEqual(reopened.structuredContent["state"], "running")
                closed = await session.call_tool("close_robot_window", {})
                self.assertFalse(closed.isError)
                experiment = await session.call_tool("run_step_length_experiment",
                    {"targets": [0.20], "repetitions": 1, "max_steps": 2})
                self.assertFalse(experiment.isError)
                assert experiment.structuredContent is not None
                self.assertFalse(experiment.structuredContent["target_applied_to_controller"])
                self.assertEqual(experiment.structuredContent["visualization"]["motion"], "walk")
                self.assertIn(experiment.structuredContent["visualization"]["state"], ("running", "completed"))
                saved = await session.call_tool("get_step_length_results",
                    {"experiment_id": experiment.structuredContent["experiment_id"]})
                self.assertFalse(saved.isError)
                data_only = await session.call_tool("run_step_length_experiment",
                    {"targets": [0.20], "repetitions": 1, "max_steps": 2, "show_simulation": False})
                self.assertFalse(data_only.isError)
                assert data_only.structuredContent is not None
                self.assertEqual(data_only.structuredContent["visualization"]["state"], "not_requested")


if __name__ == "__main__":
    unittest.main()
