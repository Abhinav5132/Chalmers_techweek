# G1 Motion Studio

Local React + TypeScript + Vite + Tailwind CSS v4 frontend, Hono/Node backend,
and a persistent Python/MuJoCo worker. The robot is rendered in the browser
with Three.js using the repository's actual G1 meshes and MuJoCo body poses.
No desktop viewer, API key, or external service is required for recording playback.

## Start

From the repository root (Node 22.12+ and the project's Python environment required):

```sh
uv sync --group training --group tracking
# If you haven't downloaded the example recordings:
just download-motions
npm --prefix web install
npm --prefix web run dev
```

Open **http://127.0.0.1:5173**. Vite proxies `/api` to Hono on port 8787.
Recording playback needs only `uv sync`. Routine physics tracking additionally
needs `--group tracking`; the `training` group is for the separate stand/walk tools.
Set `STUDIO_API_PORT` to change the backend port for both Vite and Hono.
Set `ROBOT_PYTHON` to an absolute Python executable to override `.venv/bin/python`.

For a production build served entirely by Hono:

```sh
npm --prefix web run build
npm --prefix web start
```

Then open **http://127.0.0.1:8787**. Both servers bind only to loopback.
This is a single-user local studio: browser tabs share one simulation session.
Closing the browser does not stop a routine; Stop ends playback, and terminating
Hono terminates its Python worker. Long-running looped routines continue until stopped.

## Use

- Add clips with **+** or drag them from the library onto the timeline.
- Drag timeline clips to reorder them. **Earlier/Later** also work with keyboard/touch.
- Select a clip to change speed (0.25–2×), repeats (1–20), duplicate, or remove it.
- **Play**, **Pause/Resume**, **Stop**, **Reset**, and scrub the playback position.
  Seeking pauses at the selected time. Pause playback to edit a routine; the first
  edit stops the paused run, and Play starts the updated sequence from the beginning.
  Running playback shows a **Pause to edit** button. Camera orbit/zoom remains available.
- **Loop** repeats the full routine. Individual repeats are separate.
- **Save routine** persists it under `.robot-runtime/web-routines/`. **Open** loads
  saved routines or creates a new one. **Export/Import JSON** transfers routines.
  The current draft is also stored in browser localStorage. Exported JSON contains
  clip IDs, not the motion recordings themselves. Importing/replacing a draft does not
  modify previously saved routines.
- `/` focuses motion search; Escape closes the saved-routines dialog.
- **Run with physics** executes the whole arranged routine using the phase-2 ONNX
  controller and its matching MuJoCo scene. The live view shows simulated body poses,
  not reference animation. Speed, repeats, ordering and full-routine looping are used.
  Pause/resume/stop work; seeking is disabled for physical trials.
- **Physics lab** shows tracking RMS joint error (radians), pelvis height, completed
  clip segments, detected falls and the failure reason. Expand **Separate stand / walk
  trials** for the older teacher/student controllers when their separate assets exist.

Recording playback directly sets poses. It is **not a physics-balanced dance**.
Clip boundaries reset to the next recording's root pose; no transition blending is
implemented. The browser interpolates displayed poses for smoother rendering only.
The waveform graphics are decorative clip markers, not measured audio or motion data.

The phase-2 tracking runtime, exported policy and matching training scene are now
integrated on `main`. See [models/README.md](../models/README.md) for their source.
Within a physics segment, only motor torques and MuJoCo dynamics move the robot.
Each clip/repeat begins with a reference-pose reset, as in the phase-2 branch.
There is no blending or balance-preserving transition between segments.

The controller was trained on these recordings, not every possible tempo or new
clip. Retiming adjusts reference velocities, not the physical timestep, and can
cause falls. A detected fall stops the whole routine; there is no animation fallback.
Completion means no fall was detected, not that every target was tracked accurately.
The fall heuristic uses pelvis height < 0.25 m or tilt > about 78 degrees and can
flag intentional acrobatics. Policies are not trained from this UI.

The existing teacher/student stand/walk tools remain separate. Their missing
checkpoints do not prevent the bundled WBC policy from tracking a routine.

## Implementation

- `src/App.tsx`: routine editor, library, inspector, persistence and controls.
- `src/RobotViewer.tsx`: mesh rendering and camera controls; simulation stays in Python.
- `server/app.ts`: validated HTTP API and atomic routine-file saves.
- `server/bridge.ts`: correlated JSON-lines requests to one managed Python process.
- `../src/web_bridge.py`: catalog validation reused from `MotionPlayer`, MuJoCo
  recording sequencing, body-pose snapshots, and existing `PhysicsSession` integration.
- `../src/wbc_session.py`: phase-2 physical sequence execution and trial outcomes.
- `../src/controllers/wbc_runner.py`: ONNX observations/inference and per-step PD
  motor torques with the branch's torque-speed envelope.

`POST /api/run` accepts the routine plus `mode: "kinematic_playback"` (default) or
`mode: "wbc_tracking"`. `GET /api/model?scene=g1|wbc` supplies the matching geometry;
status includes `model_id` to prevent rendering physics poses against the wrong model.
Missing assets or incompatible clips fail explicitly; recording preview stays available.

The frontend fetches status sequentially approximately 14 times/second, slowing down
on disconnect. Geometry is loaded once and compressed over HTTP. No browser request
is executed as a shell command; commands, clip IDs, speeds, repeats, and file IDs are
validated. Failed commands are displayed in the UI.

## Verification

```sh
npm --prefix web run build
npm --prefix web test
uv run --group training --group tracking python -m unittest discover -s tests -p test_web_bridge.py
uv run --group training --group tracking python -m unittest discover -s tests -p test_wbc_tracking.py
uv run --group training --group tracking ty check
```

Browser drag-and-drop and physics-control regression tests use real mouse gestures with an isolated,
mocked API, so they never change the live simulation:

```sh
npm --prefix web exec -- playwright install chromium
npm --prefix web run test:e2e
```
