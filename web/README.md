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

## Task demos

Open **Tasks** to select **Climb a 5 cm step**, **Climb a 35 cm step**, or
**Lift a box**. Selecting a card prepares that scene without starting it.
**Run demo**, **Pause/Resume**, **Stop**, and **Reset demo** control its own trial;
Reset returns to the selected scene's initial state. Fullscreen remains available
on the live viewer. These tasks run independently of the dance timeline.

The Tasks integration imports `grabObj` commit `5d44ab3` and reuses
`stair_curriculum.Teacher`, `grasp_learning.Simulation` and `GraspTeacher`.
`task_session.py` advances them at their native 10 ms control period; MuJoCo uses
2 ms substeps. Step geometry, pedestal and free box are streamed to the browser
along with the robot. Neither the box nor the robot is attached to an anchor.

Progress reflects measured foot/hand contact, lift height and stable hold, not
elapsed-time success guesses. Each preset runs a full 23-second step trial or
12-second lift trial. Step success requires both feet on top with a final hold
of at least 2.9 seconds. Box success uses the upstream >8 cm lift, two-hand contact,
no support contact, uprightness and >=2-second final hold checks. Failure reasons
remain visible; a stopped or paused trial is not a successful trial.

**Teacher** works without trained weights. **Learned student** appears under
Controller options only as available when a valid, height-matched qualified
checkpoint is installed. This checkout did not include the friend's local weights.
The adapter reads the branch's existing `qualified.json` / `feedback-qualified*.json`
pointers. Use `just grasp-train` or `just step-feedback-train <height>` outside the
web UI to train, then refresh the library. Missing or incompatible checkpoints
produce an explicit error; the app never substitutes a teacher for a requested student.

These are known-scene demonstrations. They do not implement general object
recognition, arbitrary stairs, walking while carrying, or a combined climb-and-pickup
mission. The current box is 150 g, lifted by friction between fixed rubber hands.

## Motion library

`just download-motions` downloads all 61 training recordings from
[exptech/g1-moves](https://huggingface.co/datasets/exptech/g1-moves), about 195 MB,
using the pinned revision and attribution in `src/motion_catalog.json`.
The source is licensed CC BY 4.0; performer credits are retained in the catalog.
The recordings remain in ignored `data/motions/`, not in Git. Existing valid
files are retained; downloads are validated before atomically replacing files.
The original `walk`, `step_touch`, and `bow` IDs remain compatible with saved routines.
Refresh the motion library after downloading; search supports names, IDs and categories
(`dance`, `karate`, `bonus`). All local valid NPZ clips are shown, including custom ones.

These additional recordings expand the available targets, not the policy's training
set. The bundled WBC policy was exported with a three-clip training manifest.
Extra clips are experimental: compatible file format does not guarantee stable
tracking, and a detected fall still stops playback. See the
[full-library physics trial results](../docs/MOTION_LIBRARY_TRIALS.md) for the
54 completed and 7 fallen trials at normal speed.

## Use

- Add clips with **+** or drag them from the library onto the timeline.
- Drag timeline clips to reorder them. **Earlier/Later** also work with keyboard/touch.
- Select a clip to change speed (0.25–2×), repeats (1–20), duplicate, or remove it.
- **Play** runs the routine with WBC physics; library **Preview** runs one clip with
  the same physics controller. **Pause/Resume**, **Stop**, and **Reset** control playback.
  Seeking is disabled for physics. Pause playback to edit a routine; the first
  edit stops the paused run, and Play starts the updated sequence from the beginning.
  Running playback shows a **Pause to edit** button. The trash button on each timeline
  clip also works during playback: it stops the run before removing that instance.
  Camera orbit/zoom remains available. Use **Enter fullscreen** in the viewer to expand
  it, then **Exit fullscreen** or Escape to return.
- **Transition delay** in the selected clip inspector sets a 0–10 second hold after
  each play of that clip, including repeats and loop boundaries. Zero switches
  immediately. The final clip of a non-looping routine has no trailing hold.
  The value is saved/exported as `delay_after`; older routines default to zero.
  This follows the phase-2 branch’s delay-after concept, not pose blending: the
  WBC tracks the final pose with zero reference base velocity while MuJoCo keeps
  stepping. A long hold on an unstable pose can still fall. Delays count toward
  the routine duration and are quantized to the 20 ms controller period.
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

The backend retains a recording-only playback API, but all UI Play/Preview actions
use physics.
Clip changes preserve position, orientation, velocity and controller history. The browser interpolates displayed poses for smoother rendering only.
The waveform graphics are decorative clip markers, not measured audio or motion data.

The phase-2 tracking runtime, exported policy and matching training scene are now
integrated in the web studio. See [models/README.md](../models/README.md) for their source.
Only starting a new routine resets to its first reference pose. Clip changes,
repeats and full-routine loops reuse the phase-2 workflow engine’s anchored
transition: the new reference is aligned to the current heading and horizontal
position while physical state, simulation time and policy history continue.
Motor torques and MuJoCo dynamics move the robot through each boundary.
Reference targets switch directly; arbitrary clip combinations can still lose balance.

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
Missing assets or incompatible clips disable the corresponding Play/Preview actions;
the UI does not fall back to recording playback.

The frontend targets 60 status updates/second, accounting for request time and
keeping only one request in flight. Slow responses or browser throttling reduce
the rate; disconnects back off to 1.5 seconds between attempts. This does not change
the WBC policy's 50 Hz control rate or MuJoCo's physics timestep.
Geometry is loaded once and compressed over HTTP. No browser request
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
