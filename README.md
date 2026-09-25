# Unitree G1 Humanoid Skill Orchestrator (SKF Hackathon)

Visual, modular, and natural-language-driven control for the **Unitree G1 29-DoF Humanoid Robot** in MuJoCo.

See the full architectural specification in [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

## Robot commands through Hermes and MCP

[Hermes Agent](https://github.com/nousresearch/hermes-agent) turns your text requests
into calls to the project's local **Model Context Protocol (MCP)** server. The server
runs saved imitation controllers for **physical standing and walking**, and can also
replay recordings in `data/motions/`, in one persistent MuJoCo window.

Physical trials use motor torques, gravity, ground contact and optional pushes. Other
recorded clips remain pose playback. Stair climbing, commanded step-length adjustment
and real hardware control are not implemented. The AI provider selects tools; a local
controller performs the movement. Physical skills require the training dependency group
and local teacher/checkpoint files; `just agent` includes those dependencies.

### 1. Install the requirements

Run every command below from the repository root, `Chalmers_techweek` (the directory
containing this README and `justfile`). The agent launcher currently uses macOS/Linux
executable paths. Interactive playback requires a desktop display.

You need Git, Python 3.11, [uv](https://docs.astral.sh/uv/), and
[just](https://github.com/casey/just). On macOS, if Homebrew is already installed:

```bash
brew install uv just
uv python install 3.11
```

Set up the simulation and install Hermes:

```bash
just init
just agent-install
```

- `just init` installs project dependencies, ensures the G1 model and URDF are present,
  downloads the walk, step-touch and bow recordings, and runs the code checker.
- `just agent-install` installs the pinned Hermes source and MCP client support into
  a separate environment. It does not require your AI provider key.

The simulation uses `.venv`. Hermes uses `.tools/hermes-venv`, with its source in
`.tools/hermes-agent`, pinned to revision `6da966f22c94c5b74a54b046bfcecde87f2ec519`.
The separate environments avoid conflicts between robot and agent dependencies.

### 2. Configure the AI provider

Create an API key with your chosen OpenAI-compatible provider. Pick a model that supports
**tool calling**, then run:

```bash
just agent-configure
```

The prompts ask for three values:

| Prompt | What to enter | Gonka Proxy example |
| --- | --- | --- |
| API base URL | The API endpoint, not the dashboard or API-key management page | `https://proxy.gonka.gg/v1` |
| Exact model ID | Full ID from your provider's available-models list | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| API key | Your provider key, without surrounding quotes | Enter privately when prompted |

For Gonka Proxy, obtain the key from **API Keys** and the model ID from
**Docs → Models → Available models**, or the provider's `GET /v1/models` endpoint.
`MiniMaxAI/MiniMax-M2.7` is another listed example. Model availability can change;
use an ID currently offered by your provider. Do not use a `/dashboard/keys` URL as the
API base URL.

API key input is hidden, so nothing appears while you paste it. Press Enter afterward.
Only the AI provider needs a key; the local MCP server and public motion downloads do not.
Provider calls use your provider account and may consume paid credits.

Configuration and sessions are stored in `.hermes-robot/`. The key is saved in
`.hermes-robot/.env` with owner-only permissions; `config.yaml` references it through
`ROBOT_LLM_API_KEY`. These files, downloaded tools, runtime logs, and motion data are
ignored by Git. Never paste real keys into chat, screenshots, or tracked files.
Re-run `just agent-configure` to replace a key or change providers. This updates the
project-specific configuration; use it instead of the generic `hermes setup` command.

### 3. Start the agent and send commands

```bash
just agent
```

Type plain-language requests **inside the Hermes prompt**, not at the normal shell prompt:

| Request | Result |
| --- | --- |
| "Stand using the learned controller." | Runs a ten-second motor-driven standing trial. |
| "Walk using physics, then tell me whether it fell." | Runs the learned walking controller and checks status. |
| "Use the teacher to stand with 20 N pushes." | Runs the teacher with timed physical pushes. |
| "What robot movements are available?" | Lists the valid local recordings. |
| "Make the robot bow." | Opens the viewer if needed and plays the bow once. |
| "Play the side-step recording at half speed." | Plays step-touch at 0.5× speed. |
| "Loop the walking recording." | Repeats the walking recording until replaced or stopped. |
| "Play the bow instead." | Switches the existing viewer to the bow recording. |
| "Pause and keep the window open." | Freezes the current pose without closing the viewer. |
| "Stop the robot." / "Close the robot window." | Stops playback and closes the viewer; Hermes remains open. |
| "What is the playback status?" | Reports the current playback state or failure. |

A finished motion pauses simulation in the **same window**; this frozen view is not continuing active balance. New movements reuse that
window, preserving its camera view. After you close it, another play request opens a new
window. Playback speed is restricted to 0.25–2× and changes timing, not step length.

Press **Ctrl+D** at the Hermes prompt to exit the agent and close its viewer.
After updating the agent/server code, exit and run `just agent` again so Hermes discovers
the updated tools. Existing sessions do not automatically gain newly added tools.

### MCP tools and files

| Tool | Purpose |
| --- | --- |
| `list_physics_skills` | Lists learned skills and checks teacher/checkpoint availability. |
| `run_physics_motion` | Runs student or teacher stand/walk with actual dynamics and optional pushes. |
| `list_motions` | Lists validated recordings available in this checkout. |
| `play_motion` | Starts or replaces a recording, with optional speed and looping. |
| `play_larger_steps` | Generates and plays a longer-stride recording using constrained leg IK. |
| `playback_status` | Reports running, completed, stopped, closed, or failed playback. |
| `stop_motion` | Pauses at the current pose while keeping the window open. |
| `close_robot_window` | Closes the viewer; safe to call when already closed. |
| `run_step_length_experiment` | Measures local recording touchdowns against requested lengths and saves results. |
| `get_step_length_results` | Retrieves the latest or a named experiment, including its trial results. |

`src/robot_agent.py` installs/configures Hermes and launches the `g1` toolset.
`src/robot_mcp.py` validates recordings and manages the viewer process.
`src/robot_viewer.py` owns the persistent MuJoCo window and receives playback commands.
On macOS, the MCP server automatically starts the viewer with `mjpython`.

For another MCP client, configure a stdio server using absolute paths for your checkout:

```json
{
  "mcpServers": {
    "g1": {
      "command": "/absolute/path/Chalmers_techweek/.venv/bin/python",
      "args": ["/absolute/path/Chalmers_techweek/src/robot_mcp.py"]
    }
  }
}
```

The `mcpServers` wrapper depends on the client. Hermes uses `mcp_servers`, generated by
`just agent-configure`. Each MCP session owns its own viewer.

Add compatible G1 `.npz` recordings to `data/motions/` and ask the agent to list motions
again. Clips need `fps`, `joint_pos` (29 joints per frame), `body_pos_w` and `body_quat_w`.
Malformed clips are reported instead of launched. Use the returned motion ID (filename
without `.npz`); the tools do not accept arbitrary file paths or shell commands.

### Make the visible steps larger

Restart Hermes after updating the code, then ask:

> Make the robot take 25% larger steps and loop the motion.

The `play_larger_steps` tool generates `walk_longer_125.npz` from the original walk,
uses leg inverse kinematics to increase forward/back foot reach, scales horizontal root
travel, and plays the result in the persistent window. Supported scale: 1.05–1.5.
Joint limits can prevent reaching every target; the returned report includes the achieved
foot-separation ratio and maximum foot-target error. Separation is not the same metric as
ground-contact step length. This is modified recording playback, not balanced motor control.
The original `walk.npz` is preserved. Generated clips and metadata stay in `data/motions/`.

Without the agent:

```bash
uv run python src/longer_steps.py --scale 1.25
just play data/motions/walk_longer_125.npz
```

### Step-length experiment: ask the agent

After updating the code, exit Hermes with **Ctrl+D**, then run `just agent` again to
load the two experiment tools. You do not need to reinstall Hermes or replace your key.
Try this prompt:

> Run a step-length experiment using walk, compare 0.15, 0.20 and 0.25 metres,
> repeat each three times with ten steps per trial, and show me the results table.

Then ask:

> Show my latest step-length results, explain any failed trials, and tell me where the CSV files are.

The experiment now displays the measured recording in the existing robot window (or opens
one), replacing any previous motion. Hermes also reports requested length, mean achieved
length, mean absolute error, valid-step count, completed trials and warnings. The recording
plays once at real-time speed and holds its final pose; measurement runs faster separately,
so the animation is not synchronized to individual trial rows. Ask for data only to set
`show_simulation=False` and leave the viewer unchanged. Use "loop the walking recording"
if you want a continuous demonstration afterward.

**This is the measurement baseline, not a step-length walking controller.** Targets are
comparison values only; they do not change the recorded movement. Repetitions replay the
same data deterministically and do not establish robustness. A controller that accepts a
requested step length and generates motor forces is still needed for the physical task.

The detector uses MuJoCo foot-to-floor collision contacts (`distance <= 0`). It requires
two consecutive airborne frames and then two consecutive contact frames, records the
first touchdown frame, and does not count an initial standing contact as a step. It measures
signed left/right ankle-roll body-origin separation projected onto a fixed walking direction
(default world +X; use `heading_degrees` for a different direction). A valid forward step
requires positive separation and the opposite foot touching the floor at touchdown.
Raw unsupported or backward touchdowns remain in the CSV but are excluded from valid-step
averages. The script processes one clip per trial without looping across its discontinuity.

A trial finishes after the requested number of valid steps or ends as incomplete at the
end of the recording. A pelvis below 0.4 m or tilted more than 60 degrees ends it with a
recorded-fall-pose warning. These are pose checks, not dynamically simulated fall outcomes.
The per-trial report also includes accumulated foot-origin travel while continuously in
contact; this is a stance-drift diagnostic, not a contact-patch slip measurement.

The current jazz-walk clip may yield **no valid forward steps** because its foot contacts
in this model are sparse. Such results are reported as incomplete with unavailable means,
not zero error or success. Review retargeting/joint mapping and floor clearance before
using the clip as a locomotion baseline; the experiment does not silently adjust the floor
or invent contact events.

Each run saves an experiment ID and these files under `.robot-runtime/experiments/<id>/`:

- `steps.csv`: every confirmed touchdown, requested/achieved distance, signed error,
  touchdown time, foot, support-contact flag and validity flag.
- `trials.csv`: each trial's status, valid-step count, error statistics and contact diagnostics.
- `results.json`: complete configuration, limitations, warnings, comparison table and trials.

The directory is ignored by Git. Copy selected results elsewhere if you want to include
them in your submission. Previous runs are retained; ask for an experiment ID to revisit one.

You can also run the same experiment without the agent or an API key:

```bash
just step-experiment  # includes the viewer; close its window or Ctrl+C to exit
# Custom data-only comparison; distances are metres, heading is degrees from world +X.
uv run python src/step_experiment.py --motion walk --targets 0.15 0.20 0.25 --repetitions 3 --steps 10
# Add --show-simulation to the custom command to display the recording too.
```

### Troubleshooting and verification

| Problem | Fix |
| --- | --- |
| `gitjust: command not found` | Run `just agent-configure`; there is no `git` prefix. |
| `just: command not found` | Install `just`, then open a new terminal if needed. |
| `justfile` not found | Change into `Chalmers_techweek` before running the commands. |
| HTTP 401 / invalid API key | Create or copy a valid key from the same provider, run `just agent-configure`, then restart Hermes. |
| Incorrect API URL / HTTP 404 | Use the provider API base URL ending in `/v1` where required, not its dashboard or full `/chat/completions` URL. |
| Model unavailable | Copy a current full model ID from the provider and reconfigure. |
| Four tools listed / no close-window tool | Exit Hermes with Ctrl+D and run `just agent` again. A fresh session should discover all eight tools. |
| `Unknown toolsets: mcp-g1` | Update to this code and rerun `just agent-configure`; the launcher selects the server name `g1`. |
| No motions available | Run `just setup-motions`, then ask for the motion list again. |
| Viewer fails to start | Ask for playback status and inspect `.robot-runtime/playback.log`. A desktop display is required. |
| Repository moved to another folder | Rerun `just agent-install` and `just agent-configure` to refresh installation/configuration paths. |

Run the automated checks without an AI key:

```bash
uv run python -m unittest discover -s tests -v
just check
```

The tests cover clip validation, MCP tool discovery, persistent playback, closing, and
reopening the worker. They do not call the AI provider. `ROBOT_MCP_HEADLESS=1` can be set
on the MCP process for accelerated playback verification without a window; it is not a
real-time visual demo.

---

## 1. Quickstart & Command Reference

This project uses [`just`](https://github.com/casey/just) and [`uv`](https://github.com/astral-sh/uv) to manage and run tasks.

| Command | Description |
| :--- | :--- |
| `just init` | **One-command full setup**: sets up G1 model, URDF, motion clips, uv dependencies, and checks types. |
| `just stand` | Run the G1 standing controller test (holding calibrated stance with pelvis anchor). |
| `just test` | Run the Pink QP arm reaching test (right arm reaches to table target). |
| `just play-walk` | Replay the retargeted G1 **forward walking** motion clip (`walk.npz`). |
| `just play-step` | Replay the retargeted G1 **side stepping** motion clip (`step_touch.npz`). |
| `just play-bow` | Replay the retargeted G1 **bowing / crouching** motion clip (`bow.npz`). |
| `just download-motions` | Pull sample G1 motion clips (`.npz`) from Hugging Face `exptech/g1-moves`. |
| `just sim` | Launch the default **G1 29-DoF** simulation scene in the MuJoCo viewer. |
| `just sim-23dof` | Launch the **G1 23-DoF** variant scene. |
| `just check` | Run static type checking with Astral **ty** (`uv run ty check`). |
| `just setup-urdf` | Download the official G1 29-DoF URDF from Unitree description repo. |

On macOS, the viewer commands automatically use MuJoCo’s `mjpython` launcher.
For direct interactive execution on macOS, replace `uv run python` with
`uv run mjpython`; headless execution can still use `uv run python`.

### Running the Standing Controller

**Interactive Viewer (Anchored Test Gantry)**:
```bash
just stand
```

**Headless / CI Mode**:
```bash
uv run python src/test_pink_ik.py --stand --headless
```

**Direct Execution via `uv`**:
```bash
# Hold standing posture with pelvis anchor (default)
uv run python src/test_pink_ik.py

# Unanchored floating base
uv run python src/test_pink_ik.py --no-anchor
```

---

## 2. Kinematics & Standing Controller

### Pink QP Inverse Kinematics
* Powered by [Pink](https://github.com/stephane-caron/pink) and [Pinocchio](https://github.com/stack-of-tasks/pinocchio) for whole-body and Cartesian IK tasks.
* **Strict Dependency**: Pinocchio, Pink, and a compatible QP solver (`proxqp` or `quadprog`) are **mandatory**. If any are missing or if IK fails, the controller immediately raises an unhandled exception and hard-crashes.

### Calibrated Zero-Penetration Standing
* **Pelvis Resting Height**: Dynamically computed via `controller.rest_pelvis_z = 0.7842m` using forward kinematics. This places the foot contact spheres precisely on the ground plane at $z = 0.000\text{m}$ with zero collision penetration.
* **Stable Contact**: Resolves the classic simulation issue where anchoring too low causes MuJoCo's contact solver to generate large repulsive normal forces that kick the feet backward.
* **Nominal Stance Angles**:
  * Hip pitch: `-0.1 rad`
  * Knee: `+0.3 rad`
  * Ankle pitch: `-0.2 rad`

### Pelvis Anchor vs. Free Floating Base
* `--anchor` (default): Simulates an industrial testing gantry / harness, pinning the pelvis at $(0, 0, 0.7842\text{m})$. This isolates upper-body manipulation and kinematics from balance dynamics.
* `--no-anchor`: Simulates a free floating base (6 unactuated degrees of freedom). Under pure joint PD control, the robot acts as an inverted pendulum and requires an active balance policy (such as RL via `wbc-mjlab`) to avoid tipping over.

---

## 3. Type Checking

This codebase enforces strict static typing using Astral's [ty](https://github.com/astral-sh/ty):

```bash
just check
```
All code in `src/` must pass with **0 diagnostics**.

---

## 4. Troubleshooting & Linux Notes

### `AttributeError: module 'pinocchio' has no attribute 'Model'`

The robotics Pinocchio library is installed through the **`pin`** dependency.
The unrelated PyPI package named `pinocchio` shadows its Python imports.
Keep only `pin` in the project dependencies and run `uv sync` to remove the
conflicting package. Do not add `pinocchio` with pip or uv.


### Warning: `Failed to load plugin 'libdecor-gtk.so': failed to init`
* **What it is**: `libdecor` is a client-side window decoration library used by GLFW/MuJoCo on Wayland desktops (Ubuntu, Fedora, Arch) to render window borders and title bars.
* **Do you need to install it?**: **No.** It is purely cosmetic. MuJoCo automatically falls back to internal decorations, and physics/rendering are completely unaffected.
* **How to silence the warning (optional)**:
  ```bash
  sudo apt install libdecor-0-plugin-1-cairo
  ```

## Behavioural cloning on CPU (no RL optimization)

This experiment trains a small neural network to imitate state/action examples
from the public [G1 Moves](https://huggingface.co/datasets/exptech/g1-moves)
JazzWalk controller. Data is CC BY 4.0; credit Experiential Technologies and the
G1 Moves contributors. The teacher was trained with RL, so describe this method
as **supervised behavioural cloning / policy distillation from an RL teacher**,
not learning independently from human demonstrations. Our student uses supervised
mean-squared action error and Adam; no rewards, PPO updates, or RL training.

```bash
uv sync --group training
just imitation-train                 # 60 demonstration episodes, 100 epochs
just imitation-evaluate              # teacher/student physical rollout comparison
just imitation-view                  # watch the saved student move through motor forces
```

On macOS the viewer command uses `mjpython`. The viewer shows the first evaluation trial, ends after five seconds
or a fall, and holds its last state until you close its window. Training runs
without rendering. No Hermes/API key or GPU is needed. ONNX Runtime runs the
public teacher; PyTorch trains the student on CPU. All data stays local.

The script downloads a checksum-verified teacher export (~1.6 MB) which includes
its motion reference. Its metadata supplies joint order, offsets, gains, action
scaling and sensor order. During collection, the teacher drives the freely moving
G1 in MuJoCo with bounded motor torques. Only resets set the robot pose directly.
The student receives the same 160 observations, including the reference motion,
robot state and previous action, and predicts 29 actions. A PD controller converts
these into torques. Training and validation are split by entire episode (80/20),
but share the same source clip; this is not a held-out-motion test.

Files (relative to this repository):

- `data/imitation/demonstrations.npz`: collected observations, actions and episode IDs.
- `data/imitation/demonstrations.json`: source URL/checksum and teacher rollout outcomes.
- `.robot-runtime/imitation/policy.pt`: best validation checkpoint.
- `.robot-runtime/imitation/training.csv`: training and validation error per epoch.
- `.robot-runtime/imitation/training.json`: training time and sample counts.
- `.robot-runtime/imitation/evaluation.json`: new-seed teacher/student falls, duration,
  displacement and supported forward touchdowns measured along fixed world +X.

These outputs are ignored by Git. Commands overwrite the current experiment's
outputs; copy the directory before conducting a comparison you want to preserve.
To retrain only on the existing demonstrations:

```bash
uv run --group training python src/imitation_g1.py train --epochs 200
```

Limitations: this first experiment follows one turning jazz-walk reference. It
**does not accept requested step length or establish larger-step walking**. A low
validation error is not proof of stable walking: evaluate the closed-loop student.
Failed teacher rollouts are recorded and retained rather than silently called
successful demonstrations. Evaluation has only three five-second trials, starts
from demonstration states, and does not test standing up from the floor. Further
work needs diverse successful walking demonstrations labelled with actual step
length, a matching command input, and held-out-command evaluation. The current
Hermes motion playback tools do not run this student; use `imitation-view`.

### Graphs and downloadable results

Matplotlib is included in the training dependencies. After `just imitation-train`
or `just imitation-evaluate`, a results page opens automatically in your browser.
To reopen the current results without retraining:

```bash
just imitation-results
```

The offline page shows training/validation error, teacher/student time before
fall or trial completion, horizontal displacement, and supported forward
footsteps. It includes each trial's data and buttons to save PNG/SVG graphs,
training/evaluation CSVs, and evaluation JSON. Displacement is **not** step length;
falling trials are labelled explicitly. The report lives at
`.robot-runtime/imitation/report.html` and is self-contained for sharing.
The PNG and SVG are also saved beside it as `graphs.png` and `graphs.svg`.
Direct Python evaluation generates the report without opening a browser, making
it suitable for headless runs. Graphs summarize saved results, not live training.

### Corrective imitation learning with DAgger

```bash
just imitation-dagger               # 5 rounds, 30 rollouts/round, 100 epochs/round
just imitation-view                 # watch the updated student
just imitation-results              # updated physical evaluation graphs
```

Each round executes the **student's actions** in MuJoCo and asks the teacher for
an action label at every state visited. Those corrective examples are appended to
the original data. Training resumes the previous checkpoint using supervised
learning, preserving its input normalization and optimizer state when available.
No rewards or RL optimizer are used. The pretrained teacher itself came from RL.
Teacher labels near a fall may not represent successful recoveries.

The original active checkpoint/results are backed up under
`.robot-runtime/imitation/dagger/<timestamp>/baseline/`. Each `round-XX/` contains
its checkpoint, collection outcomes, correction dataset, evaluation, and graphs.
`progress.json` records falls and mean trial duration per round. The final round
becomes the active model; rounds are not selected based on evaluation performance.
`data/imitation/dagger_demonstrations.npz` accumulates data across DAgger runs.
Training/validation remain separated by episode IDs. Evaluation seeds are excluded
from collection; all sets still share the same reference motion.

A shorter continuation is `just imitation-dagger 3 30 100`. For ordinary supervised
training without new corrective rollouts, `just imitation-resume 100` resumes on
the **original** demonstration dataset. The first resume from an older checkpoint
restores weights and normalization but starts a fresh optimizer if none was saved.
New checkpoints include the optimizer. Epoch numbers in each run's plots count
that run only. Avoid running multiple training commands simultaneously.

This improves tracking and balance during the existing walking clip. It does not
train a separate standing command, standing up from the floor, new step lengths,
or object pickup. Those require appropriate tasks and demonstrations.

### Teacher-guided standing and longer walking

```bash
just balance-train          # teacher qualification, demonstrations, 3 DAgger rounds
just balance-view stand     # physically hold the validated stationary reference
just balance-view walk      # ten-second walking trial
just balance-results        # stage-by-stage graphs and detailed outcomes
```

This uses the same downloaded teacher. A quiet recorded pose (frame 600) becomes
its stationary reference with zero reference velocities. The torso is **not
anchored**: movement and balance use motor torques and physics. A nominal standing
pose was tested and rejected because this walking teacher drifted and fell.
The chosen stationary reference is separately qualified before training.

Qualification/evaluation uses six ten-second trials per task, with small initial
horizontal-velocity perturbations (up to 0.04 m/s per axis). Standing success
requires no fall and maximum horizontal drift below 0.15 m. Walking requires no
fall and torso tracking RMSE below 0.30 m. Neither metric is a commanded step-length
result. Only successful teacher demonstrations are added; DAgger subsequently
adds teacher action labels at student-visited states, including unsuccessful
student rollouts. Input reference distinguishes standing from walking.

The first run starts from your walking student and accumulated DAgger data.
Later runs resume the latest balance student and its accumulated dataset. The
original walking checkpoint remains available. Each balance run has its own
folder under `.robot-runtime/imitation/balance/`, with final policy, dataset,
collection outcomes, per-round checkpoints, `progress.json`, `balance_graphs.png`,
and `report.html`. `latest.json` selects the current balance run. Training and
validation split by episode; evaluation seeds are separate but reused across
stages, and all tests share the same source motion.

This is one standing pose and one walking clip. It does not test task transitions,
large pushes, arbitrary paths, commanded strides, standing up, or grasping. Passing
short tests does not establish robustness beyond these conditions. The Hermes
playback tools remain separate; use the balance viewer commands above.

### Recovery from gentle disturbances

```bash
just recovery-train
just recovery-view stand
just recovery-view walk
just recovery-results
```

Recovery training is supervised DAgger and starts from the latest balance model
(or the last accepted recovery model). Defaults: two rounds, 20 scenarios/round,
40 training epochs/round. Each scenario lasts ten seconds and tests standing or
walking with one of five conditions: clean, pushes, pose changes, friction changes,
or all disturbances combined.

- Pushes: 8 N horizontal force at the pelvis, for 0.2 seconds at seconds 2 and 6;
  direction varies by seed. These are physical external forces, not pose teleports.
- Pose: up to 0.008 radians additional joint offsets and 0.015 radians base tilt
  at reset, on top of existing small initialization noise.
- Friction: sliding friction multiplier 0.9–1.1 on both floor and feet, because
  MuJoCo's contact mixing could hide a floor-only change. Every reset restores
  original friction before applying the next scenario's multiplier.

The teacher must pass all three qualification trials in a task/condition group
before it is used for collection. It must then pass the exact randomized training
scenario before its demonstration and student-visited corrective labels are added.
A passing teacher rollout does not guarantee recovery from every later student
state. Teacher/student examples for the same scenario share an episode split.

A separate validation set decides whether to accept the candidate: no task/condition
may lose passing trials or increase its mean error by more than 1 cm. A further
30-trial test set (three seeds per task/condition) is evaluated once after that
decision and is not used for training or model selection. Validation and test seeds
are reused on later invocations; they are not fresh unseen tests across an unlimited
series of experiments. This is still one standing reference and one jazz-walk clip.

Success requires completing ten seconds without a fall, maximum standing drift
below 0.15 m (or walking tracking RMSE below 0.30 m), plus final-second tracking
error below 0.15 m standing / 0.30 m walking. Final-second error checks settling
several seconds after the second push; it is not an estimate of recovery time.

Everything is saved under `.robot-runtime/imitation/recovery/<timestamp>/`:
`initial_policy.pt`, candidate `policy.pt`, per-round checkpoints, aggregated
`demonstrations.npz`, qualification logs, `results.json`, `test_results.csv`, and
`report.html` with graphs. `latest.json` locates the newest report. `active.json`
selects the policy used by `recovery-view`; if validation regresses, the previous
policy remains active. The balance model is never overwritten. The graphs show
candidate results even if it is rejected, with promotion status stated explicitly.
The viewer holds its final state until closed. It uses a separate preview seed,
so the displayed outcome need not match an individual evaluation trial.

No real robot commands are sent. These gentle tests do not establish robustness to
large pushes, slippery terrain, arbitrary destinations, or task transitions.

To compare a rejected candidate visually, use `just recovery-candidate stand` or
`just recovery-candidate walk`. Normal `recovery-view` always uses the accepted
policy. The report shows validation and final test outcomes separately, including
regressions; a high test score does not override a failed validation gate.

Preview stronger pushes with `just recovery-view stand 12` (12 N instead of the
usual 8 N), or `just recovery-view walk 12`. Preview force accepts 0–100 N; push
onsets and duration stay at 2/6 seconds and 0.2 seconds. This does not change the
8 N training/evaluation protocol or its saved results.

### Learned physics through the agent

Restart Hermes with `just agent` to discover the new tools. For example:

> Use my learned controller to stand with physics for ten seconds, apply 12 N pushes, then tell me whether it fell.

`run_physics_motion(task="stand" | "walk", controller="student" | "teacher",
seconds=10, push_force=0)` uses the same dynamics and action mapping as imitation
training. Each step predicts joint targets, computes bounded PD motor torques, and
advances MuJoCo. The base is free, with no anchoring or per-frame pose assignments.
Only trial reset sets the initial pose. Each command resets a separate trial;
standing-to-walking transitions are not learned. Trials are limited to ten seconds
to stay within the available walking reference. Pushes occur at 2 and 6 seconds
when those times are included in the trial.

`playback_status` reports simulated time, control steps, falls and drift. Completion
or pause freezes physics and keeps the window open. `close_robot_window` closes it.
The student is the accepted recovery policy, otherwise the latest balance policy,
otherwise the original imitation policy. The teacher can be selected for comparison.
Running a motion performs inference only: use `just balance-train` or
`just recovery-train` to update weights. Other recorded motions are not automatically
converted to learned skills; they still require suitable teacher demonstrations.
