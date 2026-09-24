# Unitree G1 Humanoid Skill Orchestrator (SKF Hackathon)

Visual, modular, and natural-language-driven control for the **Unitree G1 29-DoF Humanoid Robot** in MuJoCo.

See the full architectural specification in [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

## Robot commands through Hermes and MCP

[Hermes Agent](https://github.com/nousresearch/hermes-agent) turns your text requests
into calls to the project's local **Model Context Protocol (MCP)** server. The server
finds recordings in `data/motions/` and plays them in one persistent MuJoCo window.

The current robot actions are **recorded pose playback**. They do not implement learned
balance, physical walking control, stair climbing, step-length adjustment, or real hardware
control. The AI provider interprets your request; the local player performs the movement.

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
| "What robot movements are available?" | Lists the valid local recordings. |
| "Make the robot bow." | Opens the viewer if needed and plays the bow once. |
| "Play the side-step recording at half speed." | Plays step-touch at 0.5× speed. |
| "Loop the walking recording." | Repeats the walking recording until replaced or stopped. |
| "Play the bow instead." | Switches the existing viewer to the bow recording. |
| "Pause and keep the window open." | Freezes the current pose without closing the viewer. |
| "Stop the robot." / "Close the robot window." | Stops playback and closes the viewer; Hermes remains open. |
| "What is the playback status?" | Reports the current playback state or failure. |

A finished motion holds its final pose in the **same window**. New movements reuse that
window, preserving its camera view. After you close it, another play request opens a new
window. Playback speed is restricted to 0.25–2× and changes timing, not step length.

Press **Ctrl+D** at the Hermes prompt to exit the agent and close its viewer.
After updating the agent/server code, exit and run `just agent` again so Hermes discovers
the updated tools. Existing sessions do not automatically gain newly added tools.

### MCP tools and files

| Tool | Purpose |
| --- | --- |
| `list_motions` | Lists validated recordings available in this checkout. |
| `play_motion` | Starts or replaces a recording, with optional speed and looping. |
| `playback_status` | Reports running, completed, stopped, closed, or failed playback. |
| `stop_motion` | Pauses at the current pose while keeping the window open. |
| `close_robot_window` | Closes the viewer; safe to call when already closed. |

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

### Troubleshooting and verification

| Problem | Fix |
| --- | --- |
| `gitjust: command not found` | Run `just agent-configure`; there is no `git` prefix. |
| `just: command not found` | Install `just`, then open a new terminal if needed. |
| `justfile` not found | Change into `Chalmers_techweek` before running the commands. |
| HTTP 401 / invalid API key | Create or copy a valid key from the same provider, run `just agent-configure`, then restart Hermes. |
| Incorrect API URL / HTTP 404 | Use the provider API base URL ending in `/v1` where required, not its dashboard or full `/chat/completions` URL. |
| Model unavailable | Copy a current full model ID from the provider and reconfigure. |
| Four tools listed / no close-window tool | Exit Hermes with Ctrl+D and run `just agent` again. A fresh session should discover all five tools. |
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
