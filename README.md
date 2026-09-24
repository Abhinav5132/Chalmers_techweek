# Unitree G1 Humanoid Skill Orchestrator (SKF Hackathon)

Visual, modular, and natural-language-driven control for the **Unitree G1 29-DoF Humanoid Robot** in MuJoCo.

See the full architectural specification in [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

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

