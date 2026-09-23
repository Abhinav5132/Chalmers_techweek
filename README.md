# Unitree G1 Humanoid Skill Orchestrator (SKF Hackathon)

Visual, modular, and natural-language-driven control for the **Unitree G1 29-DoF Humanoid Robot** in MuJoCo.

See the full architectural specification in [SYSTEM_DESIGN.md](file:///mnt/idfk/programming_stuff/SKF_Hackathon/SYSTEM_DESIGN.md).

---

## 1. Setup & Robot Model

### Download Only the G1 Model (Sparse Clone)
To download only the G1 robot assets without cloning the full repository (omitting other robots and C++ build files):

```bash
just setup-model
```

Or run manually:
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_mujoco.git
cd unitree_mujoco
git sparse-checkout set unitree_robots/g1
rm -rf .git
```

### Tracking the Model in Git
When a folder is cloned via `git clone` inside an existing Git repo, Git registers it as a nested repository (submodule), preventing files inside from being tracked.

To track the G1 model files in your repository:
```bash
# 1. Remove any nested git directory inside unitree_mujoco
rm -rf unitree_mujoco/.git

# 2. Clear Git's submodule cache if it was staged as a gitlink
git rm --cached unitree_mujoco 2>/dev/null || true

# 3. Add the files to your repository
git add unitree_mujoco/
git commit -m "Add Unitree G1 robot model and meshes"
```

---

## 2. Running Simulations (`justfile`)

This project uses [`just`](https://github.com/casey/just) to automate common tasks.

### Command Reference

| Command | Description |
| :--- | :--- |
| `just sim` | Launch the default **G1 29-DoF** simulation scene in MuJoCo viewer. |
| `just sim <path/to/scene.xml>` | Launch a custom simulation scene. |
| `just sim-23dof` | Launch the **G1 23-DoF** variant scene. |
| `just setup-model` | Sparsely clone only the G1 model from `unitree_mujoco`. |
| `just hello` | Run the smoke test (`src/hello.py`). |

### Examples

**Default G1 29-DoF Scene**:
```bash
just sim
```

**Custom Scene**:
```bash
just sim unitree_mujoco/unitree_robots/g1/scene.xml
```

**Direct Execution with `uv`** (without `just`):
```bash
uv run python -m mujoco.viewer --mjcf=unitree_mujoco/unitree_robots/g1/scene_29dof.xml
```
