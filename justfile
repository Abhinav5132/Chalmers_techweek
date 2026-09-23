# justfile for SKF Hackathon - Unitree G1 Orchestrator

# List available commands by default
default:
    @just --list

default_scene := "unitree_mujoco/unitree_robots/g1/scene_29dof.xml"

# Start MuJoCo simulation viewer with a specified scene (default: G1 29-DoF scene)
sim scene=default_scene:
    @if command -v uv >/dev/null 2>&1; then \
        uv run python -m mujoco.viewer --mjcf="{{scene}}"; \
    else \
        python -m mujoco.viewer --mjcf="{{scene}}"; \
    fi

# Start simulation with 23-DoF G1 scene
sim-23dof:
    @just sim unitree_mujoco/unitree_robots/g1/scene_23dof.xml

# Run the project hello test
hello:
    @if command -v uv >/dev/null 2>&1; then \
        uv run python src/hello.py; \
    else \
        python src/hello.py; \
    fi

# Clone only the Unitree G1 robot model using git sparse-checkout
setup-model:
    @if [ -d "unitree_mujoco/unitree_robots/g1" ]; then \
        echo "G1 model already exists in unitree_mujoco/unitree_robots/g1"; \
    else \
        echo "Cloning only G1 model from unitree_mujoco..."; \
        git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_mujoco.git; \
        cd unitree_mujoco && git sparse-checkout set unitree_robots/g1 && rm -rf .git; \
        echo "G1 model setup complete!"; \
    fi

