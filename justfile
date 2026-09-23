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

# Download the official Unitree G1 29-DoF URDF for Pinocchio/Pink
setup-urdf:
    @echo "Downloading official G1 29-DoF URDF from Unitree..."
    @curl -sSL https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/g1_description/g1_29dof.urdf -o unitree_mujoco/unitree_robots/g1/g1_29dof.urdf
    @echo "URDF saved to unitree_mujoco/unitree_robots/g1/g1_29dof.urdf"

# Run the Pink QP arm reaching test (right arm reaches to table target)
test:
    @if command -v uv >/dev/null 2>&1; then \
        uv run python src/test_pink_ik.py; \
    else \
        python src/test_pink_ik.py; \
    fi

# Hold calibrated standing stance only (no arm movement)
stand:
    @if command -v uv >/dev/null 2>&1; then \
        uv run python src/test_pink_ik.py --stand; \
    else \
        python src/test_pink_ik.py --stand; \
    fi

# Run test with unanchored floating-base (requires active balance)
test-free:
    @if command -v uv >/dev/null 2>&1; then \
        uv run python src/test_pink_ik.py --no-anchor; \
    else \
        python src/test_pink_ik.py --no-anchor; \
    fi


# Run static type checking with ty
check:
    @if command -v uv >/dev/null 2>&1; then \
        uv run ty check; \
    elif command -v ty >/dev/null 2>&1; then \
        ty check; \
    else \
        echo "ty is not installed. Run: uv run ty check"; \
    fi





