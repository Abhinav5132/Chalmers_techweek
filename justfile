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

# Initialize entire environment (models, URDF, motion clips, dependencies, and verification)
init:
    @echo "=== Initializing SKF Hackathon Environment ==="
    @just setup-model
    @just setup-urdf
    @just setup-motions
    @if command -v uv >/dev/null 2>&1; then \
        echo "Syncing dependencies with uv..."; \
        uv sync; \
        echo "Running static type check verification..."; \
        uv run ty check; \
    fi
    @echo "=== Environment Initialization Complete! ==="

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
    @if [ -f "unitree_mujoco/unitree_robots/g1/g1_29dof.urdf" ]; then \
        echo "G1 URDF already exists at unitree_mujoco/unitree_robots/g1/g1_29dof.urdf"; \
    else \
        echo "Downloading official G1 29-DoF URDF from Unitree..."; \
        curl -sSL https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/g1_description/g1_29dof.urdf -o unitree_mujoco/unitree_robots/g1/g1_29dof.urdf; \
        echo "URDF saved to unitree_mujoco/unitree_robots/g1/g1_29dof.urdf"; \
    fi

# Download sample G1 motion clips (.npz) from Hugging Face g1-moves
setup-motions:
    @mkdir -p data/motions
    @base_url="https://huggingface.co/datasets/exptech/g1-moves/resolve/main"; \
    for item in "walk.npz:dance/J_ShortDance16_JazzWalk/training/J_ShortDance16_JazzWalk.npz" \
                "step_touch.npz:dance/J_Dance0_StepTouch/training/J_Dance0_StepTouch.npz" \
                "bow.npz:karate/B_BowKarate/training/B_BowKarate.npz"; do \
        target="${item%%:*}"; \
        src="${item#*:}"; \
        if [ ! -f "data/motions/$target" ]; then \
            echo "Downloading data/motions/$target from Hugging Face..."; \
            curl -L -sSL "$base_url/$src?download=true" -o "data/motions/$target"; \
            echo "Saved data/motions/$target"; \
        else \
            echo "data/motions/$target already exists."; \
        fi \
    done

download-motions: setup-motions

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

# Replay a MoCap clip (.npz) on the G1 in MuJoCo
play clip="data/motions/walk.npz":
    @if command -v uv >/dev/null 2>&1; then \
        uv run python src/play_motion.py --clip "{{clip}}" --loop; \
    else \
        python src/play_motion.py --clip "{{clip}}" --loop; \
    fi

play-walk:
    @just play data/motions/walk.npz

play-step:
    @just play data/motions/step_touch.npz

play-bow:
    @just play data/motions/bow.npz


# Run static type checking with ty
check:
    @if command -v uv >/dev/null 2>&1; then \
        uv run ty check; \
    elif command -v ty >/dev/null 2>&1; then \
        ty check; \
    else \
        echo "ty is not installed. Run: uv run ty check"; \
    fi





