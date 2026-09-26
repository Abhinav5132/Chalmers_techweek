# justfile for SKF Hackathon - Unitree G1 Orchestrator

# List available commands by default
default:
    @just --list

default_scene := "unitree_mujoco/unitree_robots/g1/scene_29dof.xml"

# Start MuJoCo simulation viewer with a specified scene (default: G1 29-DoF scene)
sim scene=default_scene:
    @if command -v uv >/dev/null 2>&1; then \
        env -u __GLX_VENDOR_LIBRARY_NAME uv run python -m mujoco.viewer --mjcf="{{scene}}"; \
    else \
        env -u __GLX_VENDOR_LIBRARY_NAME python -m mujoco.viewer --mjcf="{{scene}}"; \
    fi

# Start simulation with 23-DoF G1 scene
sim-23dof:
    @just sim unitree_mujoco/unitree_robots/g1/scene_23dof.xml

# Initialize entire environment (models, URDF, motion clips, dependencies, and verification)
init:
    @if ! command -v uv >/dev/null 2>&1; then \
        echo "Error: 'uv' is required for environment setup but was not found in PATH." >&2; \
        echo "Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; \
        exit 1; \
    fi
    @echo "=== Initializing SKF Hackathon Environment ==="
    @echo "Syncing dependencies with uv..."
    @uv sync
    @just setup-model
    @just setup-urdf
    @just setup-motions
    @echo "Running static type check verification..."
    @uv run ty check
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
        curl -f -sSL https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/g1_description/g1_29dof.urdf -o unitree_mujoco/unitree_robots/g1/g1_29dof.urdf; \
        echo "URDF saved to unitree_mujoco/unitree_robots/g1/g1_29dof.urdf"; \
    fi

# Download sample G1 motion clips (.npz) from Hugging Face g1-moves with validation
setup-motions:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p data/motions
    if command -v uv >/dev/null 2>&1; then
        PY_CMD="uv run python"
    else
        PY_CMD="python3"
    fi
    base_url="https://huggingface.co/datasets/exptech/g1-moves/resolve/main"
    val_script="import sys, numpy as np; d = np.load(sys.argv[1]); assert all(k in d for k in ['fps', 'joint_pos', 'body_pos_w', 'body_quat_w'])"
    items=(
        "walk.npz:dance/J_ShortDance16_JazzWalk/training/J_ShortDance16_JazzWalk.npz"
        "step_touch.npz:dance/J_Dance0_StepTouch/training/J_Dance0_StepTouch.npz"
        "bow.npz:karate/B_BowKarate/training/B_BowKarate.npz"
    )
    for item in "${items[@]}"; do
        target="${item%%:*}"
        src="${item#*:}"
        dest="data/motions/$target"
        tmp_file="data/motions/.${target}.tmp"
        if [ -f "$dest" ]; then
            if $PY_CMD -c "$val_script" "$dest" >/dev/null 2>&1; then
                echo "$dest already exists and is valid."
                continue
            else
                echo "Warning: $dest is corrupted or invalid. Removing and re-downloading..."
                rm -f "$dest"
            fi
        fi
        echo "Downloading $dest from Hugging Face..."
        rm -f "$tmp_file"
        if ! curl -f -L -sSL "$base_url/$src?download=true" -o "$tmp_file"; then
            echo "Error: Failed to download $target (HTTP or network error)." >&2
            rm -f "$tmp_file"
            exit 1
        fi
        if ! $PY_CMD -c "$val_script" "$tmp_file" >/dev/null 2>&1; then
            echo "Error: Downloaded file $target is invalid or corrupted (failed key validation)." >&2
            rm -f "$tmp_file"
            exit 1
        fi
        mv "$tmp_file" "$dest"
        echo "Validated and saved $dest"
    done

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
        env -u __GLX_VENDOR_LIBRARY_NAME uv run python src/play_motion.py --clip "{{clip}}" --loop; \
    else \
        env -u __GLX_VENDOR_LIBRARY_NAME python src/play_motion.py --clip "{{clip}}" --loop; \
    fi

alias play-clip := play


# === wbc-mjlab RL tracking stack (separate env in third_party/wbc-mjlab) ===

wbc_dir := "third_party/wbc-mjlab"
wbc_motion_dir := "data/g1/hackathon"

# One-command wbc env bring-up: sync env + convert bundled samples + resample clips to 50 Hz
wbc-setup:
    @uv run python -m src.app.cli setup

# Run the chained clip sequence with the trained policy (headless metrics, wbc-native)
chain clips="walk_lafan step_touch bow" steps="20000":
    @cd {{wbc_dir}} && uv run python -m wbc_mjlab.scripts.chain \
        --motion-source {{wbc_motion_dir}} --chain {{clips}} \
        --max-steps {{steps}}

# Fast train: env count + few iterations for a quick, visible result
wbc-train envs="1024" iters="3000":
    @uv run python -m src.app.cli train --envs {{envs}} --iters {{iters}}

# Fine-tune from the bundled checkpoint instead of scratch (much faster convergence)
wbc-finetune envs="512" iters="3000":
    @uv run python -m src.app.cli train --envs {{envs}} --iters {{iters}} --from-bundled

# Export deploy artifacts (policy.onnx + config.yaml) into models/
wbc-export:
    @uv run python -m src.app.cli export

# Regenerate the training-identical G1 scene used by the plain-MuJoCo deploy
wbc-export-scene:
    @{{wbc_dir}}/.venv/bin/python -m src.controllers.export_train_scene

# Play the trained policy through a chain in plain MuJoCo (main-project runtime)
play-chain clips="walk step_touch bow":
    @env -u __GLX_VENDOR_LIBRARY_NAME uv run python -m src.app.cli play --chain {{clips}}

# Launch the DearPyGui desktop window (preview / chain / train / play)
gui:
    @env -u __GLX_VENDOR_LIBRARY_NAME uv run python -m src.gui.main

# Run static type checking with ty
check:
    @if command -v uv >/dev/null 2>&1; then \
        uv run ty check src; \
    elif command -v ty >/dev/null 2>&1; then \
        ty check src; \
    else \
        echo "ty is not installed. Run: uv run ty check src"; \
    fi





