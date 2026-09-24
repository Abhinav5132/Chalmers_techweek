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


# === wbc-mjlab RL tracking stack (separate env in third_party/wbc-mjlab) ===

wbc_dir := "third_party/wbc-mjlab"
wbc_motion_dir := "data/g1/hackathon"

# Sync the wbc-mjlab RL environment (GPU torch + CUDA wheels, ~4GB; resumes from cache)
wbc-sync:
    @cd {{wbc_dir}} && uv sync --extra cu128 --group dev

# Convert bundled sample clips (LAFAN/SEED) to npz (run once after wbc-sync)
wbc-convert-samples:
    @cd {{wbc_dir}} && uv run wbc-mjlab-data-to-npz --robot g1 --dataset samples --batch-size 4

# Resample the project's motion clips to 50 Hz into the wbc-mjlab dataset layout
wbc-setup-clips:
    @uv run python -m src.controllers.clip_resample \
        --src-dir data/motions --dest-dir {{wbc_dir}}/data/g1/hackathon/npz \
        --fps 50 walk step_touch bow
    @echo "Hackathon clip library (50 Hz) ready at {{wbc_dir}}/data/g1/hackathon/npz/"

# Run the chained clip sequence with the trained policy (headless metrics)
chain clips="walk step_touch bow":
    @cd {{wbc_dir}} && uv run python -m wbc_mjlab.scripts.chain \
        --motion-source {{wbc_motion_dir}} --chain {{clips}}

# Validate the chain runner headless on the bundled samples (smoke test)
chain-samples:
    @cd {{wbc_dir}} && uv run python -m wbc_mjlab.scripts.chain \
        --motion-source data/g1/samples --chain walk1_subject1 --loops 2

# Train a WBC policy on the hackathon clip library (teammate's pipeline)
wbc-train:
    @cd {{wbc_dir}} && uv run wbc-mjlab-train --task Wbc-G1 --dataset hackathon

# Play a single clip with the trained policy in the interactive viser viewer
wbc-play dataset="hackathon":
    @cd {{wbc_dir}} && uv run wbc-mjlab-play --task Wbc-G1 --dataset {{dataset}}

# Export deploy artifacts (policy.onnx + config.yaml) into models/
wbc-export:
    @cd {{wbc_dir}} && uv run python -m wbc_mjlab.scripts.chain \
        --motion-source data/g1/hackathon --chain walk \
        --export-models ../../models

# Run static type checking with ty
check:
    @if command -v uv >/dev/null 2>&1; then \
        uv run ty check src; \
    elif command -v ty >/dev/null 2>&1; then \
        ty check src; \
    else \
        echo "ty is not installed. Run: uv run ty check src"; \
    fi





