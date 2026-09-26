# justfile for SKF Hackathon - Unitree G1 Orchestrator

# List available commands by default
default:
    @just --list

# MuJoCo passive viewers require mjpython on macOS.
viewer_python := if os() == "macos" { "mjpython" } else { "python" }
# Install the pinned Hermes agent into its own environment.
agent-install:
    uv run python src/robot_agent.py install

# Enter your provider URL, model, and API key locally (key input is hidden).
agent-configure:
    uv run python src/robot_agent.py configure

# Chat with Hermes using only the robot MCP toolset.
agent:
    uv run --group training python src/robot_agent.py chat

# MCP stdio endpoint for Hermes or another MCP client.
robot-mcp:
    uv run --group training python src/robot_mcp.py

# Measure the walking recording against requested step lengths and save CSV/JSON.
step-experiment:
    uv run python src/step_experiment.py --show-simulation

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
        uv run {{viewer_python}} src/test_pink_ik.py; \
    else \
        {{viewer_python}} src/test_pink_ik.py; \
    fi

# Hold calibrated standing stance only (no arm movement)
stand:
    @if command -v uv >/dev/null 2>&1; then \
        uv run {{viewer_python}} src/test_pink_ik.py --stand; \
    else \
        {{viewer_python}} src/test_pink_ik.py --stand; \
    fi

# Replay a MoCap clip (.npz) on the G1 in MuJoCo
play clip="data/motions/walk.npz":
    @if command -v uv >/dev/null 2>&1; then \
        uv run {{viewer_python}} src/play_motion.py --clip "{{clip}}" --loop; \
    else \
        {{viewer_python}} src/play_motion.py --clip "{{clip}}" --loop; \
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


# Collect physical teacher demonstrations, train a student on CPU, and evaluate it.
imitation-train epochs="100" episodes="60":
    uv run --group training python src/imitation_g1.py all --epochs {{epochs}} --episodes {{episodes}}
    uv run --group training python src/imitation_report.py --open

# Evaluate the saved student and teacher in real MuJoCo dynamics.
imitation-evaluate:
    uv run --group training python src/imitation_g1.py evaluate
    uv run --group training python src/imitation_report.py --open

# Watch the first trained-student trial; the window stays open until you close it.
imitation-view:
    uv run --group training {{viewer_python}} src/imitation_g1.py view

# Open graphs and downloadable data from the existing results without retraining.
imitation-results:
    uv run --group training python src/imitation_report.py --open

# Improve the saved student using teacher-labelled corrections from its own rollouts.
imitation-dagger rounds="5" episodes="30" epochs="100":
    uv run --group training python src/dagger_g1.py --rounds {{rounds}} --episodes {{episodes}} --epochs {{epochs}}
    uv run --group training python src/imitation_report.py --open

# Resume supervised training on the original demonstration dataset.
imitation-resume epochs="100":
    uv run --group training python src/imitation_g1.py train --resume --epochs {{epochs}}
    just imitation-evaluate

# Train a separate student for stationary standing and ten-second walking.
balance-train:
    uv run --group training python src/balance_training.py train
    uv run --group training python src/balance_training.py results

# Watch the latest balance student: mode is stand or walk.
balance-view mode="stand":
    uv run --group training {{viewer_python}} src/balance_training.py view --mode {{mode}}

balance-results:
    uv run --group training python src/balance_training.py results

# Teacher-qualified training with gentle pushes, pose changes and friction variation.
recovery-train:
    uv run --group training python src/recovery_training.py train
    uv run --group training python src/recovery_training.py results

# Watch the accepted recovery policy under combined disturbances.
recovery-view mode="stand" force="8":
    uv run --group training {{viewer_python}} src/recovery_training.py view --mode {{mode}} --push-force {{force}}

recovery-results:
    uv run --group training python src/recovery_training.py results

# Compare the latest candidate, including a candidate rejected by validation.
recovery-candidate mode="stand":
    uv run --group training {{viewer_python}} src/recovery_training.py view --candidate --mode {{mode}}

# Assess the existing teacher on three physical 5 cm stairs, plus flat controls.
stairs-evaluate:
    uv run --group training python src/stair_teacher.py evaluate

# Watch the stair qualification attempt; this is not a trained stair skill.
stairs-view:
    uv run --group training {{viewer_python}} src/stair_teacher.py view

# Download 99 verified terrain-adapted PMT clips, terrain, and the pretrained teacher.
pmt-setup:
    uv run --group training python src/pmt_training.py setup

# CPU supervised distillation; qualification blocks training when the teacher fails.
pmt-train epochs="60" episodes="10" rounds="2":
    uv run --group training python src/pmt_training.py train --epochs {{epochs}} --episodes {{episodes}} --rounds {{rounds}} --candidates 99

# Watch the imported teacher on its own terrain; rank selects a reference segment.
pmt-view-teacher rank="0":
    uv run --group training {{viewer_python}} src/pmt_training.py view-teacher --rank {{rank}}

pmt-view-student:
    uv run --group training {{viewer_python}} src/pmt_training.py view-student

pmt-results:
    uv run --group training python src/pmt_training.py results

# Original G1, one 5 cm step, Pinocchio + physical inverse-dynamics teacher.
step-teacher height="0.05":
    uv run --group training {{viewer_python}} src/stair_curriculum.py view --height {{height}}

step-teacher-check:
    uv run --group training python src/stair_curriculum.py evaluate

# Qualified teacher demonstrations -> supervised imitation -> DAgger.
# Stops at a failed stage; each new stage must pass teacher and student trials.
step-train epochs="150" episodes="8" rounds="4":
    uv run --group training python src/stair_learning.py train --epochs {{epochs}} --episodes {{episodes}} --rounds {{rounds}}

step-student checkpoint:
    uv run --group training {{viewer_python}} src/stair_learning.py view-student --checkpoint "{{checkpoint}}"

step-results:
    uv run --group training python src/stair_learning.py results

# Watch a qualified supervised feedback student for the requested step height (metres).
step-feedback height="0.05":
    uv run --group training {{viewer_python}} src/stair_feedback.py view --height {{height}}

# Learn local state corrections from the teacher, then qualify without assistance.
step-feedback-train height="0.05":
    uv run --group training python src/stair_feedback.py train --height {{height}}

# Start the browser motion editor and local Hono/Python backend (npm install --prefix web first).
web:
    npm --prefix web run dev

# Type-check and build the browser motion editor.
web-build:
    npm --prefix web run build
