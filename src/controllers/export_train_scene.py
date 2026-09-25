"""Export the exact mjlab training scene (robot + terrain + sensors) to MuJoCo XML.

The plain-MuJoCo deploy (src/run_workflow.py) must run the SAME dynamics model
the policy was trained on. mjlab builds its G1 robot from a spec with per-joint
armature, no joint friction/damping, capsule feet and a plane terrain — all of
which differ from the bundled unitree scene. This script compiles that exact
scene and writes it (plus mesh assets) under ``models/params/robot_train/``.

Run with the wbc-mjlab venv from the repo root::

  third_party/wbc-mjlab/.venv/bin/python src/controllers/export_train_scene.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WBC_DIR = PROJECT_ROOT / "third_party" / "wbc-mjlab"
sys.path.insert(0, str(WBC_DIR))

from mjlab.scene.scene import Scene  # ty: ignore[unresolved-import]  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # ty: ignore[unresolved-import]  # noqa: E402

from wbc_mjlab.tasks import prepare_wbc_run  # ty: ignore[unresolved-import]  # noqa: E402


def main() -> int:
    task_id = prepare_wbc_run(task_id="Wbc-G1")
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = 1

    scene = Scene(cfg.scene, device="cpu")
    scene.compile()

    out = PROJECT_ROOT / "models" / "params" / "robot_train"
    out.mkdir(parents=True, exist_ok=True)
    scene.write(out)

    # Match the training sim solver options (mjlab applies these to the warp
    # option; plain MuJoCo needs them in the XML).
    xml = out / "scene.xml"
    text = xml.read_text()
    option = '<option timestep="0.005" iterations="10" ls_iterations="20" />\n  '
    if "<option" not in text:
        text = text.replace("<compiler", f"{option}<compiler", 1)
        xml.write_text(text)

    print(f"Wrote training-identical scene -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
