"""Application layer for the Skill Orchestrator.

Exposes the four operations the GUI needs as a single callable API:
preview -> chain -> train -> play. Both the DearPyGui window and the CLI
(``python -m src.app.cli``) drive this same controller, so a GUI is a thin
wrapper around it.

The main project (plain MuJoCo) and the wbc-mjlab RL env (in
``third_party/wbc-mjlab``, separate venv) cannot be imported together, so the
controller shells out to the right interpreter per operation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "motions"
MODELS_DIR = PROJECT_ROOT / "models"
WBC_DIR = PROJECT_ROOT / "third_party" / "wbc-mjlab"
WBC_DATASET_DIR = WBC_DIR / "data" / "g1" / "hackathon"


@dataclass
class TrainConfig:
    """Training parameters for the quick (CPU/GPU-friendly) path."""

    envs: int = 1024
    iterations: int = 3000
    save_interval: int = 250


@dataclass
class Chain:
    """An ordered list of motion clip names."""

    names: list[str] = field(default_factory=list)

    def add(self, name: str) -> None:
        if name not in self.names:
            self.names.append(name)

    def remove(self, name: str) -> None:
        if name in self.names:
            self.names.remove(name)

    def clear(self) -> None:
        self.names.clear()

    @property
    def is_empty(self) -> bool:
        return not self.names


def _main_python() -> str:
    return sys.executable


def _wbc_python() -> str:
    return str(WBC_DIR / ".venv" / "bin" / "python")


def _run(cmd: list[str], cwd: Path, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), check=True, **kwargs)


class SkillApp:
    """High-level orchestration API backing both the GUI and the CLI."""

    def __init__(self) -> None:
        self.chain = Chain()

    # --- motion library ----------------------------------------------------

    def list_motions(self) -> list[str]:
        """Names of the available .npz motion clips (without extension)."""
        if not DATA_DIR.is_dir():
            return []
        return sorted(
            p.stem for p in DATA_DIR.glob("*.npz") if p.is_file()
        )

    # --- preview -----------------------------------------------------------

    def preview(self, name: str) -> int:
        """Kinematic (Phase-1) preview of a single motion in plain MuJoCo."""
        clip = self._clip_path(name)
        _run(
            [_main_python(), "-m", "src.play_motion", "--clip", str(clip), "--loop"],
            cwd=PROJECT_ROOT,
        )
        return 0

    # --- chain -------------------------------------------------------------

    def build_chain(self, names: list[str]) -> Chain:
        for name in names:
            self._require_motion(name)
        self.chain = Chain(names=list(names))
        return self.chain

    # --- training ----------------------------------------------------------

    def setup(self) -> int:
        """One-command wbc env bring-up: sync + convert samples + resample clips."""
        _run(
            [
                "uv",
                "sync",
                "--extra",
                "cu128",
                "--group",
                "dev",
            ],
            cwd=WBC_DIR,
        )
        _run(
            [
                _wbc_python(),
                "-m",
                "wbc_mjlab.motion.data_to_npz",
                "--robot",
                "g1",
                "--dataset",
                "samples",
                "--batch-size",
                "4",
            ],
            cwd=WBC_DIR,
        )
        _run(
            [
                _main_python(),
                "-m",
                "src.controllers.clip_resample",
                "--src-dir",
                str(DATA_DIR),
                "--dest-dir",
                str(WBC_DATASET_DIR / "npz"),
                "--fps",
                "50",
                *self.list_motions(),
            ],
            cwd=PROJECT_ROOT,
        )
        return 0

    def train(self, cfg: TrainConfig | None = None) -> int:
        """Train/fine-tune a WBC policy on the current clip library."""
        cfg = cfg or TrainConfig()
        _run(
            [
                _wbc_python(),
                "-m",
                "wbc_mjlab.scripts.train",
                "--task",
                "Wbc-G1",
                "--dataset",
                "hackathon",
                "--env.scene.num-envs",
                str(cfg.envs),
                "--agent.max-iterations",
                str(cfg.iterations),
                "--agent.save-interval",
                str(cfg.save_interval),
            ],
            cwd=WBC_DIR,
        )
        return 0

    def export(self) -> int:
        """Export the latest trained policy to models/params/policy.onnx + config.yaml."""
        checkpoint = self._latest_checkpoint()
        cmd = [
            _wbc_python(),
            "-m",
            "wbc_mjlab.scripts.chain",
            "--motion-source",
            str(WBC_DATASET_DIR),
            "--chain",
            "walk",
            "--export-models",
            str(MODELS_DIR),
        ]
        if checkpoint is not None:
            cmd += ["--checkpoint-file", str(checkpoint)]
        _run(cmd, cwd=WBC_DIR)
        return 0

    def _latest_checkpoint(self) -> Path | None:
        """Most recent trained checkpoint in the wbc logs (highest iteration)."""
        log_root = WBC_DIR / "logs" / "rsl_rl" / "wbc_g1"
        if not log_root.is_dir():
            return None
        best: Path | None = None
        best_iter = -1
        for run_dir in log_root.iterdir():
            if not run_dir.is_dir():
                continue
            for pt in run_dir.glob("model_*.pt"):
                try:
                    it = int(pt.stem.split("_", 1)[1])
                except ValueError:
                    continue
                # Only use checkpoints from a real training run (not iter 0).
                if it > best_iter:
                    best_iter = it
                    best = pt
        return best

    # --- play --------------------------------------------------------------

    def play(self, chain: list[str] | None = None) -> int:
        """Run the trained policy through the chain in plain MuJoCo.

        ``chain`` is optional; when omitted the current in-memory chain is used.
        """
        names = chain if chain is not None else self.chain.names
        if not names:
            raise RuntimeError("No chain built. Pass --chain or add motions first.")
        for name in names:
            self._require_motion(name)
        _run(
            [
                _main_python(),
                "-m",
                "src.run_workflow",
                "--chain",
                *names,
            ],
            cwd=PROJECT_ROOT,
        )
        return 0

    # --- helpers -----------------------------------------------------------

    def _clip_path(self, name: str) -> Path:
        p = DATA_DIR / f"{name}.npz"
        if not p.is_file():
            raise FileNotFoundError(f"No motion '{name}' at {p}")
        return p

    def _require_motion(self, name: str) -> None:
        if name not in self.list_motions():
            raise FileNotFoundError(
                f"Unknown motion '{name}'. Available: {self.list_motions()}"
            )
