"""
50 Hz resampling for motion clips (npz).

The WBC policy advances one clip frame per policy step (50 Hz). Clips recorded
at other rates (e.g. the 60 Hz hackathon clips) are resampled here so playback
stays at the correct speed. Pure NumPy, no GPU required. Shared by the runtime
(ClipReference) and the wbc-setup-clips justfile recipe (CLI below).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def resample_clip(
    clip: dict[str, np.ndarray],
    src_fps: float,
    target_fps: float,
) -> dict[str, Any]:
    """Resample a clip's joint/body arrays to ``target_fps`` (linear + slerp)."""
    n_in = int(clip["joint_pos"].shape[0])
    src = np.arange(n_in, dtype=np.float64) / float(src_fps)
    n_out = max(1, int(round((n_in - 1) * target_fps / float(src_fps))) + 1)
    dst = np.linspace(0.0, src[-1], n_out)
    idx0 = np.clip(np.searchsorted(src, dst, side="right") - 1, 0, n_in - 2)
    idx1 = idx0 + 1
    w = np.clip((dst - src[idx0]) / np.maximum(src[idx1] - src[idx0], 1e-12), 0.0, 1.0)
    wv = w[:, None]

    out: dict[str, Any] = {}
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        if key not in clip:
            continue
        a = clip[key]
        wkey = wv.reshape((n_out,) + (1,) * (a.ndim - 1))
        out[key] = (1.0 - wkey) * a[idx0] + wkey * a[idx1]

    q0 = clip["body_quat_w"][idx0]
    q1 = clip["body_quat_w"][idx1]
    wkey = wv.reshape((n_out,) + (1,) * (q0.ndim - 1))
    flip = (np.sum(q0 * q1, axis=-1, keepdims=True) < 0.0)
    q1f = np.where(flip, -q1, q1)
    dot = np.clip(np.sum(q0 * q1f, axis=-1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(dot)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(theta < 1e-8, 1.0 - wkey, np.sin((1.0 - wkey) * theta) / np.sin(theta))
        t = np.where(theta < 1e-8, wkey, np.sin(wkey * theta) / np.sin(theta))
    out["body_quat_w"] = s * q0 + t * q1f

    out["fps"] = np.array([float(target_fps)])
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Resample motion clip(s) to a target frame rate.")
    parser.add_argument(
        "--src-dir",
        default="data/motions",
        help="Directory containing the source .npz clips (default data/motions).",
    )
    parser.add_argument(
        "--dest-dir",
        required=True,
        help="Directory to write the resampled .npz clips (created if missing).",
    )
    parser.add_argument("clips", nargs="+", help="Clip names (with or without .npz).")
    parser.add_argument("--fps", type=float, default=50.0, help="Target fps (default 50).")
    args: argparse.Namespace = parser.parse_args()

    src_dir = Path(args.src_dir)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in args.clips:
        stem = name[:-4] if name.endswith(".npz") else name
        src = src_dir / f"{stem}.npz"
        dst = dest_dir / f"{stem}.npz"
        if not src.is_file():
            raise FileNotFoundError(f"Source clip not found: {src}")
        data = np.load(src)
        keys = list(data.files)
        clip: dict[str, Any] = {k: data[k] for k in keys}
        src_fps = float(clip["fps"][0])
        if src_fps == args.fps:
            np.savez(dst, **{k: clip[k] for k in keys})
            print(f"Already {args.fps:g} Hz; copied {src.name} -> {dst}")
            continue
        out = resample_clip(clip, src_fps, args.fps)
        np.savez(dst, **out)
        print(
            f"Resampled {src.name} {src_fps:g}Hz -> {args.fps:g}Hz "
            f"({out['joint_pos'].shape[0]} frames) -> {dst}"
        )


if __name__ == "__main__":
    _cli()
