"""CLI entry point for the Skill Orchestrator application layer.

Usage::

  python -m src.app.cli motions            # list available motions
  python -m src.app.cli preview walk       # kinematic preview (Phase-1)
  python -m src.app.cli chain walk step_touch bow   # build a chain
  python -m src.app.cli train --envs 1024 --iters 3000
  python -m src.app.cli export
  python -m src.app.cli play               # play the trained chain in MuJoCo
  python -m src.app.cli setup              # one-command wbc env bring-up
"""

from __future__ import annotations

import argparse
import sys

from src.app.controller import SkillApp, TrainConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="skill", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("motions", help="List available motion clips.")

    p_preview = sub.add_parser("preview", help="Preview a motion (Phase-1 kinematic).")
    p_preview.add_argument("motion")

    p_chain = sub.add_parser("chain", help="Build an ordered chain of motions.")
    p_chain.add_argument("motions", nargs="+")

    p_setup = sub.add_parser("setup", help="One-command wbc env bring-up.")

    p_train = sub.add_parser("train", help="Train/fine-tune the WBC policy.")
    p_train.add_argument("--envs", type=int, default=1024)
    p_train.add_argument("--iters", type=int, default=3000)

    sub.add_parser("export", help="Export trained policy to models/params/.")

    sub.add_parser("play", help="Play the trained chain in plain MuJoCo.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = SkillApp()

    if args.command == "motions":
        for name in app.list_motions():
            print(name)
        return 0
    if args.command == "preview":
        return app.preview(args.motion)
    if args.command == "chain":
        app.build_chain(args.motions)
        print(f"Chain: {' -> '.join(app.chain.names)}")
        return 0
    if args.command == "setup":
        return app.setup()
    if args.command == "train":
        return app.train(TrainConfig(envs=args.envs, iterations=args.iters))
    if args.command == "export":
        return app.export()
    if args.command == "play":
        return app.play()
    return 0


if __name__ == "__main__":
    sys.exit(main())
