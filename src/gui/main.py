"""DearPyGui desktop window for the Skill Orchestrator.

Workflow: preview motions -> chain them -> train -> play the result.
Thin wrapper around :class:`src.app.controller.SkillApp`; blocking operations
(preview / train / play) run in a background thread so the window stays live.
"""

from __future__ import annotations

import threading
from typing import Any

import dearpygui.dearpygui as dpg

from src.app.controller import SkillApp, TrainConfig

app = SkillApp()


def _run_in_thread(fn) -> None:
    def wrapper() -> None:
        try:
            fn()
        except Exception as exc:  # surface errors into the log text
            dpg.set_value("log", f"ERROR: {exc}")
        finally:
            dpg.set_value("status", "idle")

    dpg.set_value("status", "running...")
    threading.Thread(target=wrapper, daemon=True).start()


def _refresh_motions() -> None:
    dpg.delete_item("motion_table", children_only=True)
    for name in app.list_motions():
        with dpg.group(parent="motion_table", horizontal=True):
            dpg.add_text(name, width=180)
            dpg.add_button(
                label="Preview",
                callback=lambda s, a, n=name: _run_in_thread(lambda: app.preview(n)),
            )
            dpg.add_button(
                label="+",
                callback=lambda s, a, n=name: _add_to_chain(n),
            )


def _add_to_chain(name: str) -> None:
    app.chain.add(name)
    _refresh_chain()


def _refresh_chain() -> None:
    dpg.delete_item("chain_table", children_only=True)
    for i, name in enumerate(app.chain.names):
        with dpg.group(parent="chain_table", horizontal=True):
            dpg.add_text(f"{i}. {name}", width=180)
            dpg.add_button(
                label="Remove",
                callback=lambda s, a, n=name: _remove_from_chain(n),
            )


def _remove_from_chain(name: str) -> None:
    app.chain.remove(name)
    _refresh_chain()


def _on_train() -> None:
    cfg = TrainConfig(
        envs=int(dpg.get_value("envs")),
        iterations=int(dpg.get_value("iters")),
    )
    _run_in_thread(lambda: app.train(cfg))


def _on_play() -> None:
    _run_in_thread(app.play)


def _on_export() -> None:
    _run_in_thread(app.export)


def _on_setup() -> None:
    _run_in_thread(app.setup)


def build_window() -> None:
    dpg.create_context()
    with dpg.window(label="Skill Orchestrator", width=720, height=560):
        dpg.add_text("1. Motions (click Preview to watch, + to add to chain)")
        dpg.add_button(label="Refresh", callback=_refresh_motions)
        dpg.add_child_window(width=-1, height=160, tag="motion_table")

        dpg.add_separator()
        dpg.add_text("2. Chain (order shown is the play sequence)")
        dpg.add_child_window(width=-1, height=120, tag="chain_table")

        dpg.add_separator()
        dpg.add_text("3. Train the brain on this chain")
        with dpg.group(horizontal=True):
            dpg.add_input_int(label="envs", default_value=1024, tag="envs", width=90)
            dpg.add_input_int(label="iters", default_value=3000, tag="iters", width=90)
            dpg.add_button(label="Train", callback=_on_train)
            dpg.add_button(label="Setup (once)", callback=_on_setup)

        dpg.add_separator()
        dpg.add_text("4. Run the trained sequence")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Export policy", callback=_on_export)
            dpg.add_button(label="Play chain", callback=_on_play)

        dpg.add_separator()
        dpg.add_text("", tag="status")
        dpg.add_input_text(multiline=True, readonly=True, width=-1, height=80, tag="log")

    _refresh_motions()
    _refresh_chain()


def main() -> None:
    build_window()
    dpg.create_viewport(title="Skill Orchestrator", width=740, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
