"""Install, configure, and launch a project-local Hermes robot agent."""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / ".hermes-robot"
SOURCE = ROOT / ".tools/hermes-agent"
ENV = ROOT / ".tools/hermes-venv"
REVISION = "6da966f22c94c5b74a54b046bfcecde87f2ec519"
INSTRUCTIONS = """You control a Unitree G1 simulation using the g1 MCP tools.
First call list_motions and choose only an available motion ID matching the request.
Use play_motion for the user's requested recording; use playback_status to check outcomes.
Only loop when requested. New play commands replace the current motion in the same window.
When asked to stop, close the window, or end the robot session, call close_robot_window.
Only use stop_motion to pause when the user explicitly wants the window kept open. Never invent available movements.
Describe these actions as recorded pose playback, not learned or physically balanced walking.
Playback speed is not step length. Stair climbing, grasping, and training are not implemented.
If a request is unsupported, explain the limitation and offer available recordings.
Do not claim a motion completed based solely on a running status. No physical robot is connected.
"""


def install() -> None:
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("Install uv first, then rerun this command.")
    SOURCE.parent.mkdir(exist_ok=True)
    if not SOURCE.exists():
        subprocess.run(["git", "clone", "https://github.com/nousresearch/hermes-agent.git", str(SOURCE)], check=True)
        subprocess.run(["git", "-C", str(SOURCE), "checkout", REVISION], check=True)
    actual = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise SystemExit(f"Existing Hermes checkout differs from the tested revision {REVISION}; left unchanged.")
    if not (ENV / "bin/python").exists():
        subprocess.run([uv, "venv", str(ENV), "--python", "3.11"], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(ENV / "bin/python"), "-e", f"{SOURCE}[mcp]"], check=True)
    print("Hermes installed. Next: just agent-configure")


def configure() -> None:
    print("Configure your OpenAI-compatible provider. The key stays in an ignored local file.")
    base = input("API base URL (including /v1 if required): ").strip().rstrip("/")
    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        raise SystemExit("Enter an http(s) API endpoint without embedded credentials.")
    if "/dashboard" in parsed.path:
        raise SystemExit("Use the API base URL, not the dashboard. For Gonka: https://proxy.gonka.gg/v1")
    model = input("Exact model ID (must support tool calling): ").strip()
    if not model:
        raise SystemExit("A model ID is required.")
    key = getpass.getpass("API key (hidden; leave blank only for a local server): ").strip()
    if not key and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise SystemExit("A key is required for a remote provider.")
    HOME.mkdir(mode=0o700, exist_ok=True)
    os.chmod(HOME, 0o700)
    secret = HOME / ".env"
    fd = os.open(secret, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write("ROBOT_LLM_API_KEY=" + json.dumps(key or "local") + "\n")
    os.chmod(secret, 0o600)
    config = {
        "model": {"provider": "custom", "default": model, "base_url": base,
                  "api_key": "${ROBOT_LLM_API_KEY}"},
        "platform_toolsets": {"cli": ["g1"]},
        "mcp_servers": {"g1": {
            "command": str(ROOT / ".venv/bin/python"),
            "args": [str(ROOT / "src/robot_mcp.py")],
            "supports_parallel_tool_calls": False,
            "tools": {"resources": False, "prompts": False},
        }},
    }
    # JSON is valid YAML; this avoids a separate config-writing dependency.
    (HOME / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    (HOME / "SOUL.md").write_text(INSTRUCTIONS)
    print("Saved local configuration. Start chatting with: just agent")


def chat(extra: list[str]) -> None:
    binary = ENV / "bin/hermes"
    if not binary.exists():
        raise SystemExit("Hermes is missing. Run just agent-install first.")
    if not (HOME / "config.yaml").exists():
        raise SystemExit("Provider is not configured. Run just agent-configure first.")
    environment = dict(os.environ, HERMES_HOME=str(HOME))
    os.chdir(ROOT)
    os.execve(str(binary), [str(binary), "chat", "--toolsets", "g1", *extra], environment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "configure", "chat"))
    args, extra = parser.parse_known_args()
    if args.action == "install":
        install()
    elif args.action == "configure":
        configure()
    else:
        chat(extra)
