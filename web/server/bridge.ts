import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
export const root = fileURLToPath(new URL("../../", import.meta.url));
export interface Bridge {
  request(action: string, payload?: unknown): Promise<unknown>;
}
export class PythonBridge implements Bridge {
  private process?: ChildProcessWithoutNullStreams;
  private sequence = 0;
  private pending = new Map<
    number,
    {
      resolve: (result: unknown) => void;
      reject: (error: Error) => void;
      timer: NodeJS.Timeout;
    }
  >();
  private logs = "";
  private start() {
    const child = spawn(
      process.env.ROBOT_PYTHON || resolve(root, ".venv/bin/python"),
      ["-u", resolve(root, "src/web_bridge.py")],
      { cwd: root, stdio: "pipe" },
    );
    this.process = child;
    child.stderr.on("data", (data) => {
      this.logs = (this.logs + data.toString()).slice(-2000);
    });
    const lines = createInterface({ input: child.stdout });
    lines.on("line", (line) => {
      try {
        const message = JSON.parse(line);
        const pending = this.pending.get(message.id);
        if (!pending) return;
        clearTimeout(pending.timer);
        this.pending.delete(message.id);
        if (message.error) pending.reject(new Error(message.error));
        else pending.resolve(message.result);
      } catch {
        /* Third-party diagnostics are not protocol messages. */
      }
    });
    child.stdin.on("error", () => {});
    child.on("error", (error) =>
      this.fail(
        new Error(
          `Python worker unavailable: ${error.message}. Run uv sync first.`,
        ),
      ),
    );
    child.on("exit", () => {
      if (this.process === child) {
        this.process = undefined;
        this.fail(
          new Error(
            `Python worker exited. ${this.logs || "Check the Python environment."}`,
          ),
        );
      }
      lines.close();
    });
  }
  private fail(error: Error) {
    for (const item of this.pending.values()) {
      clearTimeout(item.timer);
      item.reject(error);
    }
    this.pending.clear();
  }
  request(action: string, payload: unknown = {}): Promise<unknown> {
    if (!this.process) this.start();
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.fail(
          new Error(
            "Python worker timed out. The simulation has been stopped; retry the action.",
          ),
        );
        this.process?.kill("SIGKILL");
        this.process = undefined;
      }, 60000);
      this.pending.set(id, { resolve, reject, timer });
      this.process!.stdin.write(JSON.stringify({ id, action, payload }) + "\n");
    });
  }
  close() {
    this.fail(new Error("Server shutting down"));
    this.process?.kill("SIGTERM");
    this.process = undefined;
  }
}
