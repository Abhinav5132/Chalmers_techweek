import { compress } from "hono/compress";
import { Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import { z } from "zod";
import {
  mkdir,
  readFile,
  readdir,
  rename,
  writeFile,
  unlink,
} from "node:fs/promises";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { routineSchema, type SavedRoutine } from "../src/shared";
import type { Bridge } from "./bridge";

export function createApp(bridge: Bridge, storage: string) {
  const app = new Hono();
  app.use("/api/model", compress());
  app.use(
    "/api/*",
    bodyLimit({
      maxSize: 100_000,
      onError: (c) => c.json({ error: "Request is too large." }, 413),
    }),
  );
  app.use("/api/*", async (c, next) => {
    if (!["GET", "HEAD"].includes(c.req.method)) {
      const origin = c.req.header("origin");
      if (
        origin &&
        !["localhost", "127.0.0.1"].some((host) =>
          ["5173", process.env.STUDIO_API_PORT || "8787"].some(
            (port) => origin === `http://${host}:${port}`,
          ),
        )
      )
        return c.json(
          { error: "Only the local studio can control this simulation." },
          403,
        );
      if (
        c.req.method !== "DELETE" &&
        !c.req.header("content-type")?.startsWith("application/json")
      )
        return c.json({ error: "Send application/json." }, 415);
    }
    await next();
  });
  app.onError((error, c) => {
    if (error instanceof z.ZodError)
      return c.json(
        {
          error: error.issues
            .map((i) => `${i.path.join(".")}: ${i.message}`)
            .join("; "),
        },
        400,
      );
    if (error instanceof SyntaxError)
      return c.json({ error: "Invalid JSON." }, 400);
    return c.json({ error: error.message }, 503);
  });
  const models = new Map<string, Promise<unknown>>();
  app.get("/api/model", async (c) => {
    const scene = z.enum(["g1", "wbc", "step_5cm", "step_35cm", "box_lift"]).parse(c.req.query("scene") || "g1");
    if (!models.has(scene))
      models.set(
        scene,
        bridge.request("model", { scene }).catch((error) => {
          models.delete(scene);
          throw error;
        }),
      );
    return c.json(await models.get(scene));
  });
  for (const action of ["catalog", "status"]) {
    app.get(`/api/${action}`, async (c) =>
      c.json(await bridge.request(action)),
    );
  }
  app.post("/api/run", async (c) => {
    const body = await c.req.json();
    const routine = routineSchema.parse(body);
    const mode = z
      .enum(["kinematic_playback", "wbc_tracking"])
      .optional()
      .parse(body.mode);
    if (!routine.entries.length)
      return c.json({ error: "Add a clip first." }, 400);
    return c.json(
      await bridge.request("run", mode ? { ...routine, mode } : routine),
    );
  });
  for (const action of ["pause", "resume", "stop", "reset", "task_exit"]) {
    app.post(`/api/${action}`, async (c) =>
      c.json(await bridge.request(action)),
    );
  }
  app.post("/api/seek", async (c) => {
    const payload = z
      .object({ seconds: z.number().min(0) })
      .parse(await c.req.json());
    return c.json(await bridge.request("seek", payload));
  });
  for (const action of ["task_prepare", "task_run"]) {
    app.post(`/api/${action}`, async (c) => {
      const payload = z.object({
        task: z.enum(["step_5cm", "step_35cm", "box_lift"]),
        controller: z.enum(["teacher", "student"]),
      }).parse(await c.req.json());
      return c.json(await bridge.request(action, payload));
    });
  }
  app.post("/api/physics", async (c) => {
    const payload = z
      .object({
        task: z.enum(["stand", "walk"]),
        controller: z.enum(["teacher", "student"]),
        seconds: z.number().min(1).max(10),
        push_force: z.number().min(0).max(100),
      })
      .parse(await c.req.json());
    return c.json(await bridge.request("physics", payload));
  });
  app.get("/api/routines", async (c) => {
    await mkdir(storage, { recursive: true });
    const routines: SavedRoutine[] = [];
    for (const file of await readdir(storage)) {
      if (!file.endsWith(".json")) continue;
      try {
        const data = JSON.parse(await readFile(join(storage, file), "utf8"));
        const value = routineSchema.parse(data);
        if (
          z.string().uuid().safeParse(data.id).success &&
          z.iso.datetime().safeParse(data.updated).success
        )
          routines.push({ ...value, id: data.id, updated: data.updated });
      } catch {
        /* One damaged local file must not hide all other routines. */
      }
    }
    return c.json(routines.sort((a, b) => b.updated.localeCompare(a.updated)));
  });
  app.post("/api/routines", async (c) => {
    const body = await c.req.json();
    const routine = routineSchema.parse(body);
    const id =
      body.id === undefined ? randomUUID() : z.string().uuid().parse(body.id);
    const saved = { ...routine, id, updated: new Date().toISOString() };
    await mkdir(storage, { recursive: true });
    const temporary = join(storage, `${id}.${randomUUID()}.tmp`);
    await writeFile(temporary, JSON.stringify(saved, null, 2));
    await rename(temporary, join(storage, `${id}.json`));
    return c.json(saved, 201);
  });
  app.delete("/api/routines/:id", async (c) => {
    const id = z.string().uuid().parse(c.req.param("id"));
    await unlink(join(storage, `${id}.json`)).catch((error) => {
      if (error.code !== "ENOENT") throw error;
    });
    return c.json({ ok: true });
  });
  app.all("/api/*", (c) => c.json({ error: "Unknown API route." }, 404));
  return app;
}
