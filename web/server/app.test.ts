import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createApp } from "./app";
const id = "7a01aa9d-4c5d-4f21-9c92-e16c69f4b795";
const entry = { id, motion: "walk", speed: 1, repeats: 1, delay_after: .5 };
const routine = {
  name: "Walk twice",
  entries: [entry, { ...entry, id: "cf890654-2e59-48c3-8f1e-88d1856d6217" }],
  loop: false,
};
const request = (body: unknown) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

test("API validates runs, preserves repeated clips, and rejects arbitrary commands and cross-site control", async () => {
  const calls: { action: string; payload: unknown }[] = [];
  const app = createApp(
    {
      request: async (action, payload) => {
        calls.push({ action, payload });
        return { state: "running" };
      },
    },
    "/unused",
  );
  assert.equal((await app.request("/api/run", request(routine))).status, 200);
  assert.deepEqual(calls[0], { action: "run", payload: routine });
  for (const body of [
    { ...routine, entries: [] },
    { ...routine, entries: [entry, entry] },
    { ...routine, entries: [{ ...entry, motion: "../secret" }] },
    { ...routine, entries: [{ ...entry, speed: 4 }] },
    { ...routine, entries: [{ ...entry, repeats: 0 }] },
    { ...routine, entries: [{ ...entry, delay_after: -1 }] },
    { ...routine, entries: [{ ...entry, delay_after: 11 }] },
  ]) {
    assert.equal((await app.request("/api/run", request(body))).status, 400);
  }
  assert.equal(calls.length, 1);
  assert.equal(
    (
      await app.request("/api/run", {
        ...request(routine),
        headers: {
          "Content-Type": "application/json",
          origin: "https://evil.example",
        },
      })
    ).status,
    403,
  );
  assert.equal(
    (
      await app.request("/api/run", {
        method: "POST",
        body: JSON.stringify(routine),
      })
    ).status,
    415,
  );
  assert.equal(
    (await app.request("/api/shell", request({ command: "pwd" }))).status,
    404,
  );
  assert.equal(
    (
      await app.request(
        "/api/physics",
        request({
          task: "dance",
          controller: "teacher",
          seconds: 10,
          push_force: 0,
        }),
      )
    ).status,
    400,
  );
});

test("routines round-trip, update atomically, and validate storage identifiers", async () => {
  const storage = await mkdtemp(join(tmpdir(), "g1-routines-"));
  try {
    const app = createApp({ request: async () => ({}) }, storage);
    const response = await app.request("/api/routines", request(routine));
    assert.equal(response.status, 201);
    const saved = await response.json();
    assert.equal(saved.entries.length, 2);
    assert.equal(saved.entries[0].delay_after, .5);
    const renamed = { ...saved, name: "Renamed" };
    assert.equal(
      (await app.request("/api/routines", request(renamed))).status,
      201,
    );
    const list = await (await app.request("/api/routines")).json();
    assert.equal(list.length, 1);
    assert.equal(list[0].name, "Renamed");
    assert.equal(list[0].entries[0].delay_after, .5);
    assert.equal(
      (
        await app.request(
          "/api/routines",
          request({ ...routine, id: "../../escape" }),
        )
      ).status,
      400,
    );
    assert.equal(
      (await app.request(`/api/routines/${saved.id}`, { method: "DELETE" }))
        .status,
      200,
    );
    assert.deepEqual(await (await app.request("/api/routines")).json(), []);
  } finally {
    await rm(storage, { recursive: true });
  }
});

test("worker failures reach the client without claiming success", async () => {
  const app = createApp(
    {
      request: async () => {
        throw new Error("Teacher checkpoint unavailable");
      },
    },
    "/unused",
  );
  const response = await app.request("/api/status");
  assert.equal(response.status, 503);
  assert.deepEqual(await response.json(), {
    error: "Teacher checkpoint unavailable",
  });
});

test("physics routine requests preserve the controller mode and isolate model caches", async () => {
  const calls: { action: string; payload: any }[] = [];
  const app = createApp(
    {
      request: async (action, payload) => {
        calls.push({ action, payload });
        return { ok: true };
      },
    },
    "/unused",
  );
  const physical = { ...routine, mode: "wbc_tracking" };
  assert.equal((await app.request("/api/run", request(physical))).status, 200);
  assert.deepEqual(calls[0], { action: "run", payload: physical });
  assert.equal(
    (await app.request("/api/run", request({ ...routine, mode: "shell" })))
      .status,
    400,
  );
  await app.request("/api/model");
  await app.request("/api/model?scene=wbc");
  await app.request("/api/model?scene=wbc");
  assert.deepEqual(calls.slice(1), [
    { action: "model", payload: { scene: "g1" } },
    { action: "model", payload: { scene: "wbc" } },
  ]);
  assert.equal((await app.request("/api/model?scene=../../file")).status, 400);
});
