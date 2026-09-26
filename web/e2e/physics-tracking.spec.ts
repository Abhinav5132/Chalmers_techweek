import { expect, test, type Page } from "@playwright/test";
import type { Playback } from "../src/shared";

async function setup(page: Page, available = true) {
  const entry = {
    id: "039e684b-0927-4d38-b6e8-28e33524f91b",
    motion: "walk",
    speed: 1,
    repeats: 2,
  };
  let state: Playback = {
    state: "idle",
    mode: "kinematic_playback",
    model_id: "g1",
    elapsed: 0,
    total: 0,
    positions: [],
    quaternions: [],
  };
  const runs: any[] = [];
  const scenes: string[] = [];
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname.split("/").at(-1);
    if (path === "catalog")
      return route.fulfill({
        json: {
          motions: [
            {
              id: "walk",
              duration_seconds: 10.6,
              frames: 636,
              description: "",
              tracking_available: true,
            },
          ],
          invalid: [],
          physics: { teacher_available: false, student_available: false },
          tracking: {
            available,
            error: available ? null : "Missing policy.onnx",
            controller: "Phase-2 WBC",
            transitions: "reset_at_each_clip_and_repeat",
          },
        },
      });
    if (path === "model") {
      const scene = url.searchParams.get("scene") || "g1";
      scenes.push(scene);
      return route.fulfill({
        json: {
          id: scene,
          root_body: scene === "wbc" ? 2 : 1,
          meshes: [],
          geoms: [],
        },
      });
    }
    if (path === "routines") return route.fulfill({ json: [] });
    if (route.request().method() === "POST") {
      if (path === "run") {
        const body = route.request().postDataJSON();
        runs.push(body);
        state = {
          ...state,
          mode: body.mode,
          model_id: body.mode === "wbc_tracking" ? "wbc" : "g1",
          state: "running",
          total: 21.2,
          elapsed: 1,
          motion: "walk",
          tracking_rmse_rad: 0.125,
          root_height_m: 0.79,
        };
      }
      if (path === "pause") state = { ...state, state: "paused" };
      if (path === "resume") state = { ...state, state: "running" };
      if (path === "stop") state = { ...state, state: "stopped" };
    }
    return route.fulfill({ json: state });
  });
  await page.addInitScript(
    (entry) =>
      localStorage.setItem(
        "g1-draft",
        JSON.stringify({ name: "Physics test", entries: [entry], loop: true }),
      ),
    entry,
  );
  await page.goto("/");
  await expect(page.getByText("Python connected")).toBeVisible();
  await expect(page.locator(".motion-card")).toHaveCount(1);
  return {
    runs,
    scenes,
    fall() {
      state = {
        ...state,
        state: "fallen",
        fell: true,
        failure_reason: "Pelvis fell below 0.25 m.",
      };
    },
  };
}

test("routine physics uses its own scene, sends the full sequence, disables seeking, and switches back to preview", async ({
  page,
}) => {
  const control = await setup(page);
  await page
    .getByRole("button", { name: "Run with physics", exact: true })
    .click();
  await expect(page.locator(".viewport-mode")).toHaveText(
    "PHYSICS · WBC TRACKING",
  );
  await expect(page.getByLabel("Routine physics results")).toContainText(
    "0.125 rad",
  );
  await expect(page.getByLabel("Playback position")).toBeDisabled();
  expect(control.runs[0]).toMatchObject({
    mode: "wbc_tracking",
    loop: true,
    entries: [{ motion: "walk", speed: 1, repeats: 2 }],
  });
  expect(control.scenes).toContain("wbc");
  await page
    .getByRole("button", { name: "Pause playback", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Resume playback", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Resume playback", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Stop playback", exact: true })
    .click();
  await page.getByRole("button", { name: "Play routine", exact: true }).click();
  await expect(page.locator(".viewport-mode")).toHaveText("MOTION PREVIEW");
  await expect(page.getByLabel("Playback position")).toBeEnabled();
  expect(control.runs[1].mode).toBe("kinematic_playback");
});

test("a fallen trial reports the reason and never silently falls back to animation", async ({
  page,
}) => {
  const control = await setup(page);
  await page
    .getByRole("button", { name: "Run with physics", exact: true })
    .click();
  await expect(page.locator(".viewport-mode")).toHaveText(
    "PHYSICS · WBC TRACKING",
  );
  await expect.poll(() => control.runs.length).toBe(1);
  control.fall();
  await expect(page.getByLabel("Routine physics results")).toContainText(
    "Pelvis fell below 0.25 m.",
  );
  await expect(page.locator(".play-state")).toContainText("fallen");
  await expect(page.locator(".viewport-mode")).toHaveText(
    "PHYSICS · WBC TRACKING",
  );
  expect(control.runs).toHaveLength(1);
});

test("missing tracking assets leave recording preview available", async ({
  page,
}) => {
  await setup(page, false);
  await expect(
    page.getByRole("button", { name: "Run with physics", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Play routine", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Physics lab", exact: true }).click();
  await expect(page.getByText("Missing policy.onnx")).toBeVisible();
});
