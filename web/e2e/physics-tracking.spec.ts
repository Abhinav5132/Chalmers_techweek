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
            transitions: "continuous_anchored",
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

test("routine physics uses its own scene, sends the full sequence, disables seeking, and Play also runs physics", async ({
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
  await expect(page.locator(".viewport-mode")).toHaveText("PHYSICS · WBC TRACKING");
  await expect(page.getByLabel("Playback position")).toBeDisabled();
  await expect.poll(() => control.runs.length).toBe(2);
  expect(control.runs[1].mode).toBe("wbc_tracking");
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

test("missing tracking assets disable all play actions", async ({
  page,
}) => {
  await setup(page, false);
  await expect(
    page.getByRole("button", { name: "Run with physics", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Play routine", exact: true }),
  ).toBeDisabled();
  await expect(page.getByRole("button", { name: "Preview Jazz walk", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Physics lab", exact: true }).click();
  await expect(page.getByText("Missing policy.onnx")).toBeVisible();
});


test("library preview runs physics and fullscreen can be entered and exited", async ({ page }) => {
  const control = await setup(page);
  await page.getByRole("button", { name: "Preview Jazz walk", exact: true }).click();
  await expect.poll(() => control.runs.length).toBe(1);
  expect(control.runs[0]).toMatchObject({ mode: "wbc_tracking", entries: [{motion: "walk", repeats: 1}] });
  await page.getByRole("button", { name: "Enter fullscreen", exact: true }).click();
  await expect(page.locator(".viewer-stage")).toHaveClass(/viewer-fullscreen/);
  const rect = await page.locator(".viewer-stage").boundingBox();
  expect(rect?.width).toBe(page.viewportSize()!.width);
  await page.getByRole("button", { name: "Exit fullscreen", exact: true }).click();
  await expect(page.locator(".viewer-stage")).not.toHaveClass(/viewer-fullscreen/);
});


test("transition delay is editable, retained in the draft and sent to physics", async ({page}) => {
  const control = await setup(page);
  await page.getByRole("button", {name: "Select clip 1: Jazz walk", exact:true}).click();
  const slider = page.getByRole("slider", {name: "Transition delay", exact:true});
  await slider.focus();
  await slider.press("Home");
  for (let i=0; i<5; i++) await slider.press("ArrowRight");
  await expect(slider).toHaveValue("0.5");
  await expect(page.locator(".timeline-clip")).toContainText("0.5s hold");
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("g1-draft")!).entries[0].delay_after)).toBe(.5);
  await page.getByRole("button", {name:"Play routine",exact:true}).click();
  await expect.poll(() => control.runs.length).toBe(1);
  expect(control.runs[0].entries[0].delay_after).toBe(.5);
});
