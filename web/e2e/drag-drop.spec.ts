import { expect, test, type Page } from "@playwright/test";
import type { Routine } from "../src/shared";
const ids = [
  "a0742149-178a-40e7-a1a4-45d4618ebc83",
  "349fc874-c677-4893-a637-a94770b2c123",
  "b8d89b0a-e89c-4d02-a3fb-47ac88c84a73",
];
const entries = ["walk", "step_touch", "bow"].map((motion, i) => ({
  id: ids[i],
  motion,
  speed: 1,
  repeats: 1,
}));

async function setup(page: Page, state = "idle", clips = entries) {
  let playback = {
    state,
    mode: "kinematic_playback",
    elapsed: 2,
    total: 30,
    positions: [],
    quaternions: [],
  };
  const commands: string[] = [];
  // Only the editor is under test: no control commands reach the user's running robot.
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname.split("/").at(-1)!;
    let result: unknown = playback;
    if (path === "catalog")
      result = {
        motions: entries.map((e) => ({
          id: e.motion,
          description: "",
          duration_seconds: 10,
          frames: 600,
          tracking_available: true,
        })),
        invalid: [],
        tracking: { available: true },
        physics: { teacher_available: false, student_available: false },
      };
    if (path === "model") result = { meshes: [], geoms: [] };
    if (path === "routines") result = [];
    if (route.request().method() === "POST") {
      commands.push(path);
      if (path === "stop") playback = { ...playback, state: "stopped" };
      if (path === "pause") playback = { ...playback, state: "paused" };
      if (path === "run")
        playback = { ...playback, state: "running", elapsed: 0 };
      result = playback;
    }
    await route.fulfill({ json: result });
  });
  await page.addInitScript(
    (clips) =>
      localStorage.setItem(
        "g1-draft",
        JSON.stringify({ name: "Drag test", entries: clips, loop: false }),
      ),
    clips,
  );
  await page.goto("/");
  await expect(page.getByText("Python connected")).toBeVisible();
  await expect(page.locator(".motion-card")).toHaveCount(3);
  return commands;
}
async function order(page: Page) {
  return page.evaluate(() =>
    (JSON.parse(localStorage.getItem("g1-draft")!) as Routine).entries.map(
      (e) => e.motion,
    ),
  );
}

test("a paused routine can be reordered with a real mouse drag and restarts instead of resuming stale playback", async ({
  page,
}) => {
  const commands = await setup(page, "paused");
  await expect(page.locator(".timeline-clip").first()).toHaveAttribute(
    "draggable",
    "true",
  );
  await page
    .locator(".timeline-clip")
    .first()
    .dragTo(page.locator(".timeline-clip").nth(1));
  await expect.poll(() => order(page)).toEqual(["step_touch", "walk", "bow"]);
  expect(commands).toContain("stop");
  await expect(
    page.getByRole("button", { name: "Play routine", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Play routine", exact: true }).click();
  await expect.poll(() => commands.at(-1)).toBe("run");
});

test("an existing clip can be dragged to the empty area at the end", async ({
  page,
}) => {
  await setup(page);
  await page
    .locator(".timeline-clip")
    .first()
    .dragTo(page.locator(".timeline-add"));
  await expect.poll(() => order(page)).toEqual(["step_touch", "bow", "walk"]);
});

test("library clips can be dropped onto an empty timeline or a specific clip", async ({
  page,
}) => {
  await setup(page, "idle", []);
  await page
    .locator(".motion-card")
    .filter({ hasText: "Jazz walk" })
    .dragTo(page.locator(".timeline-track"));
  await expect.poll(() => order(page)).toEqual(["walk"]);
  await page
    .locator(".motion-card")
    .filter({ hasText: "Karate bow" })
    .dragTo(page.locator(".timeline-clip").first());
  await expect.poll(() => order(page)).toEqual(["bow", "walk"]);
});

test("running playback explains the edit lock and offers a pause action", async ({
  page,
}) => {
  await setup(page, "running");
  await expect(
    page.getByRole("button", { name: "Pause to edit" }),
  ).toBeVisible();
  await expect(page.locator(".timeline-clip").first()).toHaveAttribute(
    "draggable",
    "false",
  );
  await page.getByRole("button", { name: "Pause to edit" }).click();
  await expect(page.locator(".timeline-clip").first()).toHaveAttribute(
    "draggable",
    "true",
  );
});


test("remove buttons delete the selected instance during playback and while stopped", async ({ page }) => {
  const commands = await setup(page, "running", [entries[0], {...entries[0], id: ids[1]}, entries[2]]);
  await page.getByRole("button", { name: "Remove clip 2: Jazz walk", exact: true }).click();
  await expect.poll(() => order(page)).toEqual(["walk", "bow"]);
  expect(commands).toContain("stop");
  await page.getByRole("button", { name: "Select clip 1: Jazz walk", exact: true }).click();
  await page.getByRole("button", { name: "Remove clip", exact: true }).click();
  await expect.poll(() => order(page)).toEqual(["bow"]);
  await page.getByRole("button", { name: "Remove clip 1: Karate bow", exact: true }).click();
  await expect(page.locator(".timeline-empty")).toBeVisible();
});
