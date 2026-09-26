import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";

const source = JSON.parse(readFileSync(new URL("../../src/motion_catalog.json", import.meta.url), "utf8"));
test("all source clips are searchable by title and category and can be arranged", async ({page}) => {
  await page.route("**/api/**", async route => {
    const path = new URL(route.request().url()).pathname.split("/").at(-1);
    if(path === "catalog") return route.fulfill({json:{
      motions: source.clips.map((c: any) => ({...c, description:"Source clip", frames:600, duration_seconds:10, tracking_available:true})),
      invalid:[], tracking:{available:true}, physics:{teacher_available:false,student_available:false}
    }});
    if(path === "model") return route.fulfill({json:{meshes:[],geoms:[]}});
    if(path === "routines") return route.fulfill({json:[]});
    return route.fulfill({json:{state:"idle",mode:"kinematic_playback",model_id:"g1",positions:[],quaternions:[],elapsed:0,total:0}});
  });
  await page.goto("/");
  await expect(page.locator(".motion-card")).toHaveCount(61);
  const search = page.getByPlaceholder("Find a movement…");
  await search.fill("bonus");
  await expect(page.locator(".motion-card")).toHaveCount(6);
  await search.fill("shuffle");
  await expect(page.locator(".motion-card")).toHaveCount(1);
  await page.getByRole("button",{name:"Add Shuffle",exact:true}).click();
  await expect(page.locator(".timeline-clip")).toContainText("Shuffle");
  expect(await page.evaluate(()=>JSON.parse(localStorage.getItem("g1-draft")!).entries[0].motion)).toBe("J_Dance17_Shuffle");
});
