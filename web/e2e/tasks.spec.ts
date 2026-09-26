import { test, expect, type Page } from "@playwright/test";
import type { Playback, TaskId } from "../src/shared";

async function setup(page: Page) {
  const calls: {action: string, body: any}[] = [];
  let state: Playback = {state:"idle",mode:"kinematic_playback",model_id:"g1",elapsed:0,total:0,positions:[],quaternions:[]};
  const presets = [
    {id:"step_5cm",title:"Climb a 5 cm step",seconds:23},
    {id:"step_35cm",title:"Climb a 35 cm step",seconds:23},
    {id:"box_lift",title:"Lift a box",seconds:12},
  ].map(p=>({...p,available:true,student_available:false,student_error:"No qualified learned student is installed for this preset. Use Teacher."}));
  await page.route("**/api/**",async route=>{
    const url = new URL(route.request().url()), action=url.pathname.split("/").at(-1)!;
    if(action==="catalog")return route.fulfill({json:{motions:[],invalid:[],physics:{},tracking:{available:true},tasks:presets}});
    if(action==="model")return route.fulfill({json:{id:url.searchParams.get("scene"),root_body:1,meshes:[],geoms:[]}});
    if(action==="routines")return route.fulfill({json:[]});
    if(route.request().method()==="POST"){
      const body=route.request().postDataJSON();calls.push({action,body});
      if(action==="task_prepare"||action==="task_run")state={...state,state:action==="task_run"?"running":"idle",mode:"task_physics",model_id:body.task,task_id:body.task,controller:body.controller,elapsed:0,total:body.task==="box_lift"?12:23,stages:body.task==="box_lift"?["Reach","Squeeze","Lift","Hold"]:["Balance","First foot","Second foot","Hold"],stage_index:0,passed:false,hand_contacts:[0,0],top_contacts:[0,0],foot_contacts:[170,170],hold_seconds:0,hold_required:2,table_contact_n:1.47,lift_m:0};
      if(action==="pause")state={...state,state:"paused"};
      if(action==="resume")state={...state,state:"running"};
      if(action==="stop")state={...state,state:"stopped"};
      if(action==="task_exit")state={state:"idle",mode:"kinematic_playback",model_id:"g1",elapsed:0,total:0,positions:[],quaternions:[]};
      if(action==="reset")state={...state,state:"idle",elapsed:0,passed:false,failure_reason:undefined};
    }
    return route.fulfill({json:state});
  });
  await page.goto("/");
  await expect(page.getByText("Python connected")).toBeVisible();
  await page.getByRole("button",{name:"Tasks",exact:true}).click();
  await expect(page.locator(".task-card")).toHaveCount(3);
  return {calls,fail:()=>{state={...state,state:"failed",failure_reason:"object_dropped",passed:false};}};
}

test("task scenes run independently of the dance timeline and expose demo controls",async({page})=>{
  const c=await setup(page);
  await expect.poll(()=>c.calls.at(-1)?.action).toBe("task_prepare");
  await page.getByRole("button",{name:/Lift a box/}).click();
  await expect(page.locator(".viewport-mode")).toHaveText("PHYSICS · BOX LIFT");
  await page.getByRole("button",{name:"Run demo",exact:true}).click();
  await expect.poll(()=>c.calls.at(-1)?.action).toBe("task_run");
  expect(c.calls.at(-1)?.body).toEqual({task:"box_lift",controller:"teacher"});
  await expect(page.getByLabel("Playback position")).toBeDisabled();
  await expect(page.getByLabel("Task results")).toContainText("Pedestal contact");
  await page.getByRole("button",{name:"Pause demo",exact:true}).click();
  await page.getByRole("button",{name:"Resume demo",exact:true}).click();
  await page.getByRole("button",{name:"Reset demo",exact:true}).click();
  await expect(page.locator(".viewport-mode")).toHaveText("PHYSICS · BOX LIFT");
  await expect(page.getByRole("button",{name:"Run demo",exact:true})).toBeEnabled();
  await page.getByText("Controller options",{exact:true}).click();
  await expect(page.locator('option[value="student"]')).toHaveAttribute('disabled', '');
  await page.getByRole("button",{name:/Climb a 35 cm step/}).click();
  await expect.poll(()=>c.calls.at(-1)?.body.task).toBe("step_35cm");
  await page.getByRole("button",{name:"Play task demo",exact:true}).click();
  await expect.poll(()=>c.calls.at(-1)?.action).toBe("task_run");
  expect(c.calls.at(-1)?.body.task).toBe("step_35cm");
});

test("switch demos during running and paused trials",async({page})=>{
  const c=await setup(page);
  await page.getByRole("button",{name:"Run demo",exact:true}).click();
  await expect(page.getByRole("button",{name:"Pause demo",exact:true})).toBeEnabled();
  await page.getByRole("button",{name:/Climb a 35 cm step/}).click();
  await expect(page.getByRole("button",{name:"Run demo",exact:true})).toBeEnabled();
  expect(c.calls.slice(-2)).toEqual([
    {action:"stop",body:{}},
    {action:"task_prepare",body:{task:"step_35cm",controller:"teacher"}},
  ]);
  await page.getByRole("button",{name:"Run demo",exact:true}).click();
  await page.getByRole("button",{name:"Pause demo",exact:true}).click();
  await expect(page.getByRole("button",{name:"Resume demo",exact:true})).toBeEnabled();
  await page.getByRole("button",{name:/Lift a box/}).click();
  await expect(page.locator(".viewport-mode")).toHaveText("PHYSICS · BOX LIFT");
  await expect(page.getByRole("button",{name:"Run demo",exact:true})).toBeEnabled();
  expect(c.calls.slice(-2)).toEqual([
    {action:"stop",body:{}},
    {action:"task_prepare",body:{task:"box_lift",controller:"teacher"}},
  ]);
  await page.getByRole("button",{name:"Run demo",exact:true}).click();
  await expect(page.getByRole("button",{name:"Pause demo",exact:true})).toBeEnabled();
  expect(c.calls.at(-1)).toEqual({action:"task_run",body:{task:"box_lift",controller:"teacher"}});
});

for (const destination of ["Compose", "Physics lab"]) {
  test(`leaving Tasks for ${destination} clears the scene and returning prepares a fresh demo`,async({page})=>{
    const c=await setup(page);
    await page.getByRole("button",{name:/Lift a box/}).click();
    await page.getByRole("button",{name:"Run demo",exact:true}).click();
    await expect(page.getByRole("button",{name:"Pause demo",exact:true})).toBeEnabled();
    await page.getByRole("button",{name:destination,exact:true}).click();
    await expect(page.locator(".scene-label")).toContainText("Flat ground");
    expect(c.calls.at(-1)?.action).toBe("task_exit");
    await expect(page.getByLabel("Task demos")).toHaveCount(0);
    await page.getByRole("button",{name:"Tasks",exact:true}).click();
    await expect(page.getByRole("button",{name:"Run demo",exact:true})).toBeEnabled();
    expect(c.calls.at(-1)).toEqual({action:"task_prepare",body:{task:"box_lift",controller:"teacher"}});
  });
}

test("task failure is visible, not success; fullscreen remains available",async({page})=>{
  const c=await setup(page);
  await page.getByRole("button",{name:"Run demo",exact:true}).click();
  await expect.poll(()=>c.calls.at(-1)?.action).toBe("task_run");
  c.fail();
  await expect(page.getByLabel("Task results")).toContainText("object dropped");
  await expect(page.getByLabel("Task results").locator("p").filter({hasText:"Outcome"})).toHaveText("Outcome failed");
  await page.getByRole("button",{name:"Enter fullscreen",exact:true}).click();
  await expect(page.locator(".viewer-stage")).toHaveClass(/viewer-fullscreen/);
  await page.getByRole("button",{name:"Exit fullscreen",exact:true}).click();
});
