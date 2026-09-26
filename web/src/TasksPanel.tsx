import { Box, Footprints, Play, RotateCcw, Pause, Square } from "lucide-react";
import type { Playback, TaskId, TaskPreset } from "./shared";

export function TaskCards({ presets, selected, disabled, choose }: {
  presets: TaskPreset[]; selected: TaskId; disabled: boolean; choose: (id: TaskId) => void;
}) {
  return <div className="task-cards" aria-label="Task demos">
    <div className="task-intro"><span className="eyebrow">PHYSICAL SKILLS</span><h2>Give the robot a task.</h2><p>Choose a prepared scene. Switching demos stops the current run and resets the scene.</p></div>
    {presets.map(p => <button key={p.id} className={`task-card ${selected === p.id ? "selected" : ""}`}
      disabled={disabled || !p.available} aria-pressed={selected === p.id} onClick={() => choose(p.id)}>
      {p.id === "box_lift" ? <Box size={23}/> : <Footprints size={23}/>}
      <strong>{p.title}</strong><span>{p.id === "box_lift" ? "Two-hand friction grasp · 150 g box" : "Single step · both feet onto the platform"}</span>
      <small>{p.error || `${p.seconds}s simulated trial`}</small>
    </button>)}
    {!presets.length && <p className="capability-note">Task demos are unavailable. Refresh the library after the backend connects.</p>}
  </div>;
}

export default function TasksPanel({ preset, playback, controller, setController, busy, connected, run, command }: {
  preset?: TaskPreset; playback: Playback | null; controller: string; setController: (value: string) => void;
  busy: boolean; connected: boolean; run: () => void; command: (action: string) => void;
}) {
  const active = playback?.mode === "task_physics" && playback.task_id === preset?.id;
  const state = active ? playback.state : "idle";
  const running = state === "running", paused = state === "paused";
  const locked = busy || running || paused;
  const unavailable = !preset?.available || (controller === "student" && !preset.student_available);
  const metrics = active ? playback : null;
  return <div className="inspector-content task-panel">
    <span className="eyebrow">PREPARED DEMO</span><h3>{preset?.title || "Select a task"}</h3>
    <p className="inspector-copy">{preset?.id === "box_lift" ? "Reach around the box, squeeze with both rubber hands, lift and hold. No fingers or hidden attachment." : "Climb one known step using motor torques and ground contact, then hold with both feet on top."}</p>
    <div className="task-controller">{active ? playback.controller === "student" ? "Learned student" : "Physics teacher" : controller === "student" ? "Learned student" : "Physics teacher"}</div>
    <button className="button primary full" disabled={busy || !connected || unavailable}
      onClick={() => running ? command("pause") : paused ? command("resume") : run()}>
      {running ? <Pause size={15}/> : <Play size={15}/>} {running ? "Pause demo" : paused ? "Resume demo" : "Run demo"}
    </button>
    <div className="task-actions">
      <button className="button outline" disabled={busy || !connected || !active} onClick={() => command("reset")}><RotateCcw size={13}/> Reset demo</button>
      <button className="button outline" disabled={busy || !connected || (!running && !paused)} onClick={() => command("stop")}><Square size={13}/> Stop</button>
    </div>
    <details className="task-advanced"><summary>Controller options</summary>
      <label className="field-label">CONTROLLER<select aria-label="Task controller" disabled={locked} value={controller} onChange={e => setController(e.target.value)}>
        <option value="teacher">Physics teacher</option>
        <option value="student" disabled={!preset?.student_available}>Learned student{!preset?.student_available ? " — unavailable" : ""}</option>
      </select></label>
      {!preset?.student_available && <p className="tracking-note">{preset?.student_error || "No qualified checkpoint installed."}</p>}
    </details>
    {preset?.error && <p className="capability-note">{preset.error}</p>}
    {metrics && <div className="trial-results" aria-label="Task results">
      <span className="eyebrow">MEASURED PROGRESS</span>
      <ol className="task-stages">{metrics.stages?.map((s,i) => <li key={s} aria-current={i === metrics.stage_index ? "step" : undefined}
        className={i === metrics.stage_index ? "active" : ""}>{s}</li>)}</ol>
      <p>Outcome <strong>{metrics.state === "completed" && metrics.passed ? "Success" : metrics.state}</strong></p>
      <p>Simulated time <strong>{metrics.elapsed.toFixed(1)} / {metrics.total.toFixed(0)} s</strong></p>
      {preset?.id === "box_lift" ? <>
        <p>Box lift <strong>{((metrics.lift_m || 0)*100).toFixed(1)} cm</strong></p>
        <p>Left / right hand <strong>{metrics.hand_contacts?.map(n => n.toFixed(1)).join(" / ")} N</strong></p>
        <p>Pedestal contact <strong>{(metrics.table_contact_n || 0).toFixed(2)} N</strong></p>
      </> : <>
        <p>Left / right foot <strong>{metrics.foot_contacts?.map(n => n.toFixed(0)).join(" / ")} N</strong></p>
        <p>Contact on step <strong>{metrics.top_contacts?.map(n => n.toFixed(0)).join(" / ")} N</strong></p>
      </>}
      <p>Stable hold <strong>{(metrics.hold_seconds || 0).toFixed(1)} / {metrics.hold_required} s</strong></p>
      <p>Pelvis uprightness <strong>{(metrics.upright || 0).toFixed(3)}</strong></p>
      {metrics.failure_reason && <p role="status" className="capability-note">{metrics.failure_reason.replaceAll("_", " ")}</p>}
      <p className="tracking-note">Stages reflect measured contacts and motion. Success requires the full trial and final hold. These are fixed-scene demos, not a combined climb-and-carry task.</p>
    </div>}
  </div>;
}
