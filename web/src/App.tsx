import { useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import {
  Activity,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  Check,
  ChevronDown,
  Copy,
  Download,
  FolderOpen,
  GripVertical,
  Layers3,
  LoaderCircle,
  Pause,
  Play,
  Plus,
  Repeat2,
  RotateCcw,
  Save,
  Search,
  SlidersHorizontal,
  Square,
  Trash2,
  Upload,
  X,
  Zap,
} from "lucide-react";
import RobotViewer from "./RobotViewer";
import {
  routineSchema,
  type Catalog,
  type Entry,
  type Motion,
  type Playback,
  type RobotModel,
  type Routine,
  type SavedRoutine,
} from "./shared";

const emptyRoutine: Routine = {
  name: "Untitled routine",
  entries: [],
  loop: false,
};
const labels: Record<string, string> = {
  walk: "Jazz walk",
  step_touch: "Step touch",
  bow: "Karate bow",
};
const fallbackTitle = (id: string) =>
  labels[id] || id.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
const time = (n: number) =>
  `${Math.floor(n / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor(n % 60)
    .toString()
    .padStart(2, "0")}`;
const colors = ["lime", "blue", "peach"];
function initialDraft(): Routine {
  try {
    return routineSchema.parse(
      JSON.parse(localStorage.getItem("g1-draft") || "null"),
    );
  } catch {
    return emptyRoutine;
  }
}
async function api<T>(
  path: string,
  body?: unknown,
  method?: string,
): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    method: method || (body === undefined ? "GET" : "POST"),
    headers:
      body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Request failed.");
  return result;
}
function Waveform({ seed = 0 }: { seed?: number }) {
  return (
    <div className="waveform" aria-hidden="true">
      {Array.from({ length: 38 }, (_, i) => (
        <i
          key={i}
          style={{
            height: `${15 + Math.abs(Math.sin(i * 1.23 + seed) * Math.cos(i * 0.31)) * 85}%`,
          }}
        />
      ))}
    </div>
  );
}
export default function App() {
  const [routine, setRoutine] = useState<Routine>(initialDraft);
  const [savedId, setSavedId] = useState<string | undefined>(() => {
    try {
      const id = localStorage.getItem("g1-draft-id");
      return id && /^[0-9a-f-]{36}$/i.test(id) ? id : undefined;
    } catch {
      return undefined;
    }
  });
  const [saved, setSaved] = useState<SavedRoutine[]>([]);
  const [selected, setSelected] = useState<string>();
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [model, setModel] = useState<RobotModel | null>(null);
  const models = useRef(new Map<string, Promise<RobotModel>>());
  const [playback, setPlayback] = useState<Playback | null>(null);
  const [connected, setConnected] = useState(false);
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [library, setLibrary] = useState(false);
  const [tab, setTab] = useState<"compose" | "physics">("compose");
  const [task, setTask] = useState("stand");
  const [controller, setController] = useState("teacher");
  const [seconds, setSeconds] = useState(10);
  const [push, setPush] = useState(0);
  const [seek, setSeek] = useState<number | null>(null);
  const importInput = useRef<HTMLInputElement>(null);
  const searchInput = useRef<HTMLInputElement>(null);
  const modal = useRef<HTMLElement>(null);
  const locked = busy || playback?.state === "running";
  const dragged = useRef<{ kind: "motion" | "entry"; id: string } | null>(null);
  const [dropTarget, setDropTarget] = useState<string | null>(null);
  const selectedEntry = routine.entries.find((e) => e.id === selected);
  const motions = catalog?.motions || [];
  const motionMap = new Map(motions.map((m) => [m.id, m]));
  const title = (id: string) => motionMap.get(id)?.title || fallbackTitle(id);
  const duration = (entry: Entry) =>
    ((motionMap.get(entry.motion)?.duration_seconds || 0) / entry.speed) *
    entry.repeats;
  const total = routine.entries.reduce((sum, e, i) => sum + duration(e) +
    (e.delay_after || 0) * (e.repeats - (!routine.loop && i === routine.entries.length - 1 ? 1 : 0)), 0);
  const available = routine.entries.every((e) => motionMap.has(e.motion));
  const color = (id: string) =>
    colors[
      Math.max(
        0,
        motions.findIndex((m) => m.id === id),
      ) % colors.length
    ];
  const state = playback?.state || "idle";
  const canPhysics =
    !!catalog?.physics.teacher_available &&
    (controller === "teacher" || !!catalog?.physics.student_available);

  useEffect(() => {
    try {
      localStorage.setItem("g1-draft", JSON.stringify(routine));
      if (savedId) localStorage.setItem("g1-draft-id", savedId);
      else localStorage.removeItem("g1-draft-id");
    } catch {
      /* Server saves still work when browser storage is unavailable. */
    }
  }, [routine, savedId]);
  useEffect(() => {
    if (!connected) return;
    let cancelled = false;
    api<Catalog>("catalog")
      .then((value) => {
        if (!cancelled) setCatalog(value);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    api<SavedRoutine[]>("routines")
      .then((value) => {
        if (!cancelled) setSaved(value);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [connected]);
  function loadModel(scene: "g1" | "wbc") {
    if (!models.current.has(scene))
      models.current.set(
        scene,
        api<RobotModel>(`model?scene=${scene}`)
          .then((value) => ({ ...value, id: scene }))
          .catch((error) => {
            models.current.delete(scene);
            throw error;
          }),
      );
    return models.current.get(scene)!;
  }
  const scene = playback?.model_id || "g1";
  useEffect(() => {
    if (!connected) return;
    let cancelled = false;
    loadModel(scene)
      .then((value) => {
        if (!cancelled) setModel(value);
      })
      .catch((error) => {
        if (!cancelled) setError(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, [connected, scene]);
  useEffect(() => {
    let cancelled = false,
      timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      const started = performance.now();
      let retryDelay = 0;
      try {
        const result = await api<Playback>("status");
        if (!cancelled) {
          setPlayback(result);
          setConnected(true);
        }
      } catch {
        if (!cancelled) setConnected(false);
        retryDelay = 1500;
      }
      // Target 60 Hz including request time, with at most one request in flight.
      // Slow responses lower the update rate instead of accumulating requests.
      if (!cancelled)
        timer = setTimeout(
          poll,
          retryDelay || Math.max(0, 1000 / 60 - (performance.now() - started)),
        );
    };
    void poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);
  useEffect(() => {
    const shortcuts = (event: KeyboardEvent) => {
      if (event.key === "Escape") setLibrary(false);
      if (
        event.key === "/" &&
        !["INPUT", "TEXTAREA", "SELECT"].includes(
          (event.target as HTMLElement).tagName,
        )
      ) {
        event.preventDefault();
        searchInput.current?.focus();
      }
    };
    window.addEventListener("keydown", shortcuts);
    return () => window.removeEventListener("keydown", shortcuts);
  }, []);
  useEffect(() => {
    if (!library) return;
    const previous = document.activeElement as HTMLElement;
    const trap = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const items = modal.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), input:not([hidden]), [tabindex="0"]',
      );
      if (!items?.length) return;
      const first = items[0],
        last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", trap);
    return () => {
      document.removeEventListener("keydown", trap);
      previous?.focus();
    };
  }, [library]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 3500);
    return () => clearTimeout(timer);
  }, [notice]);
  async function action(fn: () => Promise<void>) {
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }
  const command = (name: string, body: unknown = {}) =>
    action(async () => setPlayback(await api<Playback>(name, body)));
  const run = () => startRoutine("wbc_tracking");
  function startRoutine(mode: "kinematic_playback" | "wbc_tracking") {
    if (mode === "wbc_tracking") setTab("physics");
    return action(async () => {
      const robot = await loadModel(mode === "wbc_tracking" ? "wbc" : "g1");
      const result = await api<Playback>("run", { ...routine, mode });
      setModel(robot);
      setPlayback(result);
    });
  }
  const trackingError = !catalog?.tracking?.available
    ? catalog?.tracking?.error ||
      "The phase-2 tracking controller is unavailable."
    : routine.entries.some(
          (entry) => !motionMap.get(entry.motion)?.tracking_available,
        )
      ? "Some selected clips lack the body velocities required for physics tracking."
      : null;
  const canTrack =
    connected && !locked && routine.entries.length > 0 && !trackingError;

  function editRoutine(
    update: (routine: Routine) => Routine,
    after?: () => void,
    allowRunning = false,
  ) {
    if (busy || (locked && !allowRunning)) return;
    const apply = () => {
      setRoutine(update);
      after?.();
    };
    if (playback?.state === "paused" || playback?.state === "running") {
      void action(async () => {
        setPlayback(await api<Playback>("stop", {}));
        apply();
        setNotice(
          "Routine updated. Play starts the new sequence from the beginning.",
        );
      });
    } else apply();
  }
  function add(motion: string, beforeId?: string) {
    if (locked || routine.entries.length >= 100) return;
    const entry = { id: crypto.randomUUID(), motion, speed: 1, repeats: 1 };
    editRoutine(
      (r) => {
        const entries = [...r.entries];
        const index = beforeId
          ? entries.findIndex((e) => e.id === beforeId)
          : -1;
        entries.splice(index < 0 ? entries.length : index, 0, entry);
        return { ...r, entries };
      },
      () => setSelected(entry.id),
    );
  }
  function updateEntry(patch: Partial<Entry>) {
    editRoutine((r) => ({
      ...r,
      entries: r.entries.map((e) =>
        e.id === selected ? { ...e, ...patch } : e,
      ),
    }));
  }
  function reorder(from: number, to: number) {
    if (
      locked ||
      from === to ||
      from < 0 ||
      from >= routine.entries.length ||
      to < 0 ||
      to >= routine.entries.length
    )
      return;
    const id = routine.entries[from].id;
    editRoutine(
      (r) => {
        const entries = [...r.entries];
        const index = entries.findIndex((e) => e.id === id);
        if (index < 0) return r;
        entries.splice(to, 0, entries.splice(index, 1)[0]);
        return { ...r, entries };
      },
      () => setSelected(id),
    );
  }
  function duplicate() {
    if (!selectedEntry || locked || routine.entries.length >= 100) return;
    const entry = { ...selectedEntry, id: crypto.randomUUID() };
    editRoutine(
      (r) => {
        const entries = [...r.entries];
        entries.splice(
          entries.findIndex((e) => e.id === selected) + 1,
          0,
          entry,
        );
        return { ...r, entries };
      },
      () => setSelected(entry.id),
    );
  }
  function remove(id = selected) {
    if (!id) return;
    editRoutine(
      (r) => ({ ...r, entries: r.entries.filter((e) => e.id !== id) }),
      () => setSelected((current) => current === id ? undefined : current),
      true,
    );
  }
  function startDrag(event: DragEvent, kind: "motion" | "entry", id: string) {
    if (locked) {
      event.preventDefault();
      return;
    }
    dragged.current = { kind, id };
    event.dataTransfer.setData(`application/x-g1-${kind}`, id);
    event.dataTransfer.setData("text/plain", id);
    event.dataTransfer.effectAllowed = kind === "motion" ? "copy" : "move";
  }
  function endDrag() {
    dragged.current = null;
    setDropTarget(null);
  }
  function dragOver(event: DragEvent, target = "end") {
    if (locked || !dragged.current) return;
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect =
      dragged.current.kind === "motion" ? "copy" : "move";
    setDropTarget(target);
  }
  function drop(event: DragEvent, targetId?: string) {
    event.preventDefault();
    event.stopPropagation();
    const source = dragged.current;
    endDrag();
    if (locked || !source) return;
    if (source.kind === "motion") {
      if (motionMap.has(source.id)) add(source.id, targetId);
    } else {
      const from = routine.entries.findIndex((e) => e.id === source.id);
      const to = targetId
        ? routine.entries.findIndex((e) => e.id === targetId)
        : routine.entries.length - 1;
      reorder(from, to);
    }
  }
  function preview(motion: Motion) {
    if (locked || !catalog?.tracking?.available || !motion.tracking_available) return;
    void action(async () => {
      const robot = await loadModel("wbc");
      const result = await api<Playback>("run", {
        name: "Clip preview",
        entries: [{ id: crypto.randomUUID(), motion: motion.id, speed: 1, repeats: 1 }],
        loop: false,
        mode: "wbc_tracking",
      });
      setModel(robot);
      setPlayback(result);
    });
  }
  function saveRoutine() {
    void action(async () => {
      const result = await api<SavedRoutine>("routines", {
        ...routine,
        id: savedId,
      });
      setSavedId(result.id);
      setSaved(await api<SavedRoutine[]>("routines"));
      setNotice("Routine saved");
    });
  }
  function exportRoutine() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(routine, null, 2)], {
        type: "application/json",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `${routine.name.replace(/[^a-z0-9_-]/gi, "-") || "routine"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
  async function importRoutine(file?: File) {
    if (!file) return;
    if (file.size > 100_000) {
      setError("Routine file must be smaller than 100 KB.");
      return;
    }
    await action(async () => {
      const parsed = routineSchema.parse(JSON.parse(await file.text()));
      if (playback?.state === "paused")
        setPlayback(await api<Playback>("stop", {}));
      setRoutine(parsed);
      setSelected(undefined);
      setSavedId(undefined);
      setNotice("Routine imported");
    });
  }
  async function refresh() {
    await action(async () => {
      setCatalog(await api<Catalog>("catalog"));
      if (!model) setModel(await loadModel(scene));
      setNotice("Motion library refreshed");
    });
  }
  // Keep the full library out of the 60 Hz pose-render path. Rebuild cards when
  // their content or editing state changes; paused edits still use current state.
  const motionCards = useMemo(() => (
<div className="motion-list">
            {motions
              .filter((m) =>
                `${title(m.id)} ${m.category || ""} ${m.id}`.toLowerCase().includes(query.toLowerCase()),
              )
              .map((motion, i) => (
                <article
                  key={motion.id}
                  className={`motion-card ${color(motion.id)}`}
                  draggable={!locked}
                  onDragStart={(e) => startDrag(e, "motion", motion.id)}
                  onDragEnd={endDrag}
                >
                  <div className="motion-card-top">
                    <span className="motion-symbol">
                      {motion.id === "walk" ? (
                        <Activity size={20} />
                      ) : motion.id === "bow" ? (
                        <ArrowDown size={20} />
                      ) : (
                        <ArrowRight size={20} />
                      )}
                    </span>
                    <span className="motion-kind">
                      {motion.category?.toUpperCase() || (motion.id === "walk" ? "LOCOMOTION" : "GESTURE")}
                    </span>
                    <button
                      className="icon-button"
                      disabled={locked || routine.entries.length >= 100}
                      onClick={() => add(motion.id)}
                      aria-label={`Add ${title(motion.id)}`}
                      title="Add to routine"
                    >
                      <Plus size={17} />
                    </button>
                  </div>
                  <h3>{title(motion.id)}</h3>
                  <p>
                    {motion.id === "walk"
                      ? "A rhythmic stride forward."
                      : motion.id === "step_touch"
                        ? "Find your side-to-side rhythm."
                        : motion.id === "bow"
                          ? "Finish with a little respect."
                          : motion.description}
                  </p>
                  <Waveform seed={i * 3} />
                  <div className="motion-card-footer">
                    <span>
                      {motion.duration_seconds.toFixed(1)}s <i />{" "}
                      {motion.frames} frames
                    </span>
                    <button
                      onClick={() => preview(motion)}
                      disabled={locked || !connected || !catalog?.tracking?.available || !motion.tracking_available}
                      aria-label={`Preview ${title(motion.id)}`}
                    >
                      <Play size={11} fill="currentColor" /> Preview
                    </button>
                  </div>
                </article>
              ))}
            {catalog && !motions.length && (
              <div className="empty-library">
                <Layers3 />
                <p>No recordings yet.</p>
                <span>
                  Run <code>just download-motions</code>, then refresh.
                </span>
              </div>
            )}
            {!catalog && (
              <p className="empty-library">
                {connected
                  ? "Loading your recordings…"
                  : "Waiting for the Python worker…"}
              </p>
            )}
            {motions.length > 0 &&
              !motions.some((m) =>
                `${title(m.id)} ${m.category || ""} ${m.id}`.toLowerCase().includes(query.toLowerCase()),
              ) && (
                <p className="empty-library">No movements match “{query}”.</p>
              )}
          </div>
  ), [catalog, query, locked, busy, connected, routine, playback?.state]);
  return (
    <div className="studio selection:bg-studio-lime selection:text-studio-ink">
      <header className="topbar">
        <a href="/" className="brand" aria-label="G1 Motion Studio">
          <span className="brand-mark">
            G<span>1</span>
          </span>
          <span>
            motion<span className="brand-light">studio</span>
            <small>ROBOTICS, IN MOTION.</small>
          </span>
        </a>
        <div className="workspace-label">
          <span className="tiny-square" /> PERSONAL WORKSPACE{" "}
          <span className="divider">/</span> <span>G1 projects</span>
        </div>
        <div className="topbar-right">
          <span className={`connection ${connected ? "" : "offline"}`}>
            <i />
            {connected ? "Python connected" : "Connecting to Python"}
          </span>
          <span className="avatar">G1</span>
        </div>
      </header>
      <div className="projectbar">
        <div className="project-title">
          <span className="eyebrow">ROUTINE EDITOR</span>
          <div>
            <input
              aria-label="Routine name"
              maxLength={80}
              value={routine.name}
              onChange={(e) =>
                setRoutine((r) => ({ ...r, name: e.target.value }))
              }
            />
            <span className="draft-pill">LOCAL DRAFT</span>
          </div>
        </div>
        <div className="project-actions">
          <button
            className="button quiet"
            onClick={() => {
              setLibrary(true);
              void action(async () =>
                setSaved(await api<SavedRoutine[]>("routines")),
              );
            }}
          >
            <FolderOpen size={16} /> Open
          </button>
          <button
            className="button quiet"
            onClick={exportRoutine}
            title="Export routine JSON"
          >
            <Download size={16} />
            <span className="optional">Export</span>
          </button>
          <button
            className="button save"
            onClick={saveRoutine}
            disabled={busy || !routine.name.trim()}
          >
            <Save size={15} /> Save routine
          </button>
        </div>
      </div>
      {(error || playback?.error) && (
        <div className="error-banner" role="alert">
          <span>{error || playback?.error}</span>
          <button
            className="icon-button"
            onClick={() => {
              setError("");
              if (playback?.error) void command("reset");
            }}
            aria-label="Dismiss error"
          >
            <X size={16} />
          </button>
        </div>
      )}
      <main className="workspace">
        <aside className="motion-library">
          <div className="panel-heading">
            <h2>
              Motion library{" "}
              <span>{motions.length.toString().padStart(2, "0")}</span>
            </h2>
            <button
              className="icon-button"
              title="Refresh motion library"
              aria-label="Refresh motion library"
              onClick={refresh}
              disabled={busy}
            >
              <RotateCcw size={15} />
            </button>
          </div>
          <p className="panel-description">
            A little movement goes a long way.
          </p>
          <label className="search">
            <Search size={15} />
            <input
              ref={searchInput}
              placeholder="Find a movement…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Search motions"
            />
            <kbd>/</kbd>
          </label>
          <div className="library-filter">
            <span>ALL MOVEMENTS</span>
            <span>
              {
                motions.filter((m) =>
                  `${title(m.id)} ${m.category || ""} ${m.id}`.toLowerCase().includes(query.toLowerCase()),
                ).length
              }
            </span>
          </div>
          {motionCards}
          {catalog?.invalid.length ? (
            <div className="invalid-clips">
              {catalog.invalid.length} invalid recording(s) skipped.
              <details>
                <summary>Show details</summary>
                {catalog.invalid.map((m) => (
                  <p key={m.id}>
                    {m.id}: {m.error}
                  </p>
                ))}
              </details>
            </div>
          ) : null}
          <div className="library-tip">
            <Layers3 size={17} />
            <p>
              Your moves. Your sequence.
              <br />
              <span>Drag a clip onto the timeline to begin.</span>
            </p>
          </div>
        </aside>
        <section className="editor">
          <div className="editor-toolbar">
            <div className="tabs">
              <button
                className={tab === "compose" ? "active" : ""}
                onClick={() => setTab("compose")}
              >
                <Layers3 size={14} /> Compose
              </button>
              <button
                className={tab === "physics" ? "active" : ""}
                onClick={() => setTab("physics")}
              >
                <Zap size={14} /> Physics lab
              </button>
            </div>
            <span className="scene-label">
              <span /> Flat ground <ChevronDown size={12} />
            </span>
          </div>
          <RobotViewer model={model} playback={playback} />
          <div className="transport">
            <div className="transport-buttons">
              <button
                className="icon-button"
                disabled={busy || !connected}
                onClick={() => command("reset")}
                title="Reset simulation"
                aria-label="Reset simulation"
              >
                <RotateCcw size={17} />
              </button>
              <button
                className="play-button"
                disabled={
                  busy ||
                  !connected ||
                  (state !== "running" &&
                    (state !== "paused" || playback?.mode === "kinematic_playback") &&
                    (!routine.entries.length || !available || !!trackingError))
                }
                onClick={() =>
                  state === "running"
                    ? command("pause")
                    : state === "paused" && playback?.mode !== "kinematic_playback"
                      ? command("resume")
                      : run()
                }
                aria-label={
                  state === "running"
                    ? "Pause playback"
                    : state === "paused"
                      ? "Resume playback"
                      : "Play routine"
                }
              >
                {busy ? (
                  <LoaderCircle size={17} className="spin" />
                ) : state === "running" ? (
                  <Pause size={17} fill="currentColor" />
                ) : (
                  <Play size={17} fill="currentColor" />
                )}
              </button>
              <button
                className="icon-button"
                disabled={
                  !connected || busy || !["running", "paused"].includes(state)
                }
                onClick={() => command("stop")}
                title="Stop playback"
                aria-label="Stop playback"
              >
                <Square size={15} />
              </button>
            </div>
            <span className="timecode">
              {time(seek ?? playback?.elapsed ?? 0)}{" "}
              <span>/ {time(playback?.total || total)}</span>
            </span>
            <input
              className="playback-scrubber"
              type="range"
              min={0}
              max={playback?.total || 1}
              step={0.05}
              value={seek ?? playback?.elapsed ?? 0}
              disabled={
                busy ||
                !playback?.total ||
                playback.mode !== "kinematic_playback"
              }
              aria-label="Playback position"
              onChange={(e) => setSeek(Number(e.target.value))}
              onPointerUp={(e) => {
                const value = Number(e.currentTarget.value);
                setSeek(null);
                void command("seek", { seconds: value });
              }}
              onPointerCancel={() => setSeek(null)}
              onKeyUp={(e) => {
                if (
                  [
                    "ArrowLeft",
                    "ArrowRight",
                    "ArrowUp",
                    "ArrowDown",
                    "PageUp",
                    "PageDown",
                    "Home",
                    "End",
                  ].includes(e.key)
                ) {
                  setSeek(null);
                  void command("seek", {
                    seconds: Number(e.currentTarget.value),
                  });
                }
              }}
            />
            <span
              className={`play-state ${state === "running" ? "is-running" : ""}`}
            >
              <i />
              {state === "idle" ? "Ready" : state}
            </span>
          </div>
          <div className="timeline">
            <div className="timeline-heading">
              <h2>
                Your sequence{" "}
                <span>
                  {routine.entries.length} clips · {total.toFixed(1)}s
                </span>
              </h2>
              <div className="sequence-actions">
                <button
                  className="button mini primary"
                  disabled={!canTrack}
                  title={
                    trackingError ||
                    "Run the arranged clips through the phase-2 motor controller"
                  }
                  onClick={() => startRoutine("wbc_tracking")}
                >
                  <Zap size={13} /> Run with physics
                </button>
                <button
                  className={`button mini ${routine.loop ? "loop-active" : "quiet"}`}
                  disabled={locked}
                  onClick={() => editRoutine((r) => ({ ...r, loop: !r.loop }))}
                  aria-pressed={routine.loop}
                >
                  <Repeat2 size={14} /> Loop
                </button>
              </div>
            </div>
            {state === "running" && (
              <div className="timeline-edit-hint" role="status">
                <span>Pause playback to arrange your clips.</span>
                <button
                  className="button mini"
                  disabled={busy}
                  onClick={() => command("pause")}
                >
                  <Pause size={12} /> Pause to edit
                </button>
              </div>
            )}
            {state === "paused" && (
              <div className="timeline-edit-hint" role="status">
                Paused — edits restart the updated routine on Play.
              </div>
            )}
            <div className="timeline-ruler">
              <span>START</span>
              <span>BUILD SOMETHING THAT MOVES.</span>
              <span>{time(total)}</span>
            </div>
            <div
              className={`timeline-track ${routine.entries.length ? "" : "empty"} ${dropTarget === "end" ? "drop-end" : ""}`}
              onDragOver={(e) => dragOver(e)}
              onDragLeave={(e) => {
                if (!e.currentTarget.contains(e.relatedTarget as Node | null))
                  setDropTarget(null);
              }}
              onDrop={(e) => drop(e)}
            >
              {routine.entries.map((entry, i) => (
                <div
                  key={entry.id}
                  role="button"
                  tabIndex={0}
                  aria-label={`Select clip ${i + 1}: ${title(entry.motion)}`}
                  aria-pressed={selected === entry.id}
                  className={`timeline-clip ${color(entry.motion)} ${selected === entry.id ? "selected" : ""} ${playback?.entry_id === entry.id ? "playing" : ""} ${!motionMap.has(entry.motion) ? "missing" : ""} ${dropTarget === entry.id ? "drop-target" : ""}`}
                  draggable={!locked}
                  onDragStart={(e) => startDrag(e, "entry", entry.id)}
                  onDragEnd={endDrag}
                  onDragOver={(e) => dragOver(e, entry.id)}
                  onDrop={(e) => drop(e, entry.id)}
                  onClick={() => setSelected(entry.id)}
                  onKeyDown={(e) => {
                    if (e.target !== e.currentTarget) return;
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setSelected(entry.id);
                    }
                  }}
                >
                  <div className="clip-top">
                    <span>{String(i + 1).padStart(2, "0")}</span>
                    <button
                      className="clip-remove"
                      disabled={busy}
                      aria-label={`Remove clip ${i + 1}: ${title(entry.motion)}`}
                      title="Remove clip"
                      onPointerDown={(e) => e.stopPropagation()}
                      onClick={(e) => { e.stopPropagation(); remove(entry.id); }}
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                  <strong>{title(entry.motion)}</strong>
                  <Waveform seed={i} />
                  <div className="clip-bottom">
                    <span>
                      {motionMap.has(entry.motion)
                        ? `${duration(entry).toFixed(1)}s`
                        : "Missing clip"}
                    </span>
                    <span>
                      {entry.speed}×{" "}
                      {entry.repeats > 1 && `· ${entry.repeats} loops`}
                      {!!entry.delay_after && ` · ${entry.delay_after}s hold`}
                    </span>
                  </div>
                </div>
              ))}
              {routine.entries.length ? (
                <div className="timeline-add">
                  <Plus size={21} />
                  <span>Drop a clip</span>
                </div>
              ) : (
                <div className="timeline-empty">
                  <div>
                    <Plus size={24} />
                  </div>
                  <strong>Every routine starts with a move.</strong>
                  <span>Drag a clip here, or use + in the motion library.</span>
                </div>
              )}
            </div>
            <div className="timeline-footer">
              <span>
                <GripVertical size={12} /> Drag to reorder · Select to customise
              </span>
              <span>
                {playback?.mode === "wbc_tracking"
                  ? "Physics tracking"
                  : "Recording playback"}{" "}
                · Continuous clip transitions
              </span>
            </div>
          </div>
        </section>
        <aside className="inspector">
          <div className="panel-heading">
            <h2>{tab === "physics" ? "Physics lab" : "Clip settings"}</h2>
            <SlidersHorizontal size={16} />
          </div>
          {tab === "physics" ? (
            <div className="inspector-content physics-panel">
              <div className="section-icon">
                <Zap size={22} />
              </div>
              <h3>Track your sequence.</h3>
              <p className="inspector-copy">
                The phase-2 ONNX policy drives the simulated motors to follow
                your arranged clips. Gravity and contacts determine the
                movement.
              </p>
              <div className="property">
                <span>Controller</span>
                <strong>Phase-2 WBC</strong>
              </div>
              <div className="property">
                <span>Routine</span>
                <strong>{routine.entries.length} clips</strong>
              </div>
              {trackingError && (
                <p className="capability-note">{trackingError}</p>
              )}
              <button
                className="button primary full"
                disabled={!canTrack}
                onClick={() => startRoutine("wbc_tracking")}
              >
                <Zap size={15} /> Run routine with physics
              </button>
              <p className="tracking-note">
                Keeps position and momentum between clips and repeats. Abrupt moves
                or speed changes can affect balance.
              </p>
              {playback?.mode === "wbc_tracking" && (
                <div
                  className="trial-results"
                  aria-label="Routine physics results"
                  aria-live="off"
                >
                  {playback.phase === "transition_hold" && <p>
                    Transition hold <strong>{(playback.transition_remaining || 0).toFixed(1)}s left</strong>
                  </p>}
                  <span>LIVE TRACKING RESULTS</span>
                  <p>
                    Outcome <strong>{playback.state}</strong>
                  </p>
                  <p>
                    Current clip{" "}
                    <strong>
                      {playback.motion ? title(playback.motion) : "—"}
                    </strong>
                  </p>
                  <p>
                    Repetition <strong>{playback.repeat_index || 1}</strong>
                  </p>
                  <p>
                    Simulated time{" "}
                    <strong>
                      {(playback.simulation_seconds || 0).toFixed(2)} s
                    </strong>
                  </p>
                  <p title="RMS joint angle difference across the run">
                    Tracking RMSE{" "}
                    <strong>
                      {(playback.tracking_rmse_rad || 0).toFixed(3)} rad
                    </strong>
                  </p>
                  <p>
                    Pelvis height{" "}
                    <strong>
                      {(playback.root_height_m || 0).toFixed(3)} m
                    </strong>
                  </p>
                  <p>
                    Clips completed{" "}
                    <strong>{playback.completed_clips || 0}</strong>
                  </p>
                  <p>
                    Detected fall{" "}
                    <strong>{playback.fell ? "Yes" : "No"}</strong>
                  </p>
                  {playback.failure_reason && (
                    <p className="capability-note" role="status">
                      {playback.failure_reason}
                    </p>
                  )}
                  <p className="tracking-note">
                    Completion means the run finished without a detected fall;
                    review tracking error too.
                  </p>
                </div>
              )}
              <details className="standing-trials">
                <summary>Separate stand / walk trials</summary>
                <div className="section-icon">
                  <Zap size={22} />
                </div>
                <h3>Put balance to the test.</h3>
                <p className="inspector-copy">
                  Run an available standing or walking controller with gravity
                  and contact.
                </p>
                <label className="field-label">
                  SKILL
                  <select
                    disabled={locked}
                    value={task}
                    onChange={(e) => setTask(e.target.value)}
                  >
                    <option value="stand">Stand</option>
                    <option value="walk">Walk</option>
                  </select>
                </label>
                <label className="field-label">
                  CONTROLLER
                  <select
                    disabled={locked}
                    value={controller}
                    onChange={(e) => setController(e.target.value)}
                  >
                    <option value="teacher">
                      Teacher
                      {!catalog?.physics.teacher_available
                        ? " — unavailable"
                        : ""}
                    </option>
                    <option value="student">
                      Learned student
                      {!catalog?.physics.student_available
                        ? " — unavailable"
                        : ""}
                    </option>
                  </select>
                </label>
                <label className="field-label">
                  DURATION <span>{seconds}s</span>
                  <input
                    type="range"
                    min={1}
                    max={10}
                    value={seconds}
                    disabled={locked}
                    onChange={(e) => setSeconds(Number(e.target.value))}
                  />
                </label>
                <label className="field-label">
                  PUSH FORCE <span>{push} N</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={push}
                    disabled={locked}
                    onChange={(e) => setPush(Number(e.target.value))}
                  />
                </label>
                {!canPhysics && (
                  <p className="capability-note">
                    Controller assets are unavailable. Prepare the teacher and
                    train a student using the project’s Python setup before
                    running trials.
                  </p>
                )}
                <button
                  className="button primary full"
                  disabled={locked || !connected || !canPhysics}
                  onClick={() =>
                    command("physics", {
                      task,
                      controller,
                      seconds,
                      push_force: push,
                    })
                  }
                >
                  <Zap size={15} /> Run physical trial
                </button>
                {playback?.mode === "motor_driven_physics" && (
                  <div className="trial-results">
                    <span>TRIAL RESULTS</span>
                    <p>
                      Outcome <strong>{playback.state}</strong>
                    </p>
                    <p>
                      Fallen <strong>{playback.fell ? "Yes" : "No"}</strong>
                    </p>
                    <p>
                      Maximum drift{" "}
                      <strong>
                        {(playback.max_drift_m || 0).toFixed(3)} m
                      </strong>
                    </p>
                  </div>
                )}
              </details>
            </div>
          ) : selectedEntry ? (
            <div className="inspector-content">
              <div className={`selected-symbol ${color(selectedEntry.motion)}`}>
                <Activity size={25} />
              </div>
              <span className="eyebrow">
                CLIP{" "}
                {String(
                  routine.entries.findIndex((e) => e.id === selected) + 1,
                ).padStart(2, "0")}
              </span>
              <h3>{title(selectedEntry.motion)}</h3>
              <p className="inspector-copy">
                {motionMap.get(selectedEntry.motion)?.description ||
                  "This recording is not available locally."}
              </p>
              <div className="property">
                <span>Source duration</span>
                <strong>
                  {(
                    motionMap.get(selectedEntry.motion)?.duration_seconds || 0
                  ).toFixed(1)}
                  s
                </strong>
              </div>
              <label className="field-label">
                PLAYBACK SPEED <span>{selectedEntry.speed.toFixed(2)}×</span>
                <input
                  aria-label="Clip playback speed"
                  type="range"
                  min={0.25}
                  max={2}
                  step={0.25}
                  value={selectedEntry.speed}
                  disabled={locked}
                  onChange={(e) =>
                    updateEntry({ speed: Number(e.target.value) })
                  }
                />
                <div className="range-labels">
                  <span>0.25×</span>
                  <span>1×</span>
                  <span>2×</span>
                </div>
              </label>
              <label className="field-label">
                TRANSITION DELAY <span>{(selectedEntry.delay_after || 0).toFixed(1)}s</span>
                <input
                  aria-label="Transition delay"
                  type="range" min={0} max={10} step={0.1}
                  value={selectedEntry.delay_after || 0}
                  disabled={locked}
                  onChange={(e) => updateEntry({ delay_after: Number(e.target.value) })}
                />
              </label>
              <p className="tracking-note">
                Hold the final pose before the next clip or repeat. Physics stays active.
                No delay after the final clip unless Loop is on. This is a hold, not a motion blend.
              </p>
              <label className="field-label">
                REPEAT CLIP
                <div className="stepper">
                  <button
                    aria-label="Decrease repeats"
                    disabled={locked || selectedEntry.repeats <= 1}
                    onClick={() =>
                      updateEntry({ repeats: selectedEntry.repeats - 1 })
                    }
                  >
                    −
                  </button>
                  <span>
                    {selectedEntry.repeats}
                    <small>
                      {" "}
                      {selectedEntry.repeats === 1 ? "time" : "times"}
                    </small>
                  </span>
                  <button
                    aria-label="Increase repeats"
                    disabled={locked || selectedEntry.repeats >= 20}
                    onClick={() =>
                      updateEntry({ repeats: selectedEntry.repeats + 1 })
                    }
                  >
                    +
                  </button>
                </div>
              </label>
              <div className="property duration-property">
                <span>Clip in routine</span>
                <strong>{duration(selectedEntry).toFixed(1)}s</strong>
              </div>
              <div className="arrange-buttons">
                <button
                  className="button quiet"
                  disabled={locked || routine.entries[0].id === selected}
                  onClick={() => {
                    const i = routine.entries.findIndex(
                      (e) => e.id === selected,
                    );
                    reorder(i, i - 1);
                  }}
                >
                  <ArrowLeft size={14} /> Earlier
                </button>
                <button
                  className="button quiet"
                  disabled={locked || routine.entries.at(-1)?.id === selected}
                  onClick={() => {
                    const i = routine.entries.findIndex(
                      (e) => e.id === selected,
                    );
                    reorder(i, i + 1);
                  }}
                >
                  Later <ArrowRight size={14} />
                </button>
              </div>
              <button
                className="button outline full"
                disabled={locked || routine.entries.length >= 100}
                onClick={duplicate}
              >
                <Copy size={14} /> Duplicate clip
              </button>
              <button
                className="button danger full"
                disabled={busy}
                onClick={() => remove()}
              >
                <Trash2 size={14} /> Remove clip
              </button>
            </div>
          ) : (
            <div className="inspector-placeholder">
              <SlidersHorizontal size={29} />
              <h3>Make it your own.</h3>
              <p>
                Select a clip in your sequence to adjust its speed, repeats, and
                position.
              </p>
              <div className="steps-hint">
                <span>
                  01 <b>Choose your moves</b>
                </span>
                <span>
                  02 <b>Arrange your sequence</b>
                </span>
                <span>
                  03 <b>Bring it to life</b>
                </span>
              </div>
            </div>
          )}
          <div className="inspector-bottom">
            <span className="tiny-square" />
            <div>
              SIMULATION ONLY
              <p>
                Motor-driven tracking. Continuous clip transitions; stops on detected falls.
              </p>
            </div>
          </div>
        </aside>
      </main>
      <footer className="statusbar">
        <span>
          <span className="live-dot" /> G1 MOTION STUDIO <i /> LOCAL SESSION
        </span>
        <span>
          Built to move. <ArrowUpRight size={12} />
        </span>
      </footer>
      {notice && (
        <div className="toast" role="status">
          <Check size={16} />
          {notice}
        </div>
      )}
      {library && (
        <div className="modal-backdrop" onClick={() => setLibrary(false)}>
          <section
            ref={modal}
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-label="Saved routines"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="panel-heading">
              <h2>Your routines</h2>
              <button
                autoFocus
                className="icon-button"
                onClick={() => setLibrary(false)}
                aria-label="Close saved routines"
              >
                <X size={19} />
              </button>
            </div>
            <p className="panel-description">
              Saved on this computer. Your current draft is kept in this
              browser.
            </p>
            <div className="modal-actions">
              <button
                className="button outline"
                disabled={locked}
                onClick={() => {
                  editRoutine(
                    () => emptyRoutine,
                    () => {
                      setSavedId(undefined);
                      setSelected(undefined);
                      setLibrary(false);
                    },
                  );
                }}
              >
                <Plus size={15} /> New routine
              </button>
              <button
                className="button quiet"
                disabled={locked}
                onClick={() => importInput.current?.click()}
              >
                <Upload size={15} /> Import JSON
              </button>
            </div>
            <input
              ref={importInput}
              type="file"
              accept=".json,application/json"
              hidden
              onChange={(e) => {
                void importRoutine(e.target.files?.[0]);
                e.target.value = "";
                setLibrary(false);
              }}
            />
            <div className="saved-list">
              {saved.map((r) => (
                <div className="saved-row" key={r.id}>
                  <button
                    disabled={locked}
                    onClick={() => {
                      editRoutine(
                        () => ({
                          name: r.name,
                          entries: r.entries,
                          loop: r.loop,
                        }),
                        () => {
                          setSavedId(r.id);
                          setSelected(undefined);
                          setLibrary(false);
                        },
                      );
                    }}
                  >
                    <Layers3 size={19} />
                    <span>
                      <strong>{r.name}</strong>
                      <small>
                        {r.entries.length} clips ·{" "}
                        {new Date(r.updated).toLocaleDateString()}
                      </small>
                    </span>
                    <ArrowRight size={16} />
                  </button>
                  <button
                    className="icon-button"
                    aria-label={`Delete ${r.name}`}
                    disabled={busy}
                    onClick={() => {
                      if (confirm(`Delete saved routine “${r.name}”?`))
                        void action(async () => {
                          await api(`routines/${r.id}`, undefined, "DELETE");
                          setSaved(await api<SavedRoutine[]>("routines"));
                          if (savedId === r.id) setSavedId(undefined);
                        });
                    }}
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              ))}
              {!saved.length && (
                <p className="empty-library">
                  No saved routines yet. Arrange some moves and hit Save
                  routine.
                </p>
              )}
            </div>
            <button
              className="button quiet full"
              onClick={() => setLibrary(false)}
            >
              Back to the studio <ArrowRight size={14} />
            </button>
          </section>
        </div>
      )}
    </div>
  );
}
