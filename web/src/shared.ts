import { z } from "zod";
export const entrySchema = z.object({
  id: z.string().uuid(),
  motion: z
    .string()
    .min(1)
    .max(100)
    .regex(/^[a-zA-Z0-9_-]+$/),
  speed: z.number().min(0.25).max(2),
  repeats: z.number().int().min(1).max(20),
  delay_after: z.number().min(0).max(10).optional(),
});
export const routineSchema = z
  .object({
    name: z.string().trim().min(1).max(80),
    entries: z.array(entrySchema).max(100),
    loop: z.boolean(),
  })
  .refine(
    (r) => new Set(r.entries.map((e) => e.id)).size === r.entries.length,
    "Clip instance IDs must be unique",
  );
export type Entry = z.infer<typeof entrySchema>;
export type Routine = z.infer<typeof routineSchema>;
export type SavedRoutine = Routine & { id: string; updated: string };
export type Motion = {
  id: string;
  description: string;
  frames: number;
  duration_seconds: number;
  tracking_available?: boolean;
  tracking_error?: string | null;
};
export type Catalog = {
  motions: Motion[];
  invalid: { id: string; error: string }[];
  tracking?: {
    available: boolean;
    error?: string | null;
    controller: string;
    transitions: string;
  };
  physics: {
    teacher_available: boolean;
    student_available: boolean;
    error?: string;
  };
};
export type Playback = {
  phase?: "clip" | "transition_hold";
  transition_remaining?: number;
  state:
    | "idle"
    | "running"
    | "paused"
    | "stopped"
    | "completed"
    | "failed"
    | "fallen";
  mode: "kinematic_playback" | "motor_driven_physics" | "wbc_tracking";
  model_id?: "g1" | "wbc";
  tracking_rmse_rad?: number;
  tracking_error_rad?: number;
  max_tracking_rmse_rad?: number;
  root_height_m?: number;
  reset_count?: number;
  repeat_index?: number;
  completed_clips?: number;
  simulation_seconds?: number;
  failure_reason?: string;
  elapsed: number;
  total: number;
  index?: number;
  entry_id?: string;
  motion?: string;
  error?: string;
  positions: number[];
  quaternions: number[];
  fell?: boolean;
  max_drift_m?: number;
};
export type RobotModel = {
  id?: "g1" | "wbc";
  root_body?: number;
  meshes: { vertices: number[]; indices: number[] }[];
  geoms: {
    body: number;
    type: number;
    mesh: number;
    size: number[];
    position: number[];
    quaternion: number[];
    color: number[];
  }[];
};
