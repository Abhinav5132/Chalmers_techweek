# Phase-2 physics tracking bundle

Updated from `origin/Phase-2-Physics` commit
`601e9a81ac9e10a0d840e666041ad53482b9c30f`.

- `params/policy.onnx`: the branch's exported WBC tracking policy, unchanged.
- `params/config.yaml`: policy observation layout, PD gains, action scaling and
  20 ms policy period, unchanged.
- `params/robot_train/scene.xml` and referenced assets: the matching exported
  MuJoCo training scene, with gravity, a free base, contacts and actuators.
- `params/motion_library.yaml`: original export provenance; its upstream absolute
  dataset path is informational. The web app uses local `data/motions/*.npz`.

Only inference assets are included. The GPU training environment and its vendor
repository are not needed to run this policy. Install `uv sync --group tracking`
(or include `--group training` to retain the other project training dependencies).

`src/controllers/wbc_runner.py` and `clip_resample.py` were brought over from the
same commit. Integration changes validate the bundle/recordings, resolve paths
independently of working directory, retime references and velocities for clip
speed, initialize recorded joint velocities, reset all MuJoCo data at routine start,
and retain a body-frame angular-velocity fallback (MuJoCo `mjOBJ_XBODY`).
The latest upstream retrained policy, raw previous-action observations, IMU gyro
observations, waist-yaw torque envelope, and optional yaw anchoring are included.
The web sequencer uses the branch workflow engine’s first-clip-only reset and
anchored transitions through a shared `set_clip` path for preloaded references.

`src/wbc_session.py` adds sequencing, repeats, loop handling, trial statistics and
fall detection. Clip/repeat/loop transitions preserve the physical state and
action history. New references are aligned to current horizontal position and yaw.
Per-clip `delay_after` follows the branch’s timed-hold concept. The web runtime
uses WBC final-pose tracking with zero reference base velocities during the hold,
rather than importing the separate Pink nominal-stance controller. Physics keeps
stepping and fall detection remains active. Targets are not blended; this is not a guarantee of balance for arbitrary clip
pairs, and it does not train a new policy.

The fall detector stops at pelvis height < 0.25 m or pelvis upright cosine < 0.2
(about 78 degrees tilt). This is a heuristic; intentional acrobatics can trigger it.
Completion alone is not proof of accurate tracking or real-hardware reliability.
