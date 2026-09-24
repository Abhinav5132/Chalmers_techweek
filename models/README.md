# Trained WBC Models Directory

This directory stores trained Reinforcement Learning tracking policies exported from `wbc-mjlab`.

### Teammate Handoff:
When you finish training a policy in `wbc-mjlab` (e.g., imitating Blender or MoCap animations):
1. Export the ONNX model (or PyTorch checkpoint):
   * Recommended file path: `models/wbc_policy.onnx`
   * Associated config: `models/config.yaml`
2. Once placed here, the `WbcPhysicsRunner` in `src/controllers/wbc_runner.py` and the workflow scripting engine will automatically detect and load it for closed-loop physics tracking in MuJoCo.
