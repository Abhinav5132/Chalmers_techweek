# PMT inference components

Source: https://github.com/Mondo-Robotics/PMT
Commit: 1a92390077d329d22cbcdbf89c4a5c982003fbc8

The transformer, MLP, normalization and two utility functions are vendored from
motion_tracking_rl under their BSD-3-Clause headers (ETH Zurich/NVIDIA and
Preferred Networks). Changes only redirect imports to this package, replace the
TensorDict annotation with dict, and make the network registration decorator an
identity. No Isaac Lab, training runners, or BFM-Zero code is included.
The original mathematical inference implementation is retained.
