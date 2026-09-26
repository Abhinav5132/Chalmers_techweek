from __future__ import annotations
import torch
import hashlib
import json
TensorDict = dict

def register_network(*args, **kwargs):
    return lambda cls: cls

def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    """Resolve the activation function from the name.

    Args:
        act_name: Name of the activation function.

    Returns:
        The activation function.

    Raises:
        ValueError: If the activation function is not found.
    """
    act_dict = {
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "relu": torch.nn.ReLU(),
        "crelu": torch.nn.CELU(),
        "lrelu": torch.nn.LeakyReLU(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "softplus": torch.nn.Softplus(),
        "gelu": torch.nn.GELU(),
        "swish": torch.nn.SiLU(),
        "silu": torch.nn.SiLU(),
        "mish": torch.nn.Mish(),
        "identity": torch.nn.Identity(),
    }

    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    else:
        raise ValueError(f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}")


def build_obs_schema(obs: TensorDict, obs_groups: dict[str, list[str]] | None = None) -> dict[str, object]:
    """Build a batch-size-invariant observation schema summary with a stable hash."""

    def _sample_shape(value: object) -> list[int] | None:
        if not isinstance(value, torch.Tensor):
            return None
        if value.ndim == 0:
            return []
        if value.ndim == 1:
            return [int(value.shape[0])]
        return [int(dim) for dim in value.shape[1:]]

    available_obs: dict[str, dict[str, object]] = {}
    for key in obs.keys():
        value = obs[key]
        sample_shape = _sample_shape(value)
        if sample_shape is None:
            continue
        available_obs[str(key)] = {
            "shape": sample_shape,
            "dtype": str(value.dtype),
        }

    schema: dict[str, object] = {"available_obs": available_obs}
    if obs_groups is not None:
        schema["obs_groups"] = {set_name: [str(group) for group in groups] for set_name, groups in obs_groups.items()}

    payload = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    schema["hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return schema
