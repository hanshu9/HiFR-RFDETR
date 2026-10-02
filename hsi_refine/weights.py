"""Component weight loading for RF-DETR and the spectral adapter."""
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch


def load_checkpoint(path):
    # RF-DETR weights can include configuration namespaces alongside tensors.
    with torch.serialization.safe_globals([Namespace, SimpleNamespace]):
        return torch.load(Path(path), map_location="cpu", weights_only=True)


def extract_state(checkpoint, key="model", prefix=""):
    state = checkpoint if key == "" else checkpoint[key]
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Selected checkpoint key is not a tensor state_dict")
    if prefix:
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not state:
            raise ValueError(f"No checkpoint keys start with {prefix!r}")
    return state
