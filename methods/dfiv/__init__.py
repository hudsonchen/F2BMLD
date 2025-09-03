from typing import Dict, Any, Tuple

import torch.nn as nn

from .bsuite_network import make_value_func_bsuite
from .learner import DFIVLearner


def make_ope_networks(
    task_id: str,
    environment_spec: Dict[str, Any],
    value_layer_sizes: Dict[str, Any],
    instrumental_layer_sizes: Dict[str, Any],
    device: str = "cpu"
) -> Tuple[nn.Module, nn.Module]:
    if task_id.startswith("bsuite"):
        value_func, instrumental_feature = make_value_func_bsuite(
            environment_spec, value_layer_sizes, instrumental_layer_sizes, device
        )
    else:
        raise ValueError(f"task id {task_id} not supported (only bsuite in Torch version).")

    # In PyTorch we don't need tf2_utils.create_variables;
    # networks are lazily created on first forward pass.

    return value_func, instrumental_feature