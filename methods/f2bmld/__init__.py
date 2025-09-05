from typing import Dict, Any, Tuple
import torch.nn as nn

from .bsuite_network import make_network_bsuite
from .learner import F2BMLDLearner


def make_ope_networks(
    task_id: str,
    environment_spec: Dict[str, Any],
    treatment_layer_sizes: Dict[str, Any],
    instrument_layer_sizes: Dict[str, Any],
    device: str = "cpu"
) -> Tuple[nn.Module, nn.Module]:
    if task_id.startswith("bsuite"):
        treatment_net, instrumental_net, instrument_tilde_net = make_network_bsuite(
            environment_spec, treatment_layer_sizes, instrument_layer_sizes, device
        )
    else:
        raise ValueError(f"task id {task_id} not supported (only bsuite in Torch version).")

    return treatment_net, instrumental_net, instrument_tilde_net