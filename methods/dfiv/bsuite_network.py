import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple


def one_hot(actions: torch.Tensor, depth: int) -> torch.Tensor:
    """One-hot encode integer actions."""
    return F.one_hot(actions.long(), num_classes=depth).float()


def add_const_col(x: torch.Tensor) -> torch.Tensor:
    """Add a constant 1 column (bias feature)."""
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    return torch.cat([x, ones], dim=-1)


class CriticMultiplexer(nn.Module):
    """Concatenate observation and action features."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        # obs: [B, obs_dim], act: [B] (discrete int) or [B, act_dim] if already one-hot
        if act.ndim == 1:
            act_onehot = one_hot(act, self.act_dim)
        else:
            act_onehot = act
        return torch.cat([obs, act_onehot], dim=-1)


class MLP(nn.Module):
    """Simple MLP with ReLU activations."""

    def __init__(self, layer_sizes: Sequence[int], activate_final: bool = True):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2 or activate_final:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InstrumentalFeature(nn.Module):
    def __init__(self, environment_spec: dict, layer_sizes: Sequence[int]):
        super().__init__()
        obs_dim, act_dim = environment_spec["obs_dim"], environment_spec["act_dim"]
        self._net = nn.Sequential(
            CriticMultiplexer(obs_dim, act_dim),
            MLP([obs_dim + act_dim] + list(layer_sizes), activate_final=True)
        )
        self._feature_dim = layer_sizes[-1]

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        feat = self._net[0](obs, act)  # CriticMultiplexer
        feat = self._net[1](feat)      # MLP
        # feat = add_const_col(feat)
        return feat

    def feature_dim(self) -> int:
        return self._feature_dim


class ValueFeature(nn.Module):
    def __init__(self, environment_spec: dict, layer_sizes: Sequence[int]):
        super().__init__()
        obs_dim, act_dim = environment_spec["obs_dim"], environment_spec["act_dim"]
        self._net = nn.Sequential(
            CriticMultiplexer(obs_dim, act_dim),
            MLP([obs_dim + act_dim] + list(layer_sizes), activate_final=True)
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        feat = self._net[0](obs, act)
        return self._net[1](feat)


class ValueFunction(nn.Module):
    def __init__(self, environment_spec: dict, layer_sizes: Sequence[int]):
        super().__init__()
        self._feature = ValueFeature(environment_spec, layer_sizes)
        feat_dim = layer_sizes[-1]
        self._weight = nn.Parameter(torch.zeros(feat_dim, 1))

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        feat = self._feature(obs, act)
        # feat = add_const_col(feat)
        return feat @ self._weight

    def feature_dim(self) -> int:
        return self._weight.shape[0]

    @property
    def weight(self):
        return self._weight


def make_value_func_bsuite(
    environment_spec: dict,
    value_layer_sizes: str = "50,50",
    instrumental_layer_sizes: str = "50,50",
    device: str = "cpu"
) -> Tuple[nn.Module, nn.Module]:
    value_sizes = list(map(int, value_layer_sizes.split(",")))
    instr_sizes = list(map(int, instrumental_layer_sizes.split(",")))

    value_function = ValueFunction(environment_spec, layer_sizes=value_sizes).to(device)
    instrumental_feature = InstrumentalFeature(environment_spec, layer_sizes=instr_sizes).to(device)

    return value_function, instrumental_feature
