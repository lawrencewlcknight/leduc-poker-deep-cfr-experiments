"""Neural network components used by the Deep CFR solver.

These layers preserve the truncated-normal initialisation of the original
notebook (a Sonnet-style linear layer with stddev = 1/sqrt(fan_in)) so that
results stay comparable across the notebook and the packaged code.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn


class SonnetLinear(nn.Module):
    """A Sonnet-style linear layer with optional ReLU activation."""

    def __init__(self, in_size: int, out_size: int, activate_relu: bool = True) -> None:
        super().__init__()
        self._activate_relu = bool(activate_relu)
        self._in_size = int(in_size)
        self._out_size = int(out_size)
        # Allocated by ``reset``; declared up front so static analysers are happy.
        self._weight: nn.Parameter = nn.Parameter(
            torch.empty(self._out_size, self._in_size), requires_grad=True
        )
        self._bias: nn.Parameter = nn.Parameter(
            torch.zeros(self._out_size), requires_grad=True
        )
        self.reset()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        y = F.linear(tensor, self._weight, self._bias)
        return F.relu(y) if self._activate_relu else y

    def reset(self) -> None:
        stddev = 1.0 / math.sqrt(self._in_size)
        # Truncated to ±2 standard deviations.
        sampled = stats.truncnorm.rvs(
            -2.0, 2.0, loc=0.0, scale=stddev, size=(self._out_size, self._in_size)
        )
        with torch.no_grad():
            self._weight.copy_(torch.from_numpy(sampled).to(self._weight.dtype))
            self._bias.zero_()


class MLP(nn.Module):
    """A simple feed-forward MLP with ``reset`` propagated to every layer."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        activate_final: bool = False,
    ) -> None:
        super().__init__()
        layers: List[SonnetLinear] = []
        in_size = int(input_size)
        for size in hidden_sizes:
            layers.append(SonnetLinear(in_size=in_size, out_size=int(size)))
            in_size = int(size)
        layers.append(
            SonnetLinear(
                in_size=in_size,
                out_size=int(output_size),
                activate_relu=bool(activate_final),
            )
        )
        self.model = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model:
            x = layer(x)
        return x

    def reset(self) -> None:
        for layer in self.model:
            layer.reset()
