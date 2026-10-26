import torch
import math
from collections.abc import Iterable

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    eps = 1e-6
    parameters = list(parameters)
    l2_norm = torch.tensor(0.0, device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        l2_norm += parameter.grad.data.pow(2).sum()
    l2_norm.sqrt_()
    if l2_norm < max_l2_norm:
        return None
    factor = max_l2_norm / (l2_norm + eps)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter.grad.data *= factor