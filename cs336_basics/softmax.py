import torch

def softmax(
    x: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    max_v = x.amax(dim=dim, keepdim=True) # [..., 1, ...]
    x_exp = (x - max_v).exp()
    out = x_exp / x_exp.sum(dim=dim, keepdim=True)
    return out