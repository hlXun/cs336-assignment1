import torch
import torch.nn as nn
from cs336_basics.softmax import softmax
from einops import einsum, rearrange
import math

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    K_T = rearrange(K, "... keys d_k -> ... d_k keys")
    A = einsum(Q, K_T, "... queries d_k, ... d_k keys -> ... queries keys")
    if mask is not None:
        A = A.masked_fill(mask == False, float('-inf'))
    A = softmax(A / math.sqrt(Q.shape[-1]), dim=-1)
    out = einsum(A, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return out
    