import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        R = torch.empty(max_seq_len, d_k // 2, device=device) # [max_seq_len, d_k // 2]
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                R[i][k] = i / math.pow(theta, (2 * k) / d_k)
        self.register_buffer("R_cos", torch.cos(R))
        self.register_buffer("R_sin", torch.sin(R))
        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        g = rearrange(x, "... (groups two) -> ... groups two", two=2)
        r_cos = self.R_cos[token_positions] # [..., seq_len, d_k // 2]
        r_sin = self.R_sin[token_positions] # [..., seq_len, d_k // 2]
        x1, x2 = g[..., 0], g[..., 1]
        out = torch.stack([x1 * r_cos - x2 * r_sin, x1 * r_sin + x2 * r_cos], dim=-1)
        out = rearrange(out, "... groups two -> ... (groups two)", two=2)
        return out
            