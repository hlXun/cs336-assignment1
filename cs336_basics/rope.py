import torch
import torch.nn as nn
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
        
        pos = torch.arange(0, max_seq_len, device=device) # [max_seq_len]
        freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=device).float() / d_k)) # [d_k // 2]
        R = einsum(pos, freq, "pos, freq -> pos freq") # [max_seq_len, d_k // 2]
        self.register_buffer("R_sin", torch.sin(R), persistent=False)
        self.register_buffer("R_cos", torch.cos(R), persistent=False)
        
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
            