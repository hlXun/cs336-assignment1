import torch
import torch.nn as nn
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_selfattentnion_with_rope import Multihead_SelfAttention_with_RoPE
from cs336_basics.positionwise_feedforward_network import positionwise_FFN

class Transformer_Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.RMSNorm1 = RMSNorm(self.d_model, device=device, dtype=dtype)
        self.Attention = Multihead_SelfAttention_with_RoPE(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.RMSNorm2 = RMSNorm(self.d_model, device=device, dtype=dtype)
        self.ffn = positionwise_FFN(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.Attention(self.RMSNorm1(x), token_positions)
        x = x + self.ffn(self.RMSNorm2(x))
        return x