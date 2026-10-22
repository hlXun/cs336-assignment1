import torch
import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from einops import einsum, rearrange

class Multihead_SelfAttention_with_RoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int, 
        theta: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.qkv_proj = Linear(d_model, 3 * d_model)
        self.o_proj = Linear(d_model, d_model)
        self.RoPE = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, [self.d_model] * 3, dim=-1)
        
        q = rearrange(q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        k = rearrange(k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        v = rearrange(v, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        
        q = self.RoPE(q, token_positions)
        k = self.RoPE(k, token_positions)
        
        mask = torch.ones(q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)
        mask = torch.tril(mask, diagonal=0)
        
        o = scaled_dot_product_attention(q, k, v, mask=mask) # [... num_heads, seq_len, d_v]
        o = rearrange(o, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        out = self.o_proj(o)
        return out
        