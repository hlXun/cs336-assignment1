import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.linear import Linear

class Multihead_SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = rearrange(q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        k = rearrange(k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        v = rearrange(v, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        
        mask = torch.ones(q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)
        mask = torch.tril(mask, diagonal=0)
        
        o = scaled_dot_product_attention(q, k, v, mask=mask) # [... num_heads, seq_len, d_v]
        o = rearrange(o, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        out = self.o_proj(o)
        return out