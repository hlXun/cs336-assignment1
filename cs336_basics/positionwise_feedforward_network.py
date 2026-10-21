import torch
import torch.nn as nn
from cs336_basics.linear import Linear

class positionwise_FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ffn1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.ffn2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.ffn3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x1 = self.SiLU(self.ffn1(x)) # [... d_ff]
        x2 = self.ffn3(x) # [... d_ff]
        out = self.ffn2(x1 * x2) # [... d_model]
        return out
    
    def SiLU(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        out = x * torch.sigmoid(x)
        return out