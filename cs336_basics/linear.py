import torch
import torch.nn
from einops import rearrange, einsum
import math

class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        std = math.sqrt(2.0 / (in_features + out_features))
        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype),
                mean=0,
                std=std,
                a=-3*std,
                b=3*std,
            )
        )
        self.device = device
        self.dtype = dtype
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        result = einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
        return result