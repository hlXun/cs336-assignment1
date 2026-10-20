import torch
import torch.nn
import math

class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        origin_type = x.dtype
        x = x.float()
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt() # [batch_size, sequence_length, 1]
        out = x / rms * self.g
        out = out.to(dtype=origin_type) 
        return out