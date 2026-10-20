import torch
import torch.nn
import math

class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    num_embeddings,
                    embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
                mean = 0,
                std = 1,
                a = -3,
                b = 3,
            )
        )
        self.device = device
        self.dtype = dtype
    
    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        token_ids = token_ids.long()
        out = self.W[token_ids]
        return out