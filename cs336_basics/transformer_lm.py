import torch
import torch.nn as nn
from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import Transformer_Block
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear
from cs336_basics.softmax import softmax

class Transformer_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        context_length: int,
        theta: float,
        num_layers: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.Token_Embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.Transformer_Blocks = nn.ModuleList(
            [
                Transformer_Block(d_model, d_ff, num_heads, context_length, theta, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.RMSNorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.LM_Head = Linear(d_model, vocab_size, device=device, dtype= dtype)
        self.num_layers = num_layers
        
    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        x = self.Token_Embedding(input)
        for i in range(self.num_layers):
            x = self.Transformer_Blocks[i](x)
        x = self.RMSNorm(x)
        out = self.LM_Head(x)
        return out
    