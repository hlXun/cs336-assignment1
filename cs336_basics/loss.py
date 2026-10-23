import torch

def cross_entropy(
    x: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    x = x.float()
    max_x = x.amax(dim=-1, keepdim=True) # [... 1]
    x = x - max_x # [... vocab_size]
    x1 = x.exp().sum(dim=-1, keepdim=True).log() # [... 1]
    x2 = x.gather(dim=-1, index=target.unsqueeze(-1)) # [... 1]
    x_corss_entropy = x1 - x2 # [... 1]
    out = x_corss_entropy.sum() / x_corss_entropy.numel() # () 零维标量
    return out
    