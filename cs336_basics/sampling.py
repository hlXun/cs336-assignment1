import torch
from cs336_basics.softmax import softmax
from einops import rearrange

def sampling(
    x: torch.Tensor,
    temperature: float = 1.0,
    p: float = 0.8,
) -> torch.Tensor:
    x = x / temperature # [batch_size, context_length, vocab_size]
    q = softmax(x, dim=-1) # [batch_size, context_length, vocab_size]
    sorted_q, idx = torch.sort(q, dim=-1, descending=True) # [batch_size, context_length, vocab_size]
    cum_q = torch.cumsum(sorted_q, dim=-1) # [batch_size, context_length, vocab_size]
    is_picked = (cum_q - sorted_q) < p # [batch_size, context_length, vocab_size]
    sorted_q.masked_fill_(is_picked==False, 0.0) # [batch_size, context_length, vocab_size]
    q = sorted_q / sorted_q.sum(dim=-1, keepdim=True) # [batch_size, context_length, vocab_size]
    q = rearrange(q, "batch_size context_length vocab_size -> (batch_size context_length) vocab_size")
    sampling_idx = torch.multinomial(q, num_samples=1) # [batch_size * context_length, 1]
    sampling_idx = rearrange(sampling_idx, "(batch_size context_length) one -> batch_size context_length one", context_length=x.shape[-2])
    sampling_q_idx = idx.gather(dim=-1, index=sampling_idx) # [batch_size, context_length, 1]
    sampling_q_idx.squeeze_(dim=-1) # [batch_size, context_length]
    return sampling_q_idx