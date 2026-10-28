import torch
from cs336_basics.sampling import sampling
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import Transformer_LM
import pickle

@torch.no_grad()
def decode(
    model: Transformer_LM,
    x: torch.Tensor,
    tokenizer: Tokenizer,
    temperature: float = 1.0,
    p: float = 0.8,
    max_num_tokens: int = 1000,
    end_token: int = 256,
) -> str:
    num_completion_token = 0
    token_ids = []
    model.eval()
    while num_completion_token < max_num_tokens:
        if x.shape[-1] > model.Transformer_Blocks[0].Attention.RoPE.max_seq_len:
            break
        y = model(x) # [1, context_length, vocab_size]，先固定batch_size为1
        sampling_id = sampling(y[..., -1:, :], temperature, p) # [1, context_length]
        token_id = sampling_id[..., -1].item()
        if token_id == end_token:
            break
        token_ids.append(token_id)
        x = torch.cat([x, sampling_id[..., -1:]], dim=-1)
        num_completion_token += 1
    result = tokenizer.decode(token_ids)
    return result