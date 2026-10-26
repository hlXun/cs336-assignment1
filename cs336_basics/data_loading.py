import torch
import random
import numpy as np
import numpy.typing as npt

def data_loading(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    input_seqs_np = np.empty((batch_size, context_length), dtype=dataset.dtype)
    label_seqs_np = np.empty((batch_size, context_length), dtype=dataset.dtype)
    for i in range(batch_size):
        start = random.randint(0, dataset.size - 1 - context_length)
        end = start + context_length
        input_seqs_np[i] = dataset[start: end]
        label_seqs_np[i] = dataset[start + 1: end + 1]
    input_seqs = torch.from_numpy(input_seqs_np).to(device=device)
    label_seqs = torch.from_numpy(label_seqs_np).to(device=device)
    return input_seqs, label_seqs