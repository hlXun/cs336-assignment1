import math

def get_lr_cosine_schedule(
    it: int,
    max_lr: float,
    min_lr: float,
    warmup_it: int,
    cosine_cycle_it: int,
) -> float:
    assert cosine_cycle_it > warmup_it, "cosine_cycle_it > warmup_it"
    if it > cosine_cycle_it:
        return min_lr

    if it < warmup_it:
        return max_lr * it / warmup_it
    
    cosine = math.cos((it - warmup_it) / (cosine_cycle_it - warmup_it) * math.pi)
    lr = min_lr + 0.5 * (1 + cosine) * (max_lr - min_lr)
    return lr