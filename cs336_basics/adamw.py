import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.1,
        eps: float = 1e-8,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)
    
    def step(
        self,
    ) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p, device=p.device))
                v = state.get("v", torch.zeros_like(p, device=p.device))
                t = state.get("t", 1)
                grad = p.grad.data
                lr_t = lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                p.data -= lr * weight_decay * p.data
                state["m"] = m = beta1 * m + (1 - beta1) * grad
                state["v"] = v = beta2 * v + (1 - beta2) * grad * grad
                state["t"] = t + 1
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                