from cs336_basics.transformer_lm import Transformer_LM
from cs336_basics.loss import cross_entropy
from cs336_basics.adamw import AdamW
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_cosine_schedule import get_lr_cosine_schedule
from cs336_basics.data_loading import data_loading
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint

import torch
import numpy as np
import csv
import json
import math
from pathlib import Path

# ---------- 目录与日志 ----------
log_dir = Path('./cs336_basics/experiments')
ckpt_dir = Path('./cs336_basics/check_points')
log_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / 'train_log_1.csv'

# ---------- 数据 ----------
train_dataset_path = './data/tinystories_train.npy'
valid_dataset_path = './data/tinystories_valid.npy'
train_dataset = np.load(train_dataset_path, mmap_mode='r')
valid_dataset = np.load(valid_dataset_path, mmap_mode='r')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 超参 ----------
vocab_size     = 10000
d_model        = 512
d_ff           = 1344
num_heads      = 16
context_length = 256
theta          = 10000
num_layers     = 4

betas          = (0.9, 0.999)
weight_decay   = 0.1
batch_size     = 1
max_l2_norm    = 1.0
max_lr         = 0.001
min_lr         = 0.0001
warmup_it      = 10
cosine_cycle_it= 40
train_steps    = 100

config = dict(
    vocab_size=vocab_size, d_model=d_model, d_ff=d_ff,
    num_heads=num_heads, context_length=context_length,
    theta=theta, num_layers=num_layers,
    betas=betas, weight_decay=weight_decay, batch_size=batch_size,
    max_l2_norm=max_l2_norm, max_lr=max_lr, min_lr=min_lr,
    warmup_it=warmup_it, cosine_cycle_it=cosine_cycle_it,
    train_steps=train_steps, device=str(device),
)
with open(log_dir / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)

# ---------- 模型 / 优化器 ----------
model = Transformer_LM(
    vocab_size=vocab_size,
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads,
    context_length=context_length,  
    theta=theta,
    num_layers=num_layers,
    device=device,
    dtype=torch.float,
)
opt = AdamW(params=model.parameters(), betas=betas, weight_decay=weight_decay)

start_step = 0
checkpoint = None
if checkpoint is not None:
    start_step = load_checkpoint(checkpoint, model=model, optimizer=opt)

# ---------- 训练 + 记录 ----------
with open(log_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'train_loss', 'valid_loss',
                     'train_ppl', 'valid_ppl', 'lr', 'checkpoint'])

    model.train()
    for train_step in range(start_step, train_steps):
        input_seqs, target = data_loading(
            train_dataset, batch_size, context_length, device)
        opt.zero_grad()
        logits = model(input_seqs)
        loss = cross_entropy(logits, target)
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=max_l2_norm)

        current_lr = None
        for group in opt.param_groups:
            group["lr"] = get_lr_cosine_schedule(
                it=train_step,
                max_lr=max_lr,
                min_lr=min_lr,
                warmup_it=warmup_it,
                cosine_cycle_it=cosine_cycle_it,
            )
            current_lr = group["lr"]
        opt.step()

        if train_step % 10 == 0 or train_step == train_steps - 1:
            train_loss = loss.item()

            ckpt_name = f'checkpoint{train_step}.pt'
            if train_step == train_steps - 1:
                ckpt_name = 'trained.pt'
            ckpt_path = ckpt_dir / ckpt_name

            save_checkpoint(
                model=model,
                optimizer=opt,
                iteration=train_step + 1,
                out=str(ckpt_path),
            )

            model.eval()
            with torch.no_grad():
                val_in, val_tgt = data_loading(
                    valid_dataset, 5, context_length, device)
                val_logits = model(val_in)               
                valid_loss = cross_entropy(val_logits, val_tgt).item()
            model.train()

            writer.writerow([
                train_step,
                train_loss,
                valid_loss,
                math.exp(train_loss),
                math.exp(valid_loss),
                current_lr,
                str(ckpt_path),
            ])
            f.flush()

            print(f"step {train_step:4d} | "
                  f"train {train_loss:.4f} | "
                  f"valid {valid_loss:.4f} | "
                  f"lr {current_lr:.2e}")