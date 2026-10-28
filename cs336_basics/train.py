from cs336_basics.transformer_lm import Transformer_LM
from cs336_basics.loss import cross_entropy
from cs336_basics.adamw import AdamW
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_cosine_schedule import get_lr_cosine_schedule
from cs336_basics.data_loading import data_loading
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
import torch
import numpy as np
import pickle

train_dataset_path = './data/tinystories_train.npy'
valid_dataset_path = './data/tinystories_valid.npy'
train_dataset = np.load(train_dataset_path, mmap_mode='r')
valid_dataset = np.load(valid_dataset_path, mmap_mode='r')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_size = 10000
d_model = 512
d_ff = 1344
num_heads = 16
context_length = 256
theta = 10000
num_layers = 4
model = Transformer_LM(
    vocab_size=vocab_size,
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads,
    context_length=1000,
    theta=theta,
    num_layers=num_layers,
    device=device,
    dtype=torch.float,
)

betas = (0.9, 0.999)
weight_decay = 0.1
opt = AdamW(
    params=model.parameters(),
    betas=betas,
    weight_decay=weight_decay,
)

batch_size = 1
max_l2_norm = 1.0
max_lr = 0.001
min_lr = 0.0001
warmup_it = 10
cosine_cycle_it = 40
checkpoint_file = './cs336_basics/check_points'
train_steps = 100
start_step = 0
checkpoint = None
if checkpoint is not None:
    start_step = load_checkpoint(checkpoint, model=model, optimizer=opt)

model.train()
for train_step in range(start_step, train_steps):
    input_seqs, target = data_loading(train_dataset, batch_size, context_length, device)
    opt.zero_grad()
    out: torch.Tensor = model(input_seqs) # [batch_size, context_length, vocab_size]
    loss = cross_entropy(out, target)
    loss.backward()
    gradient_clipping(model.parameters(), max_l2_norm=max_l2_norm)
    for group in opt.param_groups:
        group["lr"] = get_lr_cosine_schedule(
            it=train_step,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_it=warmup_it,
            cosine_cycle_it=cosine_cycle_it,
        )
    opt.step()
    
    if train_step % 10 == 0 or train_step == train_steps - 1:
        print(f"Train: step:{train_step}: loss:{loss:.2f}")
        out = checkpoint_file + f'/checkpoint{train_step}.pt'
        if train_step == train_steps - 1:
            out = checkpoint_file + '/trained.pt'
        save_checkpoint(
            model=model,
            optimizer=opt,
            iteration=train_step+1,
            out=out,
        )
        model.eval()
        with torch.no_grad():
            input_seqs, target = data_loading(
                valid_dataset, 
                5, 
                context_length, 
                device,
            )
            out = model(input_seqs)
            loss = cross_entropy(out, target)
        print(f"Valid: step:{train_step}: loss:{loss:.2f}")
        model.train()
            