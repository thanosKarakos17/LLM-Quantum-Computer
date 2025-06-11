import os
import time
import math
import pickle
import numpy as np
import torch
from contextlib import nullcontext
#from config.train_maxcut import *

from model import GPT, GPTConfig

# ---------------------------------------
# Config overrides (read from config/*.py)
# ---------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# Set up device and dtype
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.manual_seed(1337)
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------
# Load Data
# ---------------------------------------
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
#block_size = meta['block_size']
print(f"Data: vocab_size={vocab_size}, block_size={block_size}")

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
train_feathers = np.load(os.path.join(data_dir, 'train_feathers.npy'))
val_feathers = np.load(os.path.join(data_dir, 'val_feathers.npy'))

def get_batch(split):
    data = train_data if split == 'train' else val_data
    feathers = train_feathers if split == 'train' else val_feathers

    num_samples = len(data) // block_size
    replace_flag = num_samples < batch_size
    ixs = np.random.choice(num_samples, size=batch_size, replace=replace_flag)

    x = torch.stack([torch.from_numpy(data[i * block_size: (i+1)*block_size].astype(np.int64)) for i in ixs])
    y = torch.stack([torch.from_numpy(data[i * block_size + 1: (i+1)*block_size + 1].astype(np.int64)) for i in ixs])
    f = torch.tensor(feathers[ixs], dtype=ptdtype)

    return x.to(device), y.to(device), f.to(device)

# ---------------------------------------
# Initialize Model
# ---------------------------------------
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=vocab_size,
    feather_dim=feather_dim,
    dropout=0.0,
)

model = GPT(GPTConfig(**model_args))
model.to(device)

optimizer = model.configure_optimizers(weight_decay, learning_rate, (0.9, 0.95), device_type)

# ---------------------------------------
# Training Loop
# ---------------------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            xb, yb, fb = get_batch(split)
            with ctx:
                _, loss = model(xb, yb, fb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

print(f"Training with device: {device}, dtype: {dtype}")
iter_num = 0
best_val_loss = 1e9

X, Y, F = get_batch('train')
t0 = time.time()

while iter_num <= max_iters:
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter_num}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # Train step
    X, Y, F = get_batch('train')
    with ctx:
        logits, loss = model(X, Y, F)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter_num % log_interval == 0:
        dt = time.time() - t0
        print(f"Iter {iter_num}: Loss {loss.item():.4f}, Time {dt*1000:.2f}ms")
        t0 = time.time()

    iter_num += 1
