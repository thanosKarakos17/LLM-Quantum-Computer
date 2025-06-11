import os
import numpy as np
import pickle
import random
from tqdm import tqdm
import ast

random.seed(42)
out_dir = os.path.dirname(__file__)

# === step 1 Load dataset===
#parse feather embedding
with open('../../../feather.txt', 'r') as f:
    lines1 = f.readlines()
feathers = [ast.literal_eval(line.strip()) for line in lines1]

#parse graph-circuit tokens
with open('../../../graph_circuit_tokens.txt', 'r') as f:
    lines = f.read().strip().splitlines()

assert len(lines) == len(feathers), "Mismatch between graph-circuit and feather count"

# === Step 2 Shuffle in sync ===
indices = list(range(len(lines)))
random.shuffle(indices)

lines = [lines[i] for i in indices]
feathers = [feathers[i] for i in indices]

# === Step 3: Build vocabulary from all tokens ===
vocab = set()
for line in lines:
    vocab.update(line.strip().split())

itos = sorted(list(vocab))               # index to string
stoi = {ch: i for i, ch in enumerate(itos)}  # string to index

vocab_size = len(itos)
print(f"Vocab size: {vocab_size}")

# === Step 4: Tokenize ===
def encode(line):
    return [stoi[tok] for tok in line.strip().split()]

encoded_lines = [encode(line) for line in tqdm(lines)]
lens = [len(seq) for seq in encoded_lines]

# === Step 5: Train/val split (90/10) ===
split = int(0.9 * len(encoded_lines))
train_ids = encoded_lines[:split]
val_ids = encoded_lines[split:]
train_feats = feathers[:split]
val_feats = feathers[split:]

train_tokens = np.concatenate(train_ids).astype(np.uint16)
val_tokens = np.concatenate(val_ids).astype(np.uint16)

# === Step 6 Save .bin token files ===
train_out = os.path.join(out_dir, 'train.bin')
val_out = os.path.join(out_dir, 'val.bin')

np.memmap(train_out, dtype='uint16', mode='w+', shape=train_tokens.shape)[:] = train_tokens
np.memmap(val_out, dtype='uint16', mode='w+', shape=val_tokens.shape)[:] = val_tokens

# === Save feather embeddings ===
np.save(os.path.join(out_dir, 'train_feathers.npy'), train_feats)
np.save(os.path.join(out_dir, 'val_feathers.npy'), val_feats)

# === Save metadata ===
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("✅ Done. Files saved to", out_dir)