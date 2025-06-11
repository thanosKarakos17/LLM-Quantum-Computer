import os
import pickle
import torch
import numpy as np
from model import GPT, GPTConfig
import networkx as nx
import sys
sys.path.append(os.path.abspath(".."))
from feather import*
from max_cut_adapt import *

# -----------------------------------------------------------------------------
out_dir = 'out-qaoa-maxcut'
data_dir = 'data/maxcut'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
feather_dim = 100
block_size = 103
# -----------------------------------------------------------------------------

# Load tokenizer metadata (once)
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
vocab_size = meta['vocab_size']

# Load model (once)
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
model_args['feather_dim'] = feather_dim
model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Encode function
def encode_prompt(prompt: str):
    tokens = prompt.strip().split()
    return torch.tensor([stoi[s] for s in tokens], dtype=torch.long)[None, :]  # (1, T)

# Sampling function
@torch.no_grad()
def generate_circuit(graph_prompt: str, feather, max_new_tokens: int = 50):
    idx = encode_prompt(graph_prompt).to(device)

    # Feather vector shape (1, d)
    if isinstance(feather, np.ndarray):
        feather_tensor = torch.tensor(feather, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        feather_tensor = feather.unsqueeze(0).to(device)

    generated = idx
    for _ in range(max_new_tokens):
        idx_cond = generated if generated.size(1) <= block_size else generated[:, -block_size:]
        logits, _ = model(idx_cond, feather=feather_tensor)
        logits = logits[:, -1, :]  # (batch, vocab)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
        if itos[next_token.item()] == '<eos>':
            break

    tokens = [itos[i] for i in generated[0].tolist()]
    return tokens

# === Generate random BA graph and use it as input ===
def random_BA_graph(nodes=4, edges=2):
    #G = nx.cycle_graph(4)  # 0-1-2-3-0
    G = nx.barabasi_albert_graph(nodes, edges)
    while nx.number_of_isolates(G) > 0:
        G = nx.barabasi_albert_graph(nodes, edges)
    return G

def assign_random_weights(G):
    for u, w in G.edges:
        G[u][w]['weight'] = round(np.random.uniform(0.01, 1.01), 2)

def get_graph_token(G):
    edge_weight = []
    for u, w in G.edges:
        edge_weight += [(u, w), G[u][w].get('weight', 1.0)]
    graph_token = ['<bos>'] + edge_weight + ['end_of_graph']
    graph_token = ' '.join(str(x) for x in graph_token)
    return graph_token

def get_feather_token(G, order=2):
    ftero = FEATHERG(order=order)
    ftero.fit([G])
    graph_embedding = ftero.get_embedding()[0]
    graph_embedding = np.array(graph_embedding)
    return graph_embedding

def generate_pool(G):
    cost_h = maxcut_hamiltonian(G)
    pool, _, _ = build_operator_pools(cost_h.num_qubits)
    return cost_h, pool
# === ===

if __name__ == "__main__":
    G = random_BA_graph(4, 2)
    assign_random_weights(G)
    prompt = get_graph_token(G)
    feather = get_feather_token(G)
    out_tokens = generate_circuit(prompt, feather)
    print('Graph token for random BA graph ', prompt)
    print("Generated tokens:")
    print(" ".join(out_tokens))