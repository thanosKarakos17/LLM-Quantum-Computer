import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from feather import*
from max_cut_adapt import *

def random_ER_graph(nodes=4, p=0.5):
    #G = nx.cycle_graph(4)  # 0-1-2-3-0
    G = nx.erdos_renyi_graph(nodes, p)
    while nx.number_of_isolates(G) > 0:
        G = nx.erdos_renyi_graph(nodes, p)
    return G

def assign_random_weights(G):
    for u, w in G.edges:
        G[u][w]['weight'] = round(np.random.uniform(0.01, 1.01), 2)

def heuristic_grad_tol(G, alpha=0.01):
    total_weight = sum(d.get("weight", 1.0) for u, v, d in G.edges(data=True))
    num_nodes = G.number_of_nodes()
    return alpha * (total_weight / num_nodes)

def draw_graph(G):
    nx.draw_networkx(G, pos=nx.drawing.circular_layout(G))
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G, pos=nx.drawing.circular_layout(G), edge_labels=labels)
    plt.show()

def get_graph_token(G):
    edge_weight = []
    for u, w in G.edges:
        edge_weight += [(u, w), G[u][w].get('weight', 1.0)]
    graph_token = ['<bos>'] + edge_weight + ['end_of_graph']
    return graph_token

def get_feather_token(G, order=2):
    ftero = FEATHERG(order=order)
    ftero.fit([G])
    graph_embedding = ftero.get_embedding()[0]
    return graph_embedding

def tokenize_graph(nodes=4):
    G = random_ER_graph(nodes=nodes)
    assign_random_weights(G)
    cost_h = maxcut_hamiltonian(G)
    #print(cost_h)
    ansatz = solve_adapt_qaoa(cost_h, heuristic_grad_tol(G, 0.1))
    circ_token = get_circuit_token(ansatz)
    graph_token = get_graph_token(G)
    feather_token = get_feather_token(G)
    token_embedding = graph_token + circ_token
    token_embedding = ' '.join(str(x) for x in token_embedding)

    return token_embedding, feather_token

file_tokens = open('graph_circuit_tokens.txt', 'a')
file_feather = open('feather.txt', 'a')

for i in range(20):
    token_embedding, feather_token = tokenize_graph(nodes=4)
    file_tokens.write(token_embedding)
    file_tokens.write('\n')

    file_feather.write(str(feather_token))
    file_feather.write('\n')
    print('saved ', i)

for i in range(30):
    token_embedding, feather_token = tokenize_graph(nodes=5)
    file_tokens.write(token_embedding)
    file_tokens.write('\n')

    file_feather.write(str(feather_token))
    file_feather.write('\n')
    print('saved ', i + 20)

for i in range(50):
    token_embedding, feather_token = tokenize_graph(nodes=6)
    file_tokens.write(token_embedding)
    file_tokens.write('\n')

    file_feather.write(str(feather_token))
    file_feather.write('\n')
    print('saved ', i + 50)

file_feather.close()
file_tokens.close()
# plot_result(ansatz)
# draw_graph(G)
