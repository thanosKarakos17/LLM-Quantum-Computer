import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from feather import*
from max_cut_adapt import *
from qiskit.circuit.library import EfficientSU2

operator_indexes = [12, 55, 44]
gammas = [2.22, 5.58, 4.95]
betas = [5.4, 5.62, 3.34]

G = nx.Graph()
G.add_edge(0, 1, weight=0.8)
G.add_edge(0, 2, weight=0.29)
G.add_edge(0, 3, weight=0.38)
G.add_edge(2, 3, weight=0.32)

def draw_graph(G):
    nx.draw_networkx(G, pos=nx.drawing.circular_layout(G))
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G, pos=nx.drawing.circular_layout(G), edge_labels=labels)
    plt.show()

#draw_graph(G)
def generate_pool(G):
    cost_h = maxcut_hamiltonian(G)
    pool, _, _ = build_operator_pools(cost_h.num_qubits)
    return cost_h, pool

cost_h, pool = generate_pool(G)
adapt_ansatz = QuantumCircuit(G.number_of_nodes())
adapt_ansatz.h(range(cost_h.num_qubits))
for index, ok in enumerate(operator_indexes):
    paramsText = [betas[index], gammas[index]]
    evo_circ = QAOAAnsatz(cost_operator=PauliSumOp(cost_h), mixer_operator=PauliSumOp(pool[ok]), initial_state=adapt_ansatz, reps=1).assign_parameters(paramsText).decompose()
    adapt_ansatz = evo_circ


estimator = Estimator()
def cost_function(params):
    bound_ansatz = ansatz
    bound_ansatz = bound_ansatz.assign_parameters(params)
    value = estimator.run([bound_ansatz], [cost_h], shots=100).result().values[0]
    return value  # Minimize expectation value

opt_fun = []
opt_vals = []
for i in range(10):
    ansatz_hea = EfficientSU2(G.number_of_nodes(), su2_gates=["ry"], entanglement="linear", reps=2).decompose()
    qc3 = adapt_ansatz.decompose(reps=3).compose(ansatz_hea)
    ansatz = qc3
    initial_params = np.random.uniform(0, 2*np.pi, len(ansatz.parameters))
    result = minimize(cost_function, initial_params, method='COBYLA', options={'maxiter': 300})
    opt_fun.append(result.fun)
    opt_vals.append(result.x)

opt_ind = np.argmin(opt_fun)
val = opt_vals[opt_ind]
parametersQAOA = val.tolist()
param_values = {param: parametersQAOA[index] for index, param in enumerate(ansatz.parameters)}
bound_circuit = ansatz.assign_parameters(param_values).decompose(reps=3)
plot_result(bound_circuit)

#display(ansatz, output='mpl', style='iqp')
# qc_temp = adapt_ansatz.compose(ansatz_hea)
# display(qc_temp, output='mpl', style='iqp')
plt.show()

#evaluation using adapt
# def heuristic_grad_tol(G, alpha=0.01):
#     total_weight = sum(d.get("weight", 1.0) for u, v, d in G.edges(data=True))
#     num_nodes = G.number_of_nodes()
#     return alpha * (total_weight / num_nodes)

# adapt_ansatz = adapt_vqa(h_cost=cost_h, pool=pool, grad_tol=heuristic_grad_tol(G))
# print('OPTIMAL ADAPT ANSATZ')
# display(adapt_ansatz, output='mpl', style='iqp')
# plt.show()
# adapt_ansatz.measure_all()
# sim = Aer.get_backend('aer_simulator')
# counts = sim.run(adapt_ansatz.decompose(reps=3), shots=10000).result().get_counts()
# fig = plot_histogram(counts)
# plt.show()