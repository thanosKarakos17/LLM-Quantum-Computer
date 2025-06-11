from adapt_qaoa import*
import networkx as nx

def maxcut_hamiltonian(G: nx.Graph) -> SparsePauliOp:
    num_nodes = G.number_of_nodes()
    pauli_terms = []
    coefs = []
    for u, v, attr in G.edges(data=True):
        weight = attr.get('weight', 1.0)
        term = ['I']*num_nodes
        term[u] = 'Z'
        term[v] = 'Z'
        pauli_string = ''.join(term)
        pauli_terms.append(pauli_string)
        coefs.append(weight)
    
    return SparsePauliOp.from_list(list(zip(pauli_terms, coefs)))

def solve_adapt_qaoa(cost_h, grad_tol=0.05):
    pool, _, _ = build_operator_pools(cost_h.num_qubits)
    #print('pool has ', len(pool))
    adapt_ansatz = adapt_vqa(h_cost=cost_h, pool=pool, grad_tol=grad_tol)
    return adapt_ansatz

def plot_result(adapt_ansatz):
    adapt_ansatz.measure_all()
    sim = Aer.get_backend('aer_simulator')
    counts = sim.run(adapt_ansatz.decompose(reps=3), shots=10000).result().get_counts()
    fig = plot_histogram(counts)
    plt.show()

def get_circuit_token(adapt_ansatz):
    gama = True
    circuit_tok = []
    for i, _instruction in enumerate(adapt_ansatz.data):
        if len(_instruction[0].params) > 0:
            #print('\nInstruction:', _instruction[0].name, _instruction[0].label)
            #print('Params:', _instruction[0].params[0])
            if gama: 
                circuit_tok = circuit_tok + ['<new_layer>'] + [circuit_tokens.pop(0)]
            
            circuit_tok = circuit_tok + [round(float(str(_instruction[0].params[0])), 2)]
            gama = not(gama)

    return circuit_tok
