from qiskit.quantum_info import SparsePauliOp
import numpy as np
from itertools import product
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector, Parameter
from qiskit.primitives import Estimator
from qiskit.opflow import PauliSumOp
from scipy.optimize import minimize
from qiskit.circuit.library import QAOAAnsatz
from qiskit.visualization import circuit_drawer as display
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

circuit_tokens = []

def compute_gradient(ansatz, evolved_ops, estimator):
    gradients = []
    #for op in evolved_ops: op.coeffs = np.real(op.coeffs)
    #grad_val = estimator.run([ansatz]*len(evolved_ops), evolved_ops, shots=100).result()
    grad_val = estimator.run([ansatz]*len(evolved_ops), evolved_ops).result()
    gradients = np.real(grad_val.values)
    return np.array(gradients)

def compute_evolved_ops(h_cost, pool, gamma):
    evolved_ops = []

    H = PauliSumOp(h_cost)
    U = (-1j * gamma * H).exp_i()
    U_dag = (1j * gamma * H).exp_i()

    for A_j in pool:
        A = PauliSumOp(A_j)
        commutator = H @ A - A @ H
        evolved_op = U_dag @ commutator @ U
        evolved_op = SparsePauliOp.from_operator(evolved_op)
        evolved_ops.append(evolved_op)
    #print('evolved ops pool done!')
    return evolved_ops


def add_layer(ansatz, h_cost: SparsePauliOp, pauli_op: SparsePauliOp, paramsText):
    evo_circ = QAOAAnsatz(cost_operator=PauliSumOp(h_cost), mixer_operator=PauliSumOp(pauli_op), initial_state=ansatz, reps=1).assign_parameters(paramsText).decompose()
    return evo_circ


def adapt_vqa(h_cost: SparsePauliOp, pool: list, max_steps=20, gamma=0.01, grad_tol=5e-3, max_iter=400):
    estimator = Estimator()
    n_qubits = h_cost.num_qubits
    ansatz = QuantumCircuit(n_qubits)

    ansatz.h(range(n_qubits))  # start with |+⟩^n
    params = ParameterVector("θ", length=0)
    ops = []
    prev = ansatz
    #print(display(ansatz))
    evolved_ops = compute_evolved_ops(h_cost, pool, gamma)
    for step in range(max_steps):
        #print(f"\n--- ADAPT Step {step+1} ---")

        # Compute gradients
        gradients = compute_gradient(prev, evolved_ops, estimator)
        norm = np.linalg.norm(gradients)
        #print(f"Gradient norm: {norm:.6f}")

        if norm < grad_tol:
            #print("Converged!")
            break

        best_idx = np.argmax(np.abs(gradients))
        best_op = pool[best_idx]
        #print(f"Adding operator {best_op} with gradient {gradients[best_idx]:.6f}")

        # Add new parameter and update ansatz
        ops.append(best_op)
        paramst = ParameterVector("a", len(ops)*2)
        evo_circuit = add_layer(ansatz, h_cost, best_op, paramst)
        ansatz = evo_circuit
        #print(display(ansatz))

        # Define cost function over current ansatz
        def cost_fn(x):
            #binding = {p: x[i] for i, p in enumerate(params)}
            #pbinding = {param: parametersQAOA[index] for index, param in enumerate(ansatz.parameters)}
            circ = ansatz.assign_parameters(x)
            val = estimator.run([circ], [h_cost]).result().values[0]
            return np.real(val)

        x0 = np.random.uniform(0, 2*np.pi, len(ops)*2)
        res = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": max_iter})
        optimal_energy = res.fun
        optimal_params = res.x
        #print(f"Step {step+1} minimized energy: {optimal_energy}")

        circuit_tokens.append(best_idx)
        # Update ansatz with optimized parameters
        bound = ansatz.assign_parameters(optimal_params)
        prev = bound
        #print(display(ansatz))

    return prev


def build_operator_pools(n_qubits):
    paulis = []

    # P_single: X_i and Y_i
    for i in range(n_qubits):
        for pauli in ['X', 'Y']:
            label = ['I'] * n_qubits
            label[i] = pauli
            paulis.append(SparsePauliOp("".join(label), coeffs=[1.0]))

    # P_single: global mixer ∑ Y_i
    global_Y_label = ['Y'] * n_qubits
    global_Y = SparsePauliOp(["".join(global_Y_label)], coeffs=[1.0])
    paulis.append(global_Y)

    # P_QAOA: global mixer ∑ X_i (but here included as a single operator)
    global_X_label = ['X'] * n_qubits
    global_X = SparsePauliOp(["".join(global_X_label)], coeffs=[1.0])
    P_qaoa = global_X.copy()
    paulis.append(global_X)

    P_single = paulis.copy()

    # P_multi: all two-qubit combinations {B_i C_j}
    two_qubit_ops = []
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            for B, C in product(['X', 'Y', 'Z'], repeat=2):
                label = ['I'] * n_qubits
                label[i] = B
                label[j] = C
                two_qubit_ops.append(SparsePauliOp("".join(label), coeffs=[1.0]))

    P_multi = P_single + two_qubit_ops
    #P_multi = two_qubit_ops

    return P_multi, P_single, P_qaoa