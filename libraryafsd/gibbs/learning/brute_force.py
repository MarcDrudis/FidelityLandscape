from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp,entropy,partial_trace,Statevector,state_fidelity
from gibbs.utils import simple_purify_hamiltonian
from scipy.optimize import minimize
import numpy as np

def free_energy(qc:QuantumCircuit,parameters:np.ndarray,beta:float,H:SparsePauliOp):
    mixed_state = partial_trace(
        Statevector(qc.bind_parameters(parameters)),
        range(qc.num_qubits//2,qc.num_qubits)
        )
    temperature  = 1/(1.380649e-23*beta)
    free_energy = np.real(mixed_state.expectation_value(H) - temperature*entropy(mixed_state))
    print(free_energy)
    return free_energy

def infidelity(qc:QuantumCircuit,parameters:np.ndarray,state:Statevector):
    fid = state_fidelity(state,Statevector(qc.bind_parameters(parameters)))
    return 1-fid

def best_parameters(qc:QuantumCircuit,x0:np.ndarray,H:SparsePauliOp,beta:float):
    loss = lambda par: infidelity(qc,par,simple_purify_hamiltonian(H))
    result =  minimize(loss,x0)
    return result