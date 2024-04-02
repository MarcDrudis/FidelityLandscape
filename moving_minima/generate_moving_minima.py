import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector, state_fidelity
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm
from windsurfer.ansatz.hamiltonian import ansatz_QRTE_Hamiltonian
from windsurfer.hamiltonians.oneDlattice import lattice_hamiltonian
from windsurfer.utils.norms import eigenrange

directory = pathlib.Path(__file__).parent.resolve()
print(directory)

runnable_args = sys.argv[1:]

num_qubits = int(runnable_args[0])

termsA = [("Y", -0.95)]
termsB = [("XX", 1)]
terms = termsA + termsB
HB = lattice_hamiltonian(num_qubits, termsB)
HB.paulis = HB.paulis[::2] + HB.paulis[1::2]
H = lattice_hamiltonian(num_qubits, termsA) + HB

normalization = eigenrange(H)
print(normalization)
H /= normalization
print(H)

qc = ansatz_QRTE_Hamiltonian(H, reps=2)
print(qc)


def lossfunction(
    perturbation: np.ndarray, initial_parameters: np.ndarray, H: float | None = None
) -> float:
    state1 = Statevector(qc.assign_parameters(initial_parameters + perturbation))
    state2 = Statevector(qc.assign_parameters(initial_parameters))
    if H is not None:
        state2 = expm_multiply(-1.0j * H.to_matrix(sparse=True), state2.data)
        state2 = Statevector(state2 / np.linalg.norm(state2))

    return 1 - state_fidelity(state1, state2)


initial_parameters = np.random.default_rng(num_qubits).uniform(
    -np.pi, np.pi, qc.num_parameters
)


# times = np.linspace(0, 10, 4)
# times = np.linspace(0, 10, 10 + 1)
times = np.linspace(0, 10, 4 * 10 + 1)
print("Times", times)
global_inf = [lossfunction(0, initial_parameters)]
global_params = [np.zeros(qc.num_parameters)]

for t in tqdm(times[1:]):
    result_global = minimize(
        lossfunction, global_params[0], args=(initial_parameters, H * t)
    )
    global_inf.append(result_global.fun)
    global_params.append(result_global.x)

data = {
    "qc": qc,
    "initial_parameters": initial_parameters,
    "infidelity": global_inf,
    "perturbation": global_params,
    "Hamiltonian": H,
    "times": times,
}

if not (directory / str(terms)).is_dir():
    os.makedirs(directory / str(terms))

np.save(
    directory / f"{terms}/moving_minima_qubits={num_qubits}",
    data,
    allow_pickle=True,
)

# plt.plot(data["times"], data["infidelity"])
# plt.show()
