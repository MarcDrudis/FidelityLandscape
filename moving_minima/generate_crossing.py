import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm

from fidlib.basicfunctions import create_ising, find_local_minima

np.random.seed(2)


params = ParameterVector("t", 3)
qc = QuantumCircuit(3)
qc.rxx(params[0], 0, 1)
qc.ryy(params[1], 1, 2)
qc.rzz(params[2], 0, 2)
qc.rxx(params[0], 0, 1)
qc.ryy(params[1], 1, 2)
qc.rzz(params[2], 0, 2)
qc.rxx(params[0], 0, 1)
qc.ryy(params[1], 1, 2)
qc.rzz(params[2], 0, 2)
# print(qc.draw())


def lossfunction(
    perturbation: np.ndarray, initial_parameters: np.ndarray, t: float | None = None
) -> float:
    state1 = Statevector(qc.assign_parameters(initial_parameters + perturbation))
    state2 = Statevector(qc.assign_parameters(initial_parameters))
    if t is not None:
        H = t * create_ising(3, 0.3, -0.9)
        state2 = expm_multiply(1.0j * H.to_matrix(sparse=True), state2.data)
        state2 = Statevector(state2 / np.linalg.norm(state2))

    return 1 - state_fidelity(state1, state2)


direction = np.random.uniform(-1, 1, 3)
direction /= np.linalg.norm(direction)
deltas = np.linspace(-1, 1, 100) * np.pi
initial_parameters = np.random.uniform(-np.pi, np.pi, 3)

infidelities = [lossfunction(d * direction, initial_parameters) for d in deltas]
# plt.plot(deltas, infidelities)
# plt.show()

cut_minima = -2.4 * direction

local_minima = find_local_minima(lossfunction, cut_minima, initial_parameters)

direction = np.mod(local_minima, np.pi)
deltas = np.linspace(-1, 1, 100) * 1.3
infidelities = [lossfunction(d * direction, initial_parameters) for d in deltas]

# plt.plot(deltas, infidelities)
# plt.show()
times = np.linspace(0, 2, 50)
global_inf = [lossfunction(0, initial_parameters)]
global_params = [np.zeros(local_minima.size)]
local_inf = [lossfunction(local_minima, initial_parameters)]
local_params = [local_minima]

for t in tqdm(times[1:]):
    # result_global = find_local_minima(
    #     lossfunction,
    #     global_params[-1],
    #     initial_parameters,
    #     t,
    #     learning_rate=1e-1,
    #     epsilon=1e-4,
    # )
    # result_local = find_local_minima(
    #     lossfunction, local_params[-1], initial_parameters, t
    # )
    result_global = minimize(
        lossfunction, global_params[-1], args=(initial_parameters, t)
    ).x
    result_local = minimize(
        lossfunction, local_params[-1], args=(initial_parameters, t)
    ).x
    global_inf.append(lossfunction(result_global, initial_parameters, t))
    local_inf.append(lossfunction(result_local, initial_parameters, t))
    global_params.append(result_global)
    local_params.append(result_local)

np.save(
    "/home/marc/Documents/Fidelity/FidelityLandscape/moving_minima/data_crossing.npy",
    {
        "qc": qc,
        "global_inf": global_inf,
        "global_params": global_params,
        "local_inf": local_inf,
        "local_params": local_params,
        "times": times,
    },
    allow_pickle=True,
)
plt.plot(times, global_inf, marker=".")
plt.plot(times, local_inf)
plt.show()

##Leave it as it is. We notice a crossing in this graphic at time 1.25 - 1.5
