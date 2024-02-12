from threading import local

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm

from fidlib.basicfunctions import create_ising

np.random.seed(2)


params = ParameterVector("t", 9)
qc = QuantumCircuit(2)
qc.rx(params[0], 0)
qc.rz(params[1], 1)
qc.cnot(0, 1)
qc.rx(params[2], 0)
qc.rz(params[3], 1)
qc.cnot(0, 1)


def lossfunction(
    perturbation: np.ndarray, initial_parameters: np.ndarray, t: float | None = None
) -> float:
    state1 = Statevector(qc.assign_parameters(initial_parameters + perturbation))
    state2 = Statevector(qc.assign_parameters(initial_parameters))
    if t is not None:
        # H = t * create_ising(3, 0.25, -1)
        H = t * create_ising(3, -5, -1)
        state2 = expm_multiply(1.0j * H.to_matrix(sparse=True), state2.data)
        state2 = Statevector(state2 / np.linalg.norm(state2))

    return 1 - state_fidelity(state1, state2)


# block_params = np.random.uniform(-np.pi, np.pi, qc.num_qubits)
block_params = np.array([0.5, 0.2])
initial_parameters = np.zeros(qc.num_parameters)
initial_parameters[qc.num_qubits :] = block_params
direction = np.concatenate((-block_params, block_params))

deltas = np.linspace(-1, 1, 101)

for d in deltas:
    print(d, d * direction + initial_parameters)

infidelities = [lossfunction(d * direction, initial_parameters) for d in deltas]
plt.plot(deltas, infidelities)
plt.show()

cut_minima = -4.07 * direction

result = minimize(lossfunction, cut_minima, args=(initial_parameters))
print(result)

print(
    "Now that we have some optimal point, we plot the landscape in a cut that has this optimal point at x=1. We see that we have an almost degenerate ground state. Now the question is which local minima will remain the smallest."
)

local_minima = result.x

np.save(
    "/home/marc/Documents/Fidelity/FidelityLandscape/moving_minima/degenerate_minimas.npy",
    {"Original": initial_parameters, "Degenerate": local_minima, "qc": qc},
    allow_pickle=True,
)


deltas = np.linspace(-1, 1, 100) * 1.3
infidelities = [lossfunction(d * local_minima, initial_parameters) for d in deltas]

plt.plot(deltas, infidelities)
plt.show()

print(
    "We let the system evolve for different times and then see the trajectory of our local minima. We can see that our initial parameters don't actually correspond to the minima we wanted to be at."
)

times = np.linspace(0, 5e-1, 50)
global_inf = [lossfunction(0, initial_parameters)]
global_params = [np.zeros(qc.num_parameters)]
local_inf = [lossfunction(local_minima, initial_parameters)]
local_params = [local_minima]

for t in tqdm(times[1:]):
    result_global = minimize(
        lossfunction, global_params[-1], args=(initial_parameters, t)
    )
    result_local = minimize(
        lossfunction, local_params[-1], args=(initial_parameters, t)
    )
    global_inf.append(result_global.fun)
    local_inf.append(result_local.fun)
    global_params.append(result_global.x)
    local_params.append(result_local.x)

np.save(
    "moving_minima/degenerate_trajectory.npy",
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

plt.plot([np.linalg.norm(l - g) for l, g in zip(local_params, global_params)])
plt.show()

plt.plot(
    [
        np.linalg.norm(global_params[i] - global_params[i + 1]) / 25e-4
        for i in range(len(global_params) - 1)
    ]
)
plt.show()

plt.plot(times, global_inf, marker=".", label="Initial Params")
plt.plot(times, local_inf, label="Local Minima")
plt.xlabel("Time")
plt.ylabel("Infidelity")
plt.legend()
plt.show()
