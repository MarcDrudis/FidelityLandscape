import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply

from fidlib.basicfunctions import create_ising

np.random.seed(2)


params = ParameterVector("t", 9)
qc = QuantumCircuit(3)
qc.rxx(params[0], 0, 1)
qc.ryy(params[1], 1, 2)
qc.rzz(params[2], 0, 2)
qc.rxx(params[3], 0, 1)
qc.ryy(params[4], 1, 2)
qc.rzz(params[5], 0, 2)
qc.rxx(params[6], 0, 1)
qc.ryy(params[7], 1, 2)
qc.rzz(params[8], 0, 2)
# print(qc.draw())


def lossfunction(
    perturbation: np.ndarray, initial_parameters: np.ndarray, t: float | None = None
) -> float:
    state1 = Statevector(qc.assign_parameters(initial_parameters + perturbation))
    state2 = Statevector(qc.assign_parameters(initial_parameters))
    if t is not None:
        H = t * create_ising(3, 0.25, -1)
        state2 = expm_multiply(1.0j * H.to_matrix(sparse=True), state2.data)
        state2 = Statevector(state2 / np.linalg.norm(state2))

    return 1 - state_fidelity(state1, state2)


print(
    "Let's start by choosing the cut in some random direction. It seems like there is a local minima at -4.07 in this cut so we are going to take that as our starting point and minimize from there."
)
direction = np.random.uniform(-1, 1, 3)
direction = np.tile(direction, 3)
direction /= np.linalg.norm(direction)
deltas = np.linspace(-1, 1, 100) * 9
initial_parameters = np.random.uniform(-np.pi, np.pi, 3)
initial_parameters = np.tile(initial_parameters, 3)

# infidelities = [lossfunction(d * direction, initial_parameters) for d in deltas]
# plt.plot(deltas, infidelities)
# plt.show()

cut_minima = -4.07 * direction

result = minimize(lossfunction, cut_minima, args=(initial_parameters))
print(result)

print(
    "Now that we have some optimal point, we plot the landscape in a cut that has this optimal point at x=1. We see that we have an almost degenerate ground state. Now the question is which local minima will remain the smallest."
)

local_minima = result.x


deltas = np.linspace(-1, 1, 100) * 1.3
infidelities = [lossfunction(d * local_minima, initial_parameters) for d in deltas]

# plt.plot(deltas, infidelities)
# plt.show()

print(
    "We let the system evolve for different times and then see the trajectory of our local minima. We can see that our initial parameters don't actually correspond to the minima we wanted to be at."
)

times = np.linspace(0, 5e-3, 50)
global_inf = [lossfunction(0, initial_parameters)]
global_params = [0]
local_inf = [lossfunction(local_minima, initial_parameters)]
local_params = [local_minima]
for t in times[1:]:
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
    "/home/marc/Documents/Fidelity/FidelityLandscape/degenenerate_data.npy",
    {"Time": times, "Local": local_inf, "Global": global_inf},
    allow_pickle=True,
)

plt.plot(times, global_inf, marker=".", label="Initial Params")
plt.plot(times, local_inf, label="Local Minima")
plt.xlabel("Time")
plt.ylabel("Infidelity")
plt.legend()
plt.show()
