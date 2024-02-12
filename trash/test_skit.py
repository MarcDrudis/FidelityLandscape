from itertools import product
from time import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.figure import SubFigure
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, state_fidelity

np.random.seed(1)

params = ParameterVector("t", 9)
qc = QuantumCircuit(3)
qc.rxx(params[0], 0, 1)
qc.ryy(params[1], 1, 2)
qc.rzz(params[2], 0, 2)
qc.rxx(params[3], 0, 1)
qc.ryy(params[4], 1, 2)
qc.rzz(params[5], 0, 2)
# qc.rxx(params[6], 0, 1)
# qc.ryy(params[7], 1, 2)
# qc.rzz(params[8], 0, 2)


initial_parameters = np.random.uniform(-np.pi, np.pi, qc.num_parameters)
initial_state = Statevector(qc.assign_parameters(initial_parameters))


def f(x):
    state = Statevector(qc.assign_parameters(initial_parameters + x))
    return 1 - state_fidelity(state, initial_state)


# brute_grid = np.array(
#     np.meshgrid(*([np.linspace(-1e-2, 1e-2, 100)] * qc.num_parameters))
# )
# brute_values = np.apply_along_axis(f, 0, brute_grid)

eps = 1e-8
res = 10 ** (-4.5)
# ratio = 1.0

# c = plt.imshow(np.float32(brute_values < 1e-2))
# c = plt.imshow(brute_values < eps)
# plt.colorbar(c)
# plt.legend()
# plt.show()


def next_explore(curr: set) -> set:
    next = set()
    for e in curr:
        point = np.frombuffer(e)
        next.update(i.tobytes() for i in point + increments)
    return next - tested


def evaluate_classify(c):
    tested.add(c)
    val = np.frombuffer(c)
    if f(val) < eps:
        new_good.add(c)


tested = {np.zeros(qc.num_parameters).tobytes()}
good = set()
current_explore = {np.zeros(qc.num_parameters).tobytes()}

# increments = np.array(list(product((-res, 0, res), repeat=qc.num_parameters)))
increments = np.concatenate(
    (res * np.eye(qc.num_parameters), -res * np.eye(qc.num_parameters))
)

counter = 0
while True:
    new_good = set()
    t = time()
    jobs = (delayed(evaluate_classify)(c) for c in current_explore)
    Parallel(n_jobs=32, require="sharedmem")(jobs)
    # [evaluate_classify(c) for c in current_explore]
    ratio = len(new_good) / len(current_explore)
    print(
        len(current_explore),
        np.round(ratio, 3),
        time() - t,
    )
    good.update(new_good)
    current_explore = next_explore(new_good)
    if len(current_explore) == 0:
        break

good = np.array([np.frombuffer(g) for g in good])
tested = np.array([np.frombuffer(t) for t in tested])

print(good.shape, tested.shape)

plt.scatter(good[:, 0], good[:, 1])
plt.show()
plt.scatter(good[:, 2], good[:, 3])
plt.show()
plt.scatter(good[:, 4], good[:, 5])
plt.show()
plt.scatter(good[:, 2], good[:, 0])
plt.show()
plt.scatter(good[:, 2], good[:, 5])
plt.show()
