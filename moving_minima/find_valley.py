from itertools import product
from time import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_algorithms.gradients import ReverseQGT

degenerate_minima = np.load(
    "/home/marc/Documents/Fidelity/FidelityLandscape/moving_minima/degenerate_minimas.npy",
    allow_pickle=True,
).item()

qc = degenerate_minima["qc"]
initial_parameters = degenerate_minima["Original"]
degenerate_parameters = initial_parameters + degenerate_minima["Degenerate"]
initial_state = Statevector(qc.assign_parameters(initial_parameters))
degenerate_infidelity = 1 - state_fidelity(
    Statevector(qc.assign_parameters(degenerate_parameters)), initial_state
)

print(f"The initial infidelity is {degenerate_infidelity}")
eps = degenerate_infidelity * 1e2
print("At least this much:", (2 * eps / qc.num_parameters**2) ** (1 / 2))
res = 1e-4

s, v = np.linalg.eigh(ReverseQGT().run(qc, [initial_parameters]).result().qgts[0])
rank = np.sum(s > 1e-10)
print(s, s[-rank:])
increments_kernel = v[:-rank] * (24 * eps / qc.num_parameters**3) ** (1 / 4)
increments_significant = np.array() [v[-r] np.sqrt(s[-r] * 2 / qc.num_parameters**3) for r in range(1,rank+1)])


# increments = np.array(list(product((-res, 0, res), repeat=qc.num_parameters)))
# increments = np.concatenate(
#     (res * np.eye(qc.num_parameters), -res * np.eye(qc.num_parameters))
# )


def f(x):
    state = Statevector(qc.assign_parameters(initial_parameters + x))
    return 1 - state_fidelity(state, initial_state)


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

np.save(
    "/home/marc/Documents/Fidelity/FidelityLandscape/moving_minima/tested_region.npy",
    {"Good": good, "Tested": tested},
    allow_pickle=True,
)

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
