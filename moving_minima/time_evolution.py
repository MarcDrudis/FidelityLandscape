import pathlib

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from qiskit import QuantumCircuit
from qiskit_algorithms.gradients import ReverseQGT

directory = pathlib.Path(__file__).parent.resolve()
plt.style.use(directory.parent / "plots/plot_style.mplstyle")

minima = np.load(
    directory.parent / "moving_minima/moving_minima.npy",
    allow_pickle=True,
).item()

qc = minima["qc"]
global_params = np.array(minima["global_pert"])


def get_fisher_value(qc: QuantumCircuit, params: np.ndarray, update: np.ndarray):
    fisher = np.real(ReverseQGT().run([qc], [params]).result().qgts[0]) / 2
    norm = np.linalg.norm(update, 2) ** 2

    return 0 if norm == 0 else update.T @ fisher @ update / norm


jobs = (
    delayed(get_fisher_value)(qc, global_params[0], u)
    for u in global_params - global_params[0]
)
# fv = Parallel(n_jobs=11)(jobs)
# print(fv)

colors = ["#4056A1", "#075C2F", "#7D8238", "#453F3F", "#692411", "#D79922", "#F13C20"]


# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].scatter(
    x=minima["times"][1:],
    y=np.linalg.norm(
        np.array(minima["global_pert"]) % np.pi,
        axis=1,
    ),
)
axs[1].scatter(x=minima["times"], y=minima["global_inf"])

plt.show()
