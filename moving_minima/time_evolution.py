import pathlib
from itertools import product
from time import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_algorithms.gradients import ReverseQGT

directory = pathlib.Path(__file__).parent.resolve()
plt.style.use(directory.parent / "plots/plot_style.mplstyle")

minima = np.load(
    directory.parent / "moving_minima/data_crossing.npy",
    allow_pickle=True,
).item()

qc = minima["qc"]
global_params = np.array(minima["global_params"])


def get_fisher_value(qc: QuantumCircuit, params: np.ndarray, update: np.ndarray):
    fisher = np.real(ReverseQGT().run([qc], [params]).result().qgts[0]) / 2
    norm = np.linalg.norm(update, 2) ** 2

    return 0 if norm == 0 else update.T @ fisher @ update / norm


fv = [
    get_fisher_value(qc, global_params[0], u) for u in global_params - global_params[0]
]

x = minima["times"][1:]
y = fv[1:]
y2 = np.linalg.norm(global_params, axis=1)[1:]

# Set a consistent color palette
colors = ["#4056A1", "#075C2F", "#7D8238", "#453F3F", "#692411", "#D79922", "#F13C20"]

# Create subplots
fig, axs = plt.subplots(1, figsize=(12, 5))

# Plot data1 in the first subplot
axs.plot(x, y, marker=".", color=colors[-1], label="Fisher Value")
axs.set_xlabel("t")
axs.set_ylabel(r"$\hat{F}$", color=colors[-1])
axs.tick_params(axis="y", labelcolor=colors[-1])
axs.set_yscale("log")
ax2 = axs.twinx()
ax2.plot(x, y2, color=colors[0])
ax2.tick_params(axis="y", labelcolor=colors[0])
ax2.set_ylabel(r"$\norm{\delta \mathbf{\theta}}$", color=colors[0])
# Create a folder for plots if it doesn't exist
import os

if not os.path.exists("plots"):
    os.makedirs("plots")
# Save the figure in the "plots" folder
plt.savefig("plots/FisherValue.png")
plt.show()
