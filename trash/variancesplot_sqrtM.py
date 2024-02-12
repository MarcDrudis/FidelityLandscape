from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from qiskit.circuit.library import EfficientSU2
from scipy.interpolate import CubicSpline

from fidlib.variance import VarianceComputer, _abs2, bound, kplus

variances = np.load("variance_data_sqrtM.npy", allow_pickle=True).item()
qubits = variances["qubits"]
omegas = variances["Omega"]
variances = variances["Variances"]


# Set a consistent color palette
colors = ["#4056A1", "#075C2F", "#7D8238", "#453F3F", "#692411", "#D79922", "#F13C20"]

plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)
plt.rc("font", family="Times New Roman")
plt.rcParams.update({"errorbar.capsize": 2})

# Create subplots
fig, axs = plt.subplots(1, 1, figsize=(7, 5))

# Plot data1 in the first subplot
axs2 = axs.twinx()
mksize = 5
axs.plot(
    qubits,
    np.ravel(variances["Varsqrt"]),
    marker=".",
    markersize=2 * mksize,
    color=colors[0],
    label=r"$\omega=1/\sqrt{m}$",
)
axs2.plot(
    qubits,
    np.ravel(variances["Varsqrtsmall"]),
    marker=".",
    markersize=2 * mksize,
    linestyle="--",
    color=colors[-1],
    label=r"$\omega=0.1/\sqrt{m}$",
)
axs.plot(
    qubits,
    np.ravel(variances["Varconst"]),
    marker="s",
    markersize=mksize,
    color=colors[0],
    label=r"$\omega=1$",
)
axs2.plot(
    qubits,
    np.ravel(variances["Varconstsmall"]),
    marker="s",
    markersize=mksize,
    linestyle="--",
    color=colors[-1],
    label=r"$\omega=0.1$",
)
axs.set_xlabel(r"n")
axs.set_ylabel(r"Var[$\mathcal{L}$]")
axs.set_yscale("log")
axs.tick_params(axis="y", colors=colors[0])
axs.spines["left"].set_color(colors[0])
axs.set_xticks(qubits)
axs.legend(loc="lower left")
axs2.set_yscale("log")
axs2.legend(loc="right")
# axs2.set_ylabel(r"Var[$\mathcal{L}$]", color=colors[-1])
axs2.tick_params(axis="y", labelcolor=colors[-1])
axs.spines["right"].set_color(colors[-1])
axs.set_title(f"Experimental Variance vs System Size")
# axs.text(-0.1, 1.05, "(a)", transform=axs.transAxes, fontsize=14, fontweight="bold")

# Create a folder for plots if it doesn't exist
import os

if not os.path.exists("plots"):
    os.makedirs("plots")
# Save the figure in the "plots" folder
plt.savefig("plots/variance.png")
plt.show()
