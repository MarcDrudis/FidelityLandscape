from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from qiskit.circuit.library import EfficientSU2
from scipy.interpolate import CubicSpline

from fidlib.variance import VarianceComputer, _abs2, bound, kplus

variances = np.load("variance/variance_data_rs_linear.npy", allow_pickle=True).item()
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
mksize = 5
for i, o in enumerate(omegas):
    axs.plot(
        qubits,
        variances[i],
        marker=".",
        markersize=2 * mksize,
        # color=colors[i],
        label=r"$r=$" + f"{o:.1e}",
    )
axs.set_xlabel(r"n")
axs.set_ylabel(r"Var[$\mathcal{L}$]")
axs.set_yscale("log")
axs.set_xticks(qubits)
axs.legend(loc="lower left")
axs.set_title(f"Variance for linear depth")
# axs.text(-0.1, 1.05, "(a)", transform=axs.transAxes, fontsize=14, fontweight="bold")

# Create a folder for plots if it doesn't exist
import os

if not os.path.exists("plots"):
    os.makedirs("plots")
# Save the figure in the "plots" folder
plt.savefig("plots/variance_lindepth_rconst.png")
plt.show()
