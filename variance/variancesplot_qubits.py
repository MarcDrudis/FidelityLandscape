from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from qiskit.circuit.library import EfficientSU2
from scipy.interpolate import CubicSpline

from fidlib.variance import VarianceComputer, _abs2, bound, kplus

variances = np.load("variance_data_qubits.npy", allow_pickle=True).item()

data = [
    (variances["Variance"][i][0], *z)
    for i, z in enumerate(product(variances["qubits"], variances["Omega"]))
]

df = pd.DataFrame(data)

df.columns = ["Variance", "n", "Omega"]
df["n"] = df["n"].astype("category")


# Set a consistent color palette
colors = ["#4056A1", "#075C2F", "#7D8238", "#453F3F", "#692411", "#D79922", "#F13C20"]

plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)
plt.rc("font", family="Times New Roman")
plt.rcParams.update({"errorbar.capsize": 2})

# Create subplots
fig, axs = plt.subplots(1, 1, figsize=(7, 5))

# Plot data1 in the first subplot
for n, c in zip(
    range(4, 8),
    ["#4056A1", "#075C2F", "#D79922", "#F13C20"],
):
    x = df[df["n"] == n]["Omega"]
    y = df[df["n"] == n]["Variance"]
    axs.plot(
        x,
        y,
        marker=".",
        color=c,
        label=r"$n_{qubits}$=" + f"{n}",
    )
axs.set_xlabel(r"$\omega$")
axs.set_ylabel(r"Var[$\mathcal{L}$]")
axs.legend()
axs.set_title("Experimental Variance vs Bound")
# axs.text(-0.1, 1.05, "(a)", transform=axs.transAxes, fontsize=14, fontweight="bold")

# Create a folder for plots if it doesn't exist
import os

if not os.path.exists("plots"):
    os.makedirs("plots")
# Save the figure in the "plots" folder
plt.savefig("plots/variance.png")
plt.show()
