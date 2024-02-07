import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import EfficientSU2
from scipy.interpolate import CubicSpline

from fidlib.variance import VarianceComputer, _abs2, bound, kplus

variances = np.load("varianc_data.npy", allow_pickle=True).item()
lower_bound = np.array([bound(o, 1, 156) for o in variances["Omega"]])


# Set a consistent color palette
colors = ["#4056A1", "#075C2F", "#7D8238", "#453F3F", "#692411", "#D79922", "#F13C20"]

plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)
plt.rc("font", family="Times New Roman")
plt.rcParams.update({"errorbar.capsize": 2})

# Create subplots
fig, axs = plt.subplots(1, 1, figsize=(7, 5))

# Plot data1 in the first subplot
axs.plot(
    variances["Omega"],
    variances["Variance"][:, 0],
    marker=".",
    color=colors[0],
    label="Experimental Variance",
)
axs.set_xlabel(r"$\omega$")
axs.set_ylabel(r"Var[$\mathcal{L}$]", color=colors[0])
axs.tick_params(axis="y", labelcolor=colors[0])
# axs.set_yscale("log")
# axs.legend()
axs.set_title("Experimental Variance vs Bound")
# axs.text(-0.1, 1.05, "(a)", transform=axs.transAxes, fontsize=14, fontweight="bold")
axs2 = axs.twinx()
axs2.set_xlabel(r"$\omega$")
axs2.set_ylabel(r"Var[$\mathcal{L}$]", color=colors[-2])
axs2.tick_params(axis="y", labelcolor=colors[-2])
axs2.scatter(
    variances["Omega"],
    lower_bound,
    marker=".",
    color=colors[-2],
    label="Lower Bound",
)
xinter = np.linspace(0, variances["Omega"][-1], 10000)
interpolate = CubicSpline(variances["Omega"], lower_bound)

axs2.plot(
    xinter,
    interpolate(xinter),
    marker="",
    color=colors[-2],
    label="Lower Bound",
)

# Create a folder for plots if it doesn't exist
import os

if not os.path.exists("plots"):
    os.makedirs("plots")
# Save the figure in the "plots" folder
plt.savefig("plots/variance.png")
plt.show()
