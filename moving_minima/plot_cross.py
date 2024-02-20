import pathlib

import matplotlib.pyplot as plt
import numpy as np

directory = pathlib.Path(__file__).parent.resolve()
data = np.load(directory / "data_crossing.npy")
x = data["times"]
initial_minima = data["global_inf"]
alternative_minima = data["local_inf"]

# Set a consistent color palette
colors = ["#4056A1", "#075C2F", "#7D8238", "#453F3F", "#692411", "#D79922", "#F13C20"]

plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)
plt.rc("font", family="Times New Roman")
plt.rcParams.update({"errorbar.capsize": 2})

# Create subplots
fig, ax = plt.subplots(figsize=(12, 5))

# Plot data1 in the first subplot
ax.plot(x, initial_minima, marker=".", color=colors[-2], label="Initial Minima")
ax.plot(x, alternative_minima, color=colors[0], label="Alternative Minima")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathcal{L}(\mathbb{\theta},t)$")
ax.legend()
ax.text(-0.1, 1.05, "(a)", transform=ax[0].transAxes, fontsize=14, fontweight="bold")

# Create a folder for plots if it doesn't exist
import os

if not os.path.exists("plots"):
    os.makedirs("plots")

# Save the figure in the "plots" folder
plt.savefig("plots/figure.png")
