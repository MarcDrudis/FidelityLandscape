import pathlib

import matplotlib.pyplot as plt
import numpy as np

directory = pathlib.Path(__file__).parent.resolve()

# Set a consistent color palette
colors = ["#4056A1", "#075C2F", "#7D8238", "#453F3F", "#692411", "#D79922", "#F13C20"]

plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)
plt.rc("font", family="Times New Roman")
plt.rcParams.update({"errorbar.capsize": 2})


# Make Figure constant depth constant r
fig, axs = plt.subplots(1, 1, figsize=(7, 5))

variances = np.load(directory / "var_const_const.npy", allow_pickle=True).item()
qubits = variances["qubits"]
omegas = variances["omegas"]
variances = variances["Variances"]

for i, o in enumerate(omegas):
    axs.plot(
        qubits,
        variances[i],
        marker=".",
        markersize=10,
        # color=colors[i],
        label=r"$r=$" + f"{o:.1e}",
    )
axs.set_xlabel(r"n")
axs.set_ylabel(r"Var[$\mathcal{L}$]")
axs.set_yscale("log")
axs.set_xticks(qubits)
axs.legend(loc="lower left")
axs.set_title(f"Variance for constant depth")
plt.show()


# # Make Figure linear depth constant r
# fig, axs = plt.subplots(1, 1, figsize=(7, 5))
#
# variances = np.load(directory / "var_linear_const.npy", allow_pickle=True).item()
# qubits = variances["qubits"]
# omegas = variances["omegas"]
# variances = variances["Variances"]
#
# for i, o in enumerate(omegas):
#     axs.plot(
#         qubits,
#         variances[i],
#         marker=".",
#         markersize=10,
#         # color=colors[i],
#         label=r"$r=$" + f"{o:.1e}",
#     )
# axs.set_xlabel(r"n")
# axs.set_ylabel(r"Var[$\mathcal{L}$]")
# axs.set_yscale("log")
# axs.set_xticks(qubits)
# axs.legend(loc="lower left")
# axs.set_title(f"Variance for linear depth")
# plt.show()


# Make Figure linear depth constant r
fig, axs = plt.subplots(1, 1, figsize=(7, 5))

variances = np.load(directory / "var_const_sqrt.npy", allow_pickle=True).item()
qubits = variances["qubits"]
omegas = variances["omegas"]
variances = variances["Variances"]

for i, o in enumerate(omegas):
    axs.plot(
        qubits,
        variances[i],
        marker=".",
        markersize=10,
        # color=colors[i],
        label=r"$r=\frac{1}{\sqrt{m}}$" + f"{o:.1e}",
    )
axs.set_xlabel(r"n")
axs.set_ylabel(r"Var[$\mathcal{L}$]")
axs.set_yscale("log")
axs.set_xticks(qubits)
axs.legend(loc="lower left")
axs.set_title(f"Variance for linear depth")
plt.show()


# Make Figure const depth different scalings for r
fig, axs = plt.subplots(1, 1, figsize=(7, 5))

name = "var_const_scalings"
variances = np.load(directory / (name + ".npy"), allow_pickle=True).item()
qubits = variances["qubits"]
scalings = variances["scalings"]
variances = variances["Variances"]

for i, p in enumerate(
    [
        r"$r=5 \cdot 10^{1} m^{-\frac{1}{4}}$",
        r"$r=5 \cdot 10^{1} m^{-\frac{1}{2}}$",
        r"$r=5 \cdot 10^{1} m^{-\frac{3}{4}}$",
        r"$r=5 \cdot 10^{1} m^{-1}$",
    ]
):
    axs.plot(
        qubits,
        variances[i],
        marker=".",
        markersize=10,
        # color=colors[i],
        label=p,
    )
axs.set_xlabel(r"n")
axs.set_ylabel(r"Var[$\mathcal{L}$]")
axs.set_yscale("log")
axs.set_xticks(qubits)
axs.legend(loc="lower left")
axs.set_title(f"Variance for constant depth and different scalings")
plt.savefig("plots/" + name + ".jpg")
plt.show()
