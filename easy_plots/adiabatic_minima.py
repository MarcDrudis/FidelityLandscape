import pathlib
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from scipy.sparse.linalg import expm_multiply

directory = pathlib.Path(__file__).parents[1].resolve() / "moving_minima"
name = "weird_cuts.npy"
plt.style.use(directory.parent / "plots/plot_style.mplstyle")
# terms = [("Y", -0.95), ("ZZ", 1)]
terms = str([("Y", -0.95), ("XZ", 1)])

print(directory / terms / name)
if not (directory / terms / name).exists():
    # if True:
    data = np.load(
        directory / terms / "moving_minima_qubits=10.npy",
        allow_pickle=True,
    ).item()
    print(data.keys())
    print(data["Hamiltonian"])

    cut_samples = np.linspace(-np.pi, np.pi, 501) * 2

    def lossfunction(
        perturbation: np.ndarray,
        initial_parameters: np.ndarray,
        H: SparsePauliOp | None = None,
    ) -> float:
        state1 = Statevector(
            data["qc"].assign_parameters(initial_parameters + perturbation)
        )
        state2 = Statevector(data["qc"].assign_parameters(initial_parameters))
        if H is not None:
            state2 = expm_multiply(-1.0j * H.to_matrix(sparse=True), state2.data)
            state2 = Statevector(state2 / np.linalg.norm(state2))

        return 1 - state_fidelity(state1, state2)

    def get_cuts(
        initial_parameters: np.ndarray, H: SparsePauliOp, unit_cut: np.ndarray
    ):
        return (
            delayed(lossfunction)(unit_cut * p, initial_parameters, H)
            for p in cut_samples
        )

    new_parameters_array = data["initial_parameters"] + data["perturbation"]
    unit_cuts = [p / np.linalg.norm(p, ord=np.inf) for p in data["perturbation"][1:]]
    unit_cuts = [unit_cuts[0]] + unit_cuts

    landscapes = [
        Parallel(n_jobs=11)(
            get_cuts(data["initial_parameters"], t * data["Hamiltonian"], u)
        )
        for t, u in zip(data["times"], unit_cuts)
    ]

    cuts_data = {
        "Landscapes": landscapes,
        "cut_samples": cut_samples,
        "times": data["times"],
        "perturbation": data["perturbation"],
    }

    np.save(
        directory / terms / name,
        cuts_data,
        allow_pickle=True,
    )
else:
    cuts_data = np.load(directory / terms / name, allow_pickle=True).item()

# Plotting


import seaborn as sns

cmap = sns.color_palette("flare", as_cmap=True)
norm = plt.Normalize(cuts_data["times"].min(), cuts_data["times"].max())
line_colors = cmap(norm(cuts_data["times"]))[::-1]

width_document = 510 / 72.27
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(width_document, width_document / 3.2))
# axs = [axs]
count = 0
relevant_times = [0, 1.75, 2, 3.75, 4]
for l, t, c in zip(cuts_data["Landscapes"], cuts_data["times"], line_colors):
    # if t in [0, 1, 2, 4, 6]:
    if t in cuts_data["times"]:
        if t == 4:
            continue
        axs[0].plot(
            cuts_data["cut_samples"] / np.pi,
            l,
            color=c,
            linestyle="-" if t not in relevant_times else "-.",
            linewidth=1 if t in relevant_times else 1,
            alpha=1 if t in relevant_times else 0.5,
            label=rf"$\delta t={t*0.04158516:.2f}$" if t in relevant_times else None,
        )
# axs.set_xlabel(r"$\norm{\theta}_{\infty}$")
axs[0].set_xlabel(r"Update Size, $\norm{\bm{\theta}}_{\infty}$")
axs[0].tick_params(axis="x", labelsize=11)
axs[0].set_ylabel(r"Infidelity, $\mathcal{L}(\bm{\theta})$")
axs[0].legend(
    borderpad=0.00001,
)

for n in [4, 6, 8, 10]:
    data_mov = np.load(
        directory / f"{terms}/moving_minima_qubits={n}.npy", allow_pickle=True
    ).item()
    ps = [np.linalg.norm(c, np.inf) for c in data_mov["perturbation"]]
    axs[1].plot(data_mov["times"][:-1], ps[:-1], marker=".", label=f"n={n}")
    # ax2.plot(data_mov["times"][1:], angles, marker="x", label=f"n={n} Angle")

axs[1].set_ylabel(r"Update Size, $\norm{\bm{\theta}}_{\infty}$")
axs[1].tick_params(axis="x", labelsize=11)
axs[1].set_xlabel(r"Time, $\delta t$")
axs[1].legend(
    borderpad=0.00001,
)

plt.savefig(directory.parent / f"plots/adiabatic_minima.svg")
