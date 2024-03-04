import pathlib

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from qiskit.quantum_info import Statevector, state_fidelity

directory = pathlib.Path(__file__).parent.resolve()
name = "weird_cuts.npy"
plt.style.use(directory.parent / "plots/plot_style.mplstyle")


if not (directory / name).exists():
    data = np.load(directory / "moving_minima.npy", allow_pickle=True).item()
    print(data.keys())

    cut_samples = np.linspace(-np.pi, np.pi, 501) * 4

    def lossfunction(
        new_parameters: np.ndarray, initial_parameters: np.ndarray
    ) -> float:
        state1 = Statevector(data["qc"].assign_parameters(new_parameters))
        state2 = Statevector(data["qc"].assign_parameters(initial_parameters))

        return 1 - state_fidelity(state1, state2)

    def get_cuts(initial_parameters: np.ndarray, new_parameters: np.ndarray, unit_cut):
        return (
            delayed(lossfunction)(unit_cut * p + initial_parameters, new_parameters)
            for p in cut_samples
        )

    # landscape_cuts = np.diff(data["perturbation"], axis=0)
    # landscape_cuts = np.concatenate(
    #     (landscape_cuts[0][np.newaxis, :], landscape_cuts), axis=0
    # )
    # print(landscape_cuts.shape, data["times"].shape, data["initial_parameters"].shape)
    #
    # landscapes = [
    #     Parallel(n_jobs=11)(
    #         get_cuts(initial_parameters=data["initial_parameters"] + per, cut=l)
    #     )
    #     for per, l in zip(data["perturbation"][1:], landscape_cuts)
    # ]

    new_parameters_array = data["initial_parameters"] + data["perturbation"]
    unit_cuts = [p / np.linalg.norm(p, np.inf) for p in data["perturbation"][1:]]
    unit_cuts = unit_cuts[0] + unit_cuts

    landscapes = [
        Parallel(n_jobs=11)(
            get_cuts(data["initial_parameters"], data["initial_parameters"] + per, u)
        )
        for per, u in zip(data["perturbation"], unit_cuts)
    ]

    cuts_data = {
        "Landscapes": landscapes,
        "cut_samples": cut_samples,
        "times": data["times"],
        "perturbation": data["perturbation"],
    }

    np.save(
        directory / name,
        cuts_data,
        allow_pickle=True,
    )
else:
    cuts_data = np.load(directory / name, allow_pickle=True).item()

# Plotting


colors = [
    "#4056A1",
    "#F13C20",
    "#692411",
    "#D79922",
    "#075C2F",
]


width_document = 246 / 72.27
# Create subplots
fig, axs = plt.subplots(1, 1, figsize=(width_document, width_document / 1.6))
count = 0
for pert, l, t, c in zip(
    cuts_data["perturbation"], cuts_data["Landscapes"], cuts_data["times"], colors
):
    count += 1
    axs.plot(
        (cuts_data["cut_samples"] + np.linalg.norm(pert, ord=np.inf)) / np.pi,
        l,
        label=f"$\delta t={t:.1f}$",
        color=c,
    )
axs.set_xlabel(r"$\norm{\theta}_{\infty}$")
axs.tick_params(axis="x", labelsize=11)
axs.set_xlim((-1.5, 2))
axs.set_ylabel(r"$\mathcal{L}(\theta)$")
axs.legend()

plt.savefig(directory.parent / "plots/weird_plot.svg")
