import pathlib
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from qiskit.pulse import num_qubits
from qiskit.quantum_info import Statevector, state_fidelity
from scipy.interpolate import CubicSpline

from fidlib.basicfunctions import get_ansatz
from fidlib.variance import VarianceComputer

directory = pathlib.Path(__file__).parent.resolve()
plt.style.use(directory.parent / "plots/plot_style.mplstyle")
depth = "linear"


def infi(num_qubits: int, r: float, depth: int, seed: int):
    qc = get_ansatz(int(num_qubits), depth)
    initial_parameters = initial_parameters_list[num_qubits]
    direction = np.random.default_rng(seed).uniform(-np.pi, np.pi, qc.num_parameters)
    return state_fidelity(
        Statevector(qc.assign_parameters(initial_parameters)),
        Statevector(
            qc.assign_parameters(
                initial_parameters + direction / np.linalg.norm(direction, np.inf) * r
            )
        ),
    )


def qubit_variance(num_qubits: int, r: float, depth: str, samples: int) -> float:
    """
    Computes the variance for a given quantum circuit.
    Args:
        num_qubits(int): number of qubits of the system
        r(float): side of the hypercube to sample from
        depth(str): "linear" or "const" for the number of repetitions
        of the ansatz
    """
    qc = get_ansatz(int(num_qubits), depth)
    vc = VarianceComputer(
        qc=qc,
        initial_parameters=initial_parameters_list[num_qubits],
        times=None,
        H=None,
    )

    return vc.direct_compute_variance(samples, r)


rs = np.logspace(-1.5, 0, 50) * np.pi
qubits = np.arange(4, 14)
rng_initial_parameters = np.random.default_rng(0)
initial_parameters_list = [
    rng_initial_parameters.uniform(
        -np.pi, np.pi, get_ansatz(int(n), depth).num_parameters
    )
    for n in range(20)
]

name_variance = f"var_shape_{depth}.npy"
if not (directory / name_variance).is_file():
    print("simulating")
    jobs = (
        delayed(qubit_variance)(
            n,
            r,
            depth,
            # 20000,
            5000,
        )
        for r, n in product(rs, qubits)
    )
    variances = Parallel(n_jobs=11)(jobs)
    variances = np.array(variances).reshape((len(rs), len(qubits))).T

    result_variance = {
        "qubits": qubits,
        "rs": rs,
        "variances": variances,
    }
    np.save(directory / name_variance, result_variance, allow_pickle=True)
else:
    print("loading")
    result_variance = np.load(directory / name_variance, allow_pickle=True).item()


name_landscape = f"landscape_shape_{depth}.npy"
if not (directory / name_landscape).is_file():
    print("simulating landscape")
    N_directions = 100
    jobs = (
        delayed(infi)(n, r, depth, seed)
        for r, n, seed in product(rs, qubits, range(N_directions))
    )
    landscape = Parallel(n_jobs=11)(jobs)
    landscape = np.array(landscape).reshape((len(rs), len(qubits), N_directions))
    result_landscape = {
        "qubits": qubits,
        "rs": rs,
        "landscapes": landscape,
    }
    np.save(directory / name_landscape, result_landscape, allow_pickle=True)
else:
    print("loading landscape")
    result_landscape = np.load(directory / name_landscape, allow_pickle=True).item()

result = result_variance
result["landscapes"] = [
    np.mean(result_landscape["landscapes"][:, i, :], axis=1).T
    for i, _ in enumerate(result["qubits"])
]
result["rs_landscape"] = result_landscape["rs"]
result["num_parameters"] = np.array(
    [get_ansatz(n, depth).num_parameters for n in result["qubits"]]
)

###########################Plotting
# Set a consistent color palette
colors = [
    "#4056A1",
    "#075C2F",
    "#7D8238",
    "#453F3F",
    "#692411",
    "#D79922",
    "#F13C20",
] * 2

# fig, axs = plt.subplots(3, 1, figsize=(5, 12))
# fig.tight_layout(pad=1.0)


fig = plt.figure(layout="constrained", figsize=(8, 4))
subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1.5, 1])
axA = subfigs[0].subplots(1, 1)
axB = subfigs[1].subplots(2, 1)

axs = [axA, axB[0], axB[1]]

ax2 = axs[0].twinx()
maximas = list()
maxima_value = list()


for i, n in enumerate(qubits):
    resolution_rs = np.logspace(-1.5, 0, 1000) * np.pi
    interpolated_variance = CubicSpline(result["rs"] / np.pi, result["variances"][i])(
        resolution_rs / np.pi
    )
    maximas.append(resolution_rs[np.argmax(interpolated_variance)] / np.pi)
    maxima_value.append(np.max(interpolated_variance))
    if n % 2 == 0:
        axs[0].scatter(
            x=result["rs"] / np.pi,
            y=result["variances"][i],
            # label=f"n={n}",
            marker=".",
            color=colors[i],
        )
        axs[0].plot(
            resolution_rs / np.pi,
            interpolated_variance,
            label=f"n={n}",
            color=colors[i],
        )
        ax2.plot(
            result["rs_landscape"] / np.pi,
            1 - result["landscapes"][i],
            label=f"n={n}",
            color=colors[i],
            marker="x",
            linestyle="--",
            alpha=0.4,
        )
        axs[0].vlines(
            x=maximas[-1],
            ymin=0,
            ymax=2e-2,
            color=colors[i],
        )

axs[0].set_xlabel(r"Size of Hypercube, $\frac{r}{ \pi}$")
axs[0].set_ylabel(r"Var. in Hypercube, $\mathrm{Var}[ \mathcal{L} ]$")
ax2.set_ylabel(r"Infidelity, $\mathcal{L}(\norm{\mathbf{\theta}}_{\infty}=r)$")
axs[0].set_yscale("log")
axs[0].set_xscale("log")


from matplotlib.legend_handler import HandlerBase


class MarkerHandler(HandlerBase):
    def create_artists(
        self, legend, tup, xdescent, ydescent, width, height, fontsize, trans
    ):
        return [
            plt.Line2D(
                [width / 2],
                [height / 2.0],
                ls="",
                marker=tup[1],
                color=tup[0],
                transform=trans,
            )
        ]


print(
    [(colors[i], "s") for i in range(0, len(qubits), 2)],
    [f"n={n}" for n in qubits[::2]],
)
axs[0].legend(
    [(colors[i], "s") for i in range(0, len(qubits), 2)],
    [f"n={n}" for n in qubits[::2]],
    handler_map={tuple: MarkerHandler()},
    bbox_to_anchor=(0, 1.02, 1, 0.2),
    loc="lower left",
    mode="expand",
    handletextpad=0,
    borderaxespad=0,
    ncol=5,
)
ax2.legend(
    [("black", "."), ("black", "x")],
    [r"$\mathrm{Var}[\mathcal{L}]$", r"$\mathcal{L}$"],
    handler_map={tuple: MarkerHandler()},
    bbox_to_anchor=(-0.03, 1),
    loc="upper left",
    handletextpad=0,
    borderaxespad=0,
    ncol=1,
    frameon=False,
)
########Plot scalings
coeff, prefactor = np.polyfit(np.log10(result["num_parameters"]), np.log10(maximas), 1)
axs[1].scatter(result["num_parameters"], maximas, label=r"$r_{max}$", color=colors[0])
axs[1].plot(
    result["num_parameters"],
    result["num_parameters"] ** coeff * 10**prefactor,
    label=f"${{{10**prefactor:.2f}}}m^{{{coeff:.2f}}}$",
    color=colors[1],
)
axs[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.05, 1.05))
axs[1].set_yscale("log", base=2)
axs[1].set_xscale("log", base=2)
axs[1].tick_params(labelbottom=False)
# axs[1].set_xlabel(r"$m$")
axs[1].set_ylabel(r"Argmax of Var., $r_{max}$")

coeff, prefactor = np.polyfit(
    np.log10(result["num_parameters"]), np.log10(maxima_value), 1
)
axs[2].scatter(
    result["num_parameters"],
    maxima_value,
    label=r"Var$[\mathcal{L}]_{max}$",
    color=colors[0],
)
axs[2].plot(
    result["num_parameters"],
    result["num_parameters"] ** coeff * 10**prefactor,
    label=f"${{{10**prefactor:.2f}}}m^{{{coeff:.2f}}}$",
    color=colors[1],
)
axs[2].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.05, 1.05))
axs[2].set_xlabel(r"Number of Parameters, $m$")
axs[2].set_ylabel(r"Max. Var. Value")
axs[2].set_yscale("log", base=2)
axs[2].set_xscale("log", base=2)
fig.savefig(directory.parent / f"plots/variance_{depth}.svg")
fig.savefig(directory.parent / f"plots/variance_{depth}.png")
fig.savefig(directory.parent / f"plots/variance_{depth}.pdf")
