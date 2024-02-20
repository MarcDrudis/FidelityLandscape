import pathlib
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from qiskit.quantum_info import Statevector, state_fidelity
from scipy.interpolate import CubicSpline

from fidlib.basicfunctions import get_ansatz
from fidlib.variance import VarianceComputer

np.random.seed(1)
directory = pathlib.Path(__file__).parent.resolve()


def infi(num_qubits: int, r: float, depth: int, seed: int):
    qc = get_ansatz(int(num_qubits), depth)
    initial_parameters = initial_parameters_list[num_qubits]
    np.random.seed(seed)
    direction = np.random.uniform(-np.pi, np.pi, qc.num_parameters)
    return state_fidelity(
        Statevector(qc.assign_parameters(initial_parameters)),
        Statevector(
            qc.assign_parameters(
                initial_parameters + direction / np.linalg.norm(direction) * r
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
    times = None
    vc = VarianceComputer(
        qc=qc,
        initial_parameters=initial_parameters_list[num_qubits],
        # initial_parameters=None,
        times=times,
        H=None,
    )

    return vc.direct_compute_variance(samples, r)


rs = np.logspace(-2, 0, 50) * np.pi
qubits = np.arange(4, 14)
depth = "const"
rng_initial_parameters = np.random.default_rng(0)
initial_parameters_list = [
    rng_initial_parameters.uniform(
        -np.pi, np.pi, get_ansatz(int(n), depth).num_parameters
    )
    for n in range(20)
]

name_variance = "var_shape.npy"
if not (directory / name_variance).is_file():
    print("simulating")
    jobs = (delayed(qubit_variance)(n, r, depth, 500) for r, n in product(rs, qubits))
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


name_landscape = "landscape_shape.npy"
if not (directory / name_landscape).is_file():
    print("simulating landscape")
    N_directions = 20
    jobs = (
        delayed(infi)(n, r, "const", seed)
        for r, n, seed in product(rs, qubits, range(N_directions))
    )
    landscape = Parallel(n_jobs=11)(jobs)
    print(len(landscape))
    landscape = np.array(landscape).reshape((len(rs), len(qubits), N_directions))
    print(landscape.shape)
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
    np.min(result_landscape["landscapes"][:, i, :], axis=1).T
    for i, q in enumerate(result["qubits"])
]

print(result["rs"])

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

plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)
plt.rc("font", family="Times New Roman")
plt.rcParams.update({"errorbar.capsize": 2})

fig, axs = plt.subplots(1, 1, figsize=(14, 10))
ax2 = axs.twinx()
maximas = list()
maxima_value = list()


for i, n in enumerate(qubits):
    resolution_rs = np.logspace(-2, 0, 1000)
    interpolated_variance = CubicSpline(result["rs"] / np.pi, result["variances"][i])(
        resolution_rs / np.pi
    )
    axs.scatter(
        x=result["rs"] / np.pi,
        y=result["variances"][i],
        # label=f"n={n}",
        marker="x",
        color=colors[i],
    )
    axs.plot(
        resolution_rs / np.pi,
        interpolated_variance,
        label=f"n={n}",
        color=colors[i],
    )
    ax2.plot(
        result["rs"] / np.pi,
        result["landscapes"][i],
        label=f"n={n}",
        color=colors[i],
        marker=".",
    )
    maximas.append(resolution_rs[np.argmax(interpolated_variance)] / np.pi)
    maxima_value.append(np.max(interpolated_variance))
    axs.vlines(
        x=maximas[-1],
        # x=2 * (get_ansatz(n, "const").num_parameters) ** (-1 / 2),
        # x=1.2 * (get_ansatz(n, "const").num_parameters) ** (-1 / 2),
        ymin=0,
        ymax=2e-2,
        color=colors[i],
    )

axs.set_xlabel(r"$\frac{r}{ \pi}$")
# axs.set_yscale("log")
# axs.set_xscale("log")
axs.legend()
plt.show()

plt.scatter(result["qubits"], maximas, label=r"$r_{max}$")
plt.plot(
    result["qubits"], result["qubits"] ** (-0.55) * 0.35, label=r"0.35*$n^{-0.55}$"
)
plt.title("r that maximizes the variance")
plt.legend()
plt.show()

plt.scatter(result["qubits"], maxima_value, label="Maxima Value")
plt.plot(result["qubits"], result["qubits"] ** (-2.0) * 0.2, label=r"$0.2*n^{-2}$")
plt.title("Maximum value of the Variance")
plt.legend()
plt.show()
