import pathlib
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply
from windsurfer.utils.norms import eigenrange

directory = pathlib.Path(__file__).parent.resolve()
plt.style.use(directory.parent / "plots/plot_style.mplstyle")


def infidelity(
    parameters: np.ndarray,
    initial_state: Statevector,
    qc: QuantumCircuit,
    H: SparsePauliOp | None = None,
) -> float:
    state2 = Statevector(qc.assign_parameters(parameters))
    if H is not None:
        state2 = expm_multiply(1.0j * H.to_matrix(sparse=True), state2.data)
        state2 = Statevector(state2 / np.linalg.norm(state2))
    return 1 - state_fidelity(initial_state, state2)


terms = [("Y", -0.95), ("XX", 1)]
data = np.load(
    directory.parent / f"moving_minima/{terms}/moving_minima_qubits=10.npy",
    allow_pickle=True,
).item()

# data = np.load(directory / "candidate.npy", allow_pickle=True).item()
print(data.keys())

qc = data["qc"]
H = data["Hamiltonian"]
index = 4
alternative_params = data["perturbation"][index]
time_of_cut = data["times"][index]
print(f"Time is {time_of_cut}")
print(H.num_qubits, qc.num_qubits)
print(qc.num_parameters)
assert np.isclose(eigenrange(H), 1), f"Not normalized norm={eigenrange(H)}"

initial_state = Statevector(qc.assign_parameters(data["initial_parameters"]))


def cut(
    initial_state: Statevector,
    initial_parameters: np.ndarray,
    direction: np.ndarray,
    qc: QuantumCircuit,
    H: SparsePauliOp | None = None,
):
    # assert np.isclose(np.linalg.norm(direction), 1), "Wrong unit vector"
    jobs = (
        delayed(infidelity)(initial_parameters + direction * p, initial_state, qc, H)
        for p in np.linspace(-0.5, 1.4, 100) * 5
    )
    return Parallel(n_jobs=11)(jobs)


width_document = 510 / 72.27
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(width_document, width_document / 3.2))
count = 0

resolution = 20
grid_axis = np.linspace(-0.8, 1.8, resolution)


def cut2D(
    initial_state: Statevector,
    initial_parameters: np.ndarray,
    direction: np.ndarray,
    qc: QuantumCircuit,
    splitter=np.ndarray,
    H: SparsePauliOp | None = None,
):
    directionA = direction * splitter
    directionB = direction * (1 - splitter)

    grid = product(*(2 * [grid_axis]))
    print("Grid is computed")
    jobs = (
        delayed(infidelity)(
            initial_parameters + directionA * a + directionB * b,
            initial_state.copy(),
            qc,
            H,
        )
        for a, b in grid
    )
    return Parallel(n_jobs=9)(jobs)


images = []

splitters = np.zeros(qc.num_parameters)
splitters[0] = 1
splitters[2] = 1
splitters[-2] = 1
splitters[-1] = 1
splitters = [splitters]


from matplotlib.colors import LogNorm

for splitter in splitters:
    splitter = np.array(splitter)
    image = cut2D(
        initial_state,
        data["initial_parameters"],
        alternative_params,
        qc,
        splitter=splitter,
        H=time_of_cut * H,
        # H=None,
    )
    image = np.array(image).reshape((resolution, resolution))

    heatmap = axs[1].imshow(
        image,
        cmap="viridis_r",
        norm=LogNorm(),
        interpolation="bicubic",
        extent=[-0.8, 1.8, 1.8, -0.8],
    )
    fig.colorbar(heatmap, ax=axs[1])


## Now the 1D cuts


name = "weird_cuts.npy"


if not (directory.parent / "moving_minima" / str(terms) / name).exists():
    # if True:
    data = np.load(
        directory.parent / "moving_minima" / str(terms) / "moving_minima_qubits=10.npy",
        allow_pickle=True,
    ).item()
    print(data.keys())

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
    unit_cuts = [unit_cuts[4]] + unit_cuts[4:]

    landscapes = [
        Parallel(n_jobs=11)(
            get_cuts(data["initial_parameters"], t * data["Hamiltonian"], u)
        )
        for t, u in zip(data["times"], unit_cuts)
    ]
    trajectory = []

    def add_to_trajectory(intermediate_result):
        trajectory.append(intermediate_result.x - data["initial_parameters"])
        print(intermediate_result)

    result = minimize(
        infidelity,
        x0=data["initial_parameters"],
        args=(initial_state, qc, H * time_of_cut),
        callback=add_to_trajectory,
    )
    print("trajectory", len(trajectory))

    cuts_data = {
        "Landscapes": landscapes,
        "cut_samples": cut_samples,
        "times": data["times"],
        "perturbation": data["perturbation"],
        "trajectory": trajectory,
    }

    np.save(
        directory.parent / "moving_minima" / str(terms) / name,
        cuts_data,
        allow_pickle=True,
    )
else:
    print(directory.parent)
    cuts_data = np.load(
        directory.parent / f"moving_minima/{terms}/{name}", allow_pickle=True
    ).item()

projectionsx = np.array(
    [
        np.dot(p - cuts_data["trajectory"][0], splitters[0])
        for p in cuts_data["trajectory"]
    ]
)
projectionsx /= projectionsx[-1]
projectionsy = np.array(
    [
        np.dot(p - cuts_data["trajectory"][0], 1 - splitters[0])
        for p in cuts_data["trajectory"]
    ]
)
projectionsy /= projectionsy[-1]
axs[1].plot(projectionsx, projectionsy, color="red")

# Plotting


colors = [
    "#4056A1",
    "#F13C20",
    "#D79922",
    "#075C2F",
    "#692411",
] * 10

cmap = plt.get_cmap("viridis")
import seaborn as sns

cmap = sns.color_palette("flare", as_cmap=True)
norm = plt.Normalize(cuts_data["times"].min(), cuts_data["times"].max())
line_colors = cmap(norm(cuts_data["times"]))


relevant_times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for l, t, c in zip(cuts_data["Landscapes"], cuts_data["times"], line_colors):
    axs[0].plot(
        cuts_data["cut_samples"] / np.pi,
        l,
        color=c,
        linestyle="-",
        linewidth=1.5 if t in relevant_times else 1,
        alpha=1 if t in relevant_times else 0.5,
        label=rf"$\delta t={t}$" if t in relevant_times else None,
    )
# axs.set_xlabel(r"$\norm{\theta}_{\infty}$")
axs[0].set_xlabel(r"Update Size, $\norm{\bm{\theta}}_{\infty}$")
axs[0].tick_params(axis="x", labelsize=11)
axs[0].set_ylabel(r"Infidelity, $\mathcal{L}(\bm{\theta})$")
axs[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.savefig(directory.parent / f"plots/riverplot.svg")
plt.show()
