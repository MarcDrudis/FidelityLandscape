import pathlib
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
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
        state2 = expm_multiply(-1.0j * H.to_matrix(sparse=True), state2.data)
        state2 = Statevector(state2 / np.linalg.norm(state2))
    return 1 - state_fidelity(initial_state, state2)


# terms = [("Y", -0.96), ("XX", 1)]
# data = np.load(
#     directory.parent / f"moving_minima/{terms}/moving_minima_qubits=4.npy",
#     allow_pickle=True,
# ).item()
# alternative_params = data["perturbation"][1]
# qc = data["qc"]
# initial_state = Statevector(qc.assign_parameters(data["initial_parameters"]))
# new_minima = minimize(
#     infidelity, alternative_params + data["initial_parameters"], (initial_state, qc)
# )
# data["perturbation"] = new_minima.x - data["initial_parameters"]
# data = np.save(directory / "candidate.npy", data, allow_pickle=True)

data = np.load(directory / "candidate.npy", allow_pickle=True).item()
print(data.keys())
print(data["perturbation"].shape)

qc = data["qc"]
H = data["Hamiltonian"]
print(H.num_qubits, qc.num_qubits)
print(qc.num_parameters)
alternative_params = data["perturbation"]
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


time_of_cut = 1.5
print(
    infidelity(
        alternative_params + data["initial_parameters"],
        initial_state,
        qc,
        time_of_cut * H,
    )
)
print(np.round(alternative_params / np.pi, 2))

print(np.linalg.norm(alternative_params))
plt.plot(
    cut(
        initial_state,
        data["initial_parameters"],
        alternative_params / np.linalg.norm(alternative_params),
        qc,
        time_of_cut * H,
    )
)
plt.grid()
plt.show()

resolution = 20
grid_axis = np.linspace(-0.8, 1.8, resolution)
grid = product(*(2 * [grid_axis]))


def cut2D(
    initial_state: Statevector,
    initial_parameters: np.ndarray,
    direction: np.ndarray,
    qc: QuantumCircuit,
    splitter=np.ndarray,
    H: SparsePauliOp | None = None,
):
    # assert np.isclose(np.linalg.norm(direction), 1), "Wrong unit vector"
    directionA = direction * splitter
    directionB = direction * (1 - splitter)

    # assert np.isclose(
    #     np.dot(directionA, directionB), 0
    # ), "The directions in 2D cut not orth"
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

splitters = [
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
]
from matplotlib.colors import LogNorm

for splitter in splitters:
    splitter = np.array(splitter)
    image = cut2D(
        initial_state,
        data["initial_parameters"],
        alternative_params,
        qc,
        splitter=splitter,
    )
    # image = np.log10(image)
    image = np.array(image).reshape((resolution, resolution))

    fig, ax = plt.subplots(1, 1)
    heatmap = ax.imshow(
        image, cmap="viridis_r", norm=LogNorm(), extent=[-0.8, 1.8, 1.8, -0.8]
    )
    print(data["perturbation"][-1].dot(splitters[0]))
    ax.scatter(x=[0, 1], y=[0, 1], marker="x")
    fig.colorbar(heatmap, ax=ax)
    plt.show()
