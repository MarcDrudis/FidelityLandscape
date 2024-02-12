import pathlib

import numpy as np
from joblib import Parallel, delayed
from qiskit.circuit.library import EfficientSU2
from qiskit.pulse import num_qubits
from tqdm import tqdm

from fidlib.basicfunctions import create_ising
from fidlib.variance import VarianceComputer

np.random.seed(1)
directory = pathlib.Path(__file__).parent.resolve()


def get_ansatz(num_qubits: int, depth: str):
    """
    Creates an ansatz with a given number of qubits and a depth that scales
    either linearly or is constant with respect to number of qubits.
    """
    if depth not in ("linear", "const"):
        raise ValueError("Depth must be either 'linear' of 'const' ")
    reps = 6 if depth == "linear" else num_qubits // 2
    return EfficientSU2(num_qubits=num_qubits, reps=reps)


def qubit_variance(num_qubits: int, r: float, depth: str) -> float:
    """
    Computes the variance for a given quantum circuit.
    Args:
        num_qubits(int): number of qubits of the system
        r(float): space in the hypercube to sample from
        depth(str): "linear" or "const" for the number of repetitions
        of the ansatz
    """
    qc = get_ansatz(num_qubits, depth)
    times = None
    vc = VarianceComputer(
        qc=qc,
        initial_parameters=None,
        times=times,
        H=create_ising(num_qubits=num_qubits, j_const=0.5, g_const=-1),
    )
    return vc.direct_compute_variance(500, r)


list_qubits = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# Constant depth and constant r
file = directory / "var_const_const.npy"
if not file.is_file():
    result = {"qubits": list_qubits}
    list_omega = np.logspace(-1, 0, 6)
    result["omegas"] = list_omega
    result["Variances"] = np.zeros((len(list_omega), len(list_qubits)))
    for i, n in tqdm(enumerate(list_qubits)):
        jobs = (delayed(qubit_variance)(n, w, "const") for w in list_omega)
        variances = Parallel(n_jobs=15)(jobs)
        result["Variances"][:, i] = np.array(variances).reshape((len(list_omega),))
    np.save(file, result, allow_pickle=True)
else:
    print(file, "already exists")


# linear depth and constant r
file = directory / "var_linear_const.npy"
if not file.is_file():
    result = {"qubits": list_qubits}
    list_omega = np.logspace(-1, 0, 6)
    result["omegas"] = list_omega
    result["Variances"] = np.zeros((len(list_omega), len(list_qubits)))
    for i, n in tqdm(enumerate(list_qubits)):
        jobs = (delayed(qubit_variance)(n, w, "linear") for w in list_omega)
        variances = Parallel(n_jobs=15)(jobs)
        result["Variances"][:, i] = np.array(variances).reshape((len(list_omega),))
    np.save(file, result, allow_pickle=True)
else:
    print(file, "already exists")


# linear depth and constant r
file = directory / "var_const_scalings.npy"
if not file.is_file():
    result = {"qubits": list_qubits}
    list_scalings = [-1 / 4, -1 / 2, -3 / 4, -1]
    result["scalings"] = list_scalings
    result["Variances"] = np.zeros((len(list_scalings), len(list_qubits)))
    depth = "const"
    for i, n in tqdm(enumerate(list_qubits)):
        nparams = get_ansatz(n, depth).num_parameters
        jobs = (
            delayed(qubit_variance)(n, 1e-4 * nparams**p, depth)
            for p in list_scalings
        )
        variances = Parallel(n_jobs=15)(jobs)
        result["Variances"][:, i] = np.array(variances).reshape((len(list_scalings),))
    np.save(file, result, allow_pickle=True)
else:
    print(file, "already exists")
