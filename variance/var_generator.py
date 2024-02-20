import pathlib

import numpy as np
from joblib import Parallel, delayed
from qiskit.circuit.library import EfficientSU2
from tqdm import tqdm

from fidlib.basicfunctions import create_ising, get_ansatz
from fidlib.variance import VarianceComputer

np.random.seed(1)
directory = pathlib.Path(__file__).parent.resolve()


def qubit_variance(num_qubits: int, r: float, depth: str, samples: int) -> float:
    """
    Computes the variance for a given quantum circuit.
    Args:
        num_qubits(int): number of qubits of the system
        r(float): side of the hypercube to sample from
        depth(str): "linear" or "const" for the number of repetitions
        of the ansatz
    """
    qc = get_ansatz(num_qubits, depth)
    times = None
    np.random.seed(1)
    vc = VarianceComputer(
        qc=qc,
        initial_parameters=None,
        times=times,
        H=create_ising(num_qubits=num_qubits, j_const=0.5, g_const=-1),
    )
    return vc.direct_compute_variance(samples, r)


samples = 500
list_qubits = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# Constant depth and constant r
file = directory / "var_const_const.npy"
if not file.is_file():
    result = {"qubits": list_qubits}
    list_omega = np.logspace(-1, 0, 6)
    result["omegas"] = list_omega
    result["Variances"] = np.zeros((len(list_omega), len(list_qubits)))
    for i, n in tqdm(enumerate(list_qubits)):
        jobs = (delayed(qubit_variance)(n, w, "const", samples) for w in list_omega)
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
        jobs = (delayed(qubit_variance)(n, w, "linear", samples) for w in list_omega)
        variances = Parallel(n_jobs=15)(jobs)
        result["Variances"][:, i] = np.array(variances).reshape((len(list_omega),))
    np.save(file, result, allow_pickle=True)
else:
    print(file, "already exists")

# const depth and sqrt(r)
file = directory / "var_const_sqrt.npy"
if not file.is_file():
    result = {"qubits": list_qubits}
    list_omega = [0.1, 0.5, 1, 2, 5]
    result["omegas"] = list_omega
    result["Variances"] = np.zeros((len(list_omega), len(list_qubits)))
    depth = "const"
    for i, n in tqdm(enumerate(list_qubits)):
        nparams = get_ansatz(n, depth).num_parameters
        jobs = (
            delayed(qubit_variance)(n, r / np.sqrt(nparams), depth, samples)
            for r in list_omega
        )
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
            delayed(qubit_variance)(n, 2 * nparams**p, depth, samples)
            for p in list_scalings
        )
        variances = Parallel(n_jobs=15)(jobs)
        result["Variances"][:, i] = np.array(variances).reshape((len(list_scalings),))
    np.save(file, result, allow_pickle=True)
else:
    print(file, "already exists")
