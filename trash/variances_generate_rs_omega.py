from itertools import product

import numpy as np
from joblib import Parallel, delayed
from qiskit.circuit.library import EfficientSU2
from tqdm import tqdm

from fidlib.basicfunctions import create_ising
from fidlib.variance import VarianceComputer

np.random.seed(1)


def qubit_variance(num_qubits: int, o: float) -> float:
    qc = EfficientSU2(num_qubits=num_qubits, reps=6)
    times = None
    vc = VarianceComputer(
        qc=qc,
        initial_parameters=None,
        times=times,
        H=create_ising(num_qubits=num_qubits, j_const=0.5, g_const=-1),
    )
    return vc.direct_compute_variance(200, o)


list_qubits = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
list_omega = np.logspace(-1, 0, 6)
list_params = [
    EfficientSU2(num_qubits=n, reps=n // 2).num_parameters for n in list_qubits
]
result = {"qubits": list_qubits, "Omega": list_omega}

result["Variances"] = np.zeros((len(list_omega), len(list_qubits)))

for i, n in tqdm(enumerate(list_qubits)):
    jobs = (delayed(qubit_variance)(n, w / np.sqrt(list_params[i])) for w in list_omega)
    variances = Parallel(n_jobs=15)(jobs)
    result["Variances"][:, i] = np.array(variances).reshape((len(list_omega),))
    np.save("variance/variance_data_rs_omega.npy", result, allow_pickle=True)
