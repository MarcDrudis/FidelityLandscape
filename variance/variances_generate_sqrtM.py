from itertools import product

import numpy as np
from joblib import Parallel, delayed
from qiskit.circuit.library import EfficientSU2
from tqdm import tqdm

from fidlib.basicfunctions import create_ising
from fidlib.variance import VarianceComputer

np.random.seed(1)


def qubit_variance(num_qubits: int, o: float) -> float:
    qc = EfficientSU2(num_qubits=num_qubits, reps=12)
    times = None
    vc = VarianceComputer(
        qc=qc,
        initial_parameters=None,
        times=times,
        H=create_ising(num_qubits=num_qubits, j_const=0.5, g_const=-1),
    )
    return vc.direct_compute_variance(2000, o)


print(qubit_variance(3, 1))

list_qubits = [10, 11, 12, 13, 14, 15, 16, 17, 18]
list_params = [EfficientSU2(num_qubits=n, reps=12).num_parameters for n in list_qubits]
result = {"qubits": list_qubits}

# jobs = (
#     delayed(qubit_variance)(n, 1 / np.sqrt(list_params[i]))
#     for i, n in enumerate(list_qubits)
# )
# variances = Parallel(n_jobs=15)(jobs)
# result["Varsqrt"] = variances
#
# jobs = (delayed(qubit_variance)(n, 1) for n in list_qubits)
# variances = Parallel(n_jobs=15)(jobs)
# result["Varconst"] = variances
#
# jobs = (
#     delayed(qubit_variance)(n, 0.1 / np.sqrt(list_params[i]))
#     for i, n in enumerate(list_qubits)
# )
# variances = Parallel(n_jobs=15)(jobs)
# result["Varsqrtsmall"] = variances
#
# jobs = (delayed(qubit_variance)(n, 0.1) for n in list_qubits)
# variances = Parallel(n_jobs=15)(jobs)
# result["Varconstsmall"] = variances

for w in [0.001, 0.01, 0.1, 0.25]:
    jobs = (delayed(qubit_variance)(n, w) for n in list_qubits)
    variances = Parallel(n_jobs=15)(jobs)
    result[f"Varconst={w}"] = variances

np.save("variance_data_sqrtM.npy", result, allow_pickle=True)
