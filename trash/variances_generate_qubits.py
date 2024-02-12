from itertools import product

import numpy as np
from joblib import Parallel, delayed
from qiskit.circuit.library import EfficientSU2
from tqdm import tqdm

from fidlib.basicfunctions import create_ising
from fidlib.variance import VarianceComputer

np.random.seed(1)


def qubit_variance(num_qubits: int, o: float) -> np.ndarray:
    qc = EfficientSU2(num_qubits=num_qubits, reps=12)
    times = None
    vc = VarianceComputer(
        qc=qc,
        initial_parameters=None,
        times=times,
        H=create_ising(num_qubits=num_qubits, j_const=0.5, g_const=-1),
    )
    return vc.direct_compute_variance(200, o)


omegas = np.linspace(1e-3, 0.8, 50)
list_qubits = [4, 5, 6, 7, 8]

jobs = (delayed(qubit_variance)(n, o) for n, o in tqdm(product(list_qubits, omegas)))
variances = Parallel(n_jobs=10)(jobs)

variances = np.array(variances)

result = {
    "Variance": variances,
    "Omega": omegas,
    "qubits": list_qubits,
    "number_params": [
        EfficientSU2(num_qubits=num_qubits, reps=12).num_parameters
        for num_qubits in list_qubits
    ],
}
np.save("variance_data_qubits.npy", result, allow_pickle=True)
