import numpy as np
from joblib import Parallel, delayed
from qiskit.circuit.library import EfficientSU2
from tqdm import tqdm

from fidlib.basicfunctions import create_ising
from fidlib.variance import VarianceComputer

np.random.seed(1)
qc = EfficientSU2(num_qubits := 6, reps=(reps := 12))
times = None
vc = VarianceComputer(
    qc=qc,
    initial_parameters=None,
    times=times,
    H=create_ising(num_qubits=num_qubits, j_const=0.5, g_const=-1),
)


omegas = np.linspace(1e-3, 0.08, 20)

variances = Parallel(n_jobs=10)(
    delayed(vc.direct_compute_variance)(200, o) for o in tqdm(omegas)
)

variances = np.array(variances)

result = {"Variance": variances, "Omega": omegas, "Time": times}
np.save("variance_data_qubits.npy", result, allow_pickle=True)
