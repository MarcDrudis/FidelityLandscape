import numpy as np
from qiskit.circuit.library import EfficientSU2

from fidlib.basicfunctions import create_ising
from fidlib.variance import VarianceComputer

np.random.seed(1)

qc = EfficientSU2(num_qubits := 4)
vc = VarianceComputer(
    qc=qc,
    initial_parameters=None,
    times=(1e-1, 4),
    H=create_ising(num_qubits=num_qubits, j_const=0.5, g_const=-1),
)


omegas = np.logspace(-3, -1, 10)
fidelities = []
for o in omegas:
    fidelities.append(vc.compute_variance(200, 1, o))

print(np.array(fidelities))
