import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit, parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector, state_fidelity
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm

from fidlib.basicfunctions import create_heisenberg, create_ising


def main():
    np.random.seed(2)

    num_qubits = 3

    H = create_ising(num_qubits, 0.3, -0.9, nonint_const=0.1).simplify()

    operators = [term[0] for term in H.to_list()]

    qc = QuantumCircuit(num_qubits)
    depth = 2
    for i, t in enumerate(ParameterVector("t", len(operators) * depth)):
        qc.append(
            PauliEvolutionGate(Pauli(operators[i % len(operators)]), time=t),
            range(num_qubits),
        )
        qc.barrier()

    Hamiltonian_matrix = create_ising(
        num_qubits, 0.3, -0.9, nonint_const=-0.1
    ).to_matrix(sparse=True)

    def infidelity(
        updated_parameters: np.ndarray,
        initial_parameters: np.ndarray,
        time: float | None = None,
    ):
        state1 = Statevector(qc.assign_parameters(updated_parameters))
        state2 = Statevector(qc.assign_parameters(initial_parameters))
        if time is not None:
            state2 = expm_multiply(-1.0j * time * Hamiltonian_matrix, state2.data)
            state2 = Statevector(state2 / np.linalg.norm(state2))
        return 1 - state_fidelity(state1, state2)

    Ntimesteps = 5
    times = np.linspace(5e-1, 5, Ntimesteps)

    minima_values = np.zeros((1 + depth, Ntimesteps))
    minima_parameters = np.zeros((1 + depth, Ntimesteps, qc.num_parameters))

    minima_parameters[-1, 0, :] = 0.1
    for i in range(depth):
        if i == 0:
            continue
        minima_parameters[i, 0, :] = np.zeros(qc.num_parameters)
        minima_parameters[i, 0, i * len(operators) : (i + 1) * len(operators)] = (
            depth * minima_parameters[-1, 0, 0]
        )
        test = [
            infidelity(
                p * minima_parameters[-1, 0, :] + (1 - p) * minima_parameters[0, 0, :],
                minima_parameters[-1, 0, :],
            )
            for p in np.linspace(0, 1, 20)
        ]
    print(test)
    for index, t in tqdm(enumerate(times)):
        for d in range(depth + 1):
            result = minimize(
                infidelity,
                x0=minima_parameters[d, index - 1, :],
                args=(minima_parameters[-1, 0, :], t),
            )
            minima_parameters[d, index, :] = result.x
            minima_values[d, index] = result.fun

    np.save(
        "/home/marc/Documents/Fidelity/FidelityLandscape/results/data.npy",
        {
            "min_vals": minima_values,
            "Times": times,
            "depth": depth,
            "min_params": minima_parameters,
        },
        allow_pickle=True,
    )


def plot():
    data = np.load(
        "/home/marc/Documents/Fidelity/FidelityLandscape/results/data.npy",
        allow_pickle=True,
    ).item()
    print(data["min_params"])
    print(data["min_vals"])
    plt.plot(
        data["Times"],
        np.log10(data["min_vals"].T),
        label=[f"{d}step" for d in range(data["depth"])] + ["ALL"],
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    plot()
