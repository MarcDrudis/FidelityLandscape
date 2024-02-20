from typing import Callable

import numpy as np
import plotly.express as px
from qiskit.algorithms import SciPyRealEvolver, TimeEvolutionProblem
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from scipy.optimize import approx_fprime


def find_local_minima(
    fun: Callable,
    x0: np.ndarray,
    *args,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Find the closest local minima using slow gradient descent.

    Parameters:
    - fun: The target function.
    - x0: Initial point.
    - *args: Extra arguments for the target function.
    - learning_rate: Step size for gradient descent.
    - max_iterations: Maximum number of iterations.
    - epsilon: Convergence threshold.

    Returns:
    - Local minima as a numpy array.
    """
    x = x0.copy()

    for _ in range(max_iterations):
        gradient = approx_fprime(x, fun, 1.49e-8, *args)
        x -= learning_rate * gradient
        if np.linalg.norm(gradient) < epsilon:
            break

    return x


def create_heisenberg(
    num_qubits: int, j_const: float, g_const: float, circular: bool = False
) -> SparsePauliOp:
    """Creates an Heisenberg Hamiltonian on a lattice."""
    xx_op = ["I" * i + "XX" + "I" * (num_qubits - i - 2) for i in range(num_qubits - 1)]
    yy_op = ["I" * i + "YY" + "I" * (num_qubits - i - 2) for i in range(num_qubits - 1)]
    zz_op = ["I" * i + "ZZ" + "I" * (num_qubits - i - 2) for i in range(num_qubits - 1)]

    circ_op = (
        ["X" + "I" * (num_qubits - 2) + "X"]
        + ["Y" + "I" * (num_qubits - 2) + "Y"]
        + ["Z" + "I" * (num_qubits - 2) + "Z"]
        if circular
        else []
    )

    z_op = ["I" * i + "Z" + "I" * (num_qubits - i - 1) for i in range(num_qubits)]

    return (
        SparsePauliOp(xx_op + yy_op + zz_op + circ_op) * j_const
        + SparsePauliOp(z_op) * g_const
    )


def create_ising(
    num_qubits: int,
    j_const: float,
    g_const: float,
    circular: bool = False,
    nonint_const: float = 0,
) -> SparsePauliOp:
    """Creates an Heisenberg Hamiltonian on a lattice."""
    zz_op = ["I" * i + "ZZ" + "I" * (num_qubits - i - 2) for i in range(num_qubits - 1)]

    circ_op = +["Z" + "I" * (num_qubits - 2) + "Z"] if circular else []

    z_op = ["I" * i + "Z" + "I" * (num_qubits - i - 1) for i in range(num_qubits)]
    x_op = ["I" * i + "X" + "I" * (num_qubits - i - 1) for i in range(num_qubits)]

    return (
        SparsePauliOp(zz_op + circ_op) * j_const
        + SparsePauliOp(z_op) * g_const
        + SparsePauliOp(x_op) * nonint_const
    )


def fidelity_var_bound(deltatheta, dt, m, eigenrange):
    return (0.5 * (1 + np.sinc(2 * deltatheta))) ** m * (
        1 - dt**2 / 4 * (eigenrange) ** 2
    )


def evolve_circuit(H, dt, qc, initial_parameters, observables, num_timesteps=100):
    prob = TimeEvolutionProblem(
        hamiltonian=H,
        time=dt,
        initial_state=qc.assign_parameters(initial_parameters),
        aux_operators=observables,
    )
    solver = SciPyRealEvolver(num_timesteps=num_timesteps)
    return solver.evolve(prob)