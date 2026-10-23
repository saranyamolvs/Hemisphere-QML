"""Exact, noiseless statevector simulation of a specified quantum feature map.

This module does not use a quantum processor.  It produces the same pure-state
fidelity kernel as an ideal overlap experiment for the explicitly defined map:
in each repetition apply H to every qubit, RZ(2*x_j) to qubit j, and
RZZ(2*(pi-x_j)*(pi-x_k)) to the requested pairs.  Gates act from left to right
in that order.  Computational-basis indices use the little-endian convention
(qubit 0 is the least significant bit), matching Qiskit.

Angles should be obtained with a scaler fitted on training data alone.  The
NumPy implementation has no shot noise, gate noise, or quantum-hardware speedup.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_map(n_features: int, reps: int, entanglement: str, max_qubits: int) -> None:
    if isinstance(n_features, (bool, np.bool_)) or not isinstance(n_features, (int, np.integer)) or n_features < 1:
        raise ValueError("The feature map needs at least one feature/qubit.")
    if isinstance(max_qubits, (bool, np.bool_)) or not isinstance(max_qubits, (int, np.integer)) or max_qubits < 1:
        raise ValueError("max_qubits must be a positive integer.")
    if n_features > max_qubits:
        raise ValueError(f"{n_features} features exceed the {max_qubits}-qubit simulation limit.")
    if isinstance(reps, (bool, np.bool_)) or not isinstance(reps, (int, np.integer)) or reps < 1:
        raise ValueError("reps must be a positive integer.")
    if entanglement not in {"linear", "full"}:
        raise ValueError("entanglement must be 'linear' or 'full'.")


def _pairs(n_features: int, entanglement: str) -> list[tuple[int, int]]:
    if entanglement == "linear":
        return [(j, j + 1) for j in range(n_features - 1)]
    return [(j, k) for j in range(n_features) for k in range(j + 1, n_features)]


def _rotation_angles(X: np.ndarray, edges: list[tuple[int, int]]):
    """Reject finite inputs whose RZ/RZZ products overflow machine precision."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            rz = 2.0 * X
            rzz = [2.0 * ((np.pi - X[:, j]) * (np.pi - X[:, k])) for j, k in edges]
    except FloatingPointError as exc:
        raise ValueError("Feature angles generate non-finite RZ/RZZ gate rotations; use bounded angle features.") from exc
    return rz, rzz


def feature_states(
    X: np.ndarray,
    reps: int = 2,
    entanglement: str = "linear",
    max_qubits: int = 12,
) -> np.ndarray:
    """Return one exact statevector per row of an (n_samples, n_qubits) array.

    Angles are radians and must be finite.  Inputs are not silently rescaled or
    clipped here.  Output storage is complex128, O(n_samples*2**n_qubits).
    A conservative 2 GiB working-memory guard includes gate temporaries rather
    than only the returned state array.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional sample-by-feature array.")
    n_samples, n_features = X.shape
    _validate_map(n_features, reps, entanglement, max_qubits)
    if not np.isfinite(X).all():
        raise ValueError("Feature angles must all be finite.")
    dimension = 1 << n_features
    estimated_working_bytes = n_samples * dimension * 64 + dimension * (n_features + 1) * 8
    if estimated_working_bytes > 2**31:
        raise MemoryError("Statevector working memory exceeds 2 GiB; reduce the sample count or qubit count.")
    edges = _pairs(n_features, entanglement)
    rz_angles, rzz_angles = _rotation_angles(X, edges)
    states = np.zeros((n_samples, dimension), dtype=np.complex128)
    states[:, 0] = 1.0
    indices = np.arange(dimension, dtype=np.uint64)
    signs = [1.0 - 2.0 * ((indices >> j) & 1).astype(np.float64) for j in range(n_features)]
    # All RZ and RZZ gates in a repetition commute, so combine their phases.
    phase = np.zeros((n_samples, dimension), dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for j in range(n_features):
                phase += (rz_angles[:, j, None] / 2.0) * signs[j][None, :]
            for (j, k), angle in zip(edges, rzz_angles):
                phase += (angle[:, None] / 2.0) * (signs[j] * signs[k])[None, :]
            diagonal = np.exp(-1j * phase)
    except FloatingPointError as exc:
        raise ValueError("Feature-map phases exceed finite floating-point precision; use bounded angle features.") from exc
    del phase
    for _ in range(reps):
        for j in range(n_features):
            # The central axis chooses the qubit's 0/1 sectors.
            paired = states.reshape(n_samples, dimension // (2 << j), 2, 1 << j)
            zero = paired[:, :, 0, :].copy()
            one = paired[:, :, 1, :].copy()
            paired[:, :, 0, :] = (zero + one) / np.sqrt(2.0)
            paired[:, :, 1, :] = (zero - one) / np.sqrt(2.0)
        states *= diagonal
    return states


def _validate_states(states: np.ndarray, name: str) -> np.ndarray:
    states = np.asarray(states, dtype=np.complex128)
    if states.ndim != 2 or states.shape[1] < 2:
        raise ValueError(f"{name} must contain rows of nontrivial statevectors.")
    dim = states.shape[1]
    if dim & (dim - 1):
        raise ValueError(f"{name} statevector dimension must be a power of two.")
    if not np.isfinite(states).all():
        raise ValueError(f"{name} contains non-finite amplitudes.")
    norms = np.sum(np.abs(states) ** 2, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-9, atol=1e-10):
        raise ValueError(f"{name} statevectors must be normalized.")
    return states


def fidelity_kernel(
    states_a: np.ndarray,
    states_b: np.ndarray | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute K_ij = |<psi(a_i)|psi(b_j)>|**2 in row batches.

    Training use is ``fidelity_kernel(train_states)``.  Prediction needs a
    test-by-training matrix: ``fidelity_kernel(test_states, train_states)``.
    The returned dense float64 matrix still costs O(n_a*n_b) storage.
    """
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    a = _validate_states(states_a, "states_a")
    b = a if states_b is None else _validate_states(states_b, "states_b")
    if a.shape[1] != b.shape[1]:
        raise ValueError("Both state sets must have the same number of qubits.")
    if a.shape[0] * b.shape[0] * 8 > 2**31:
        raise MemoryError("Dense kernel exceeds 2 GiB; select a smaller classification subset.")
    result = np.empty((a.shape[0], b.shape[0]), dtype=np.float64)
    b_conjugate_transpose = b.conj().T
    for start in range(0, len(a), batch_size):
        overlap = a[start : start + batch_size] @ b_conjugate_transpose
        result[start : start + batch_size] = np.abs(overlap) ** 2
    # Bound round-off only; physical fidelities lie in [0, 1].
    np.clip(result, 0.0, 1.0, out=result)
    return result


def kernel_diagnostics(K: np.ndarray) -> dict[str, Any]:
    """Describe a square training Gram matrix, preserving raw minimum eigenvalue.

    ``effective_rank`` is exp(entropy of normalized nonnegative eigenvalues).
    ``participation_ratio`` is (sum lambda)**2/sum(lambda**2).  These are
    diagnostics, not evidence of generalization or computational advantage.
    Eigendecomposition costs O(n_samples**3); use a bounded training subset.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] == 0:
        raise ValueError("K must be a nonempty square matrix.")
    if not np.isfinite(K).all():
        raise ValueError("K must be finite.")
    eigenvalues = np.linalg.eigvalsh((K + K.T) / 2.0)
    positive = np.maximum(eigenvalues, 0.0)
    trace = float(positive.sum())
    probabilities = positive[positive > 0] / trace if trace else np.empty(0)
    off_diagonal = K[~np.eye(len(K), dtype=bool)]
    diagonal = np.diag(K)
    tol = max(float(positive.max()), 1.0) * len(K) * np.finfo(np.float64).eps
    return {
        "n_samples": int(len(K)),
        "symmetry_max_abs_error": float(np.max(np.abs(K - K.T))),
        "diagonal_min": float(diagonal.min()),
        "diagonal_max": float(diagonal.max()),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "numerical_rank": int(np.sum(eigenvalues > tol)),
        "effective_rank": float(np.exp(-np.sum(probabilities * np.log(probabilities)))) if trace else 0.0,
        "participation_ratio": float(trace**2 / np.sum(positive**2)) if trace else 0.0,
        "off_diagonal_mean": float(off_diagonal.mean()) if len(off_diagonal) else None,
        "off_diagonal_std": float(off_diagonal.std()) if len(off_diagonal) else None,
        "off_diagonal_min": float(off_diagonal.min()) if len(off_diagonal) else None,
        "off_diagonal_max": float(off_diagonal.max()) if len(off_diagonal) else None,
        "negative_eigenvalue_tolerance": float(tol),
        "is_psd_within_tolerance": bool(eigenvalues[0] >= -tol),
    }


def feature_map_metadata(n_features: int, reps: int = 2, entanglement: str = "linear") -> dict[str, Any]:
    """Return serializable map and ideal gate-count details for provenance."""
    _validate_map(n_features, reps, entanglement, max_qubits=12)
    edges = _pairs(n_features, entanglement)
    return {
        "name": "explicit_H_RZ_RZZ_feature_map",
        "backend": "numpy_exact_noiseless_statevector_simulator",
        "hardware_execution": False,
        "shots": None,
        "qubits": int(n_features),
        "statevector_dimension": 1 << n_features,
        "repetitions": int(reps),
        "entanglement": entanglement,
        "pairs": [list(pair) for pair in edges],
        "gate_order_per_repetition": ["H_all", "RZ(2*x_j)", "RZZ(2*(pi-x_j)*(pi-x_k))"],
        "basis_order": "little_endian_qubit_0_least_significant_bit",
        "kernel": "abs(vdot(psi_x, psi_y))**2",
        "single_encoding_H_count": int(reps * n_features),
        "single_encoding_RZ_count": int(reps * n_features),
        "single_encoding_RZZ_count": int(reps * len(edges)),
        "overlap_circuit": "prepare U(x), then U(y).inverse(); probability(all_zeros) equals fidelity",
        "limitation": "Ideal classical simulation; no quantum hardware, shot noise, or speedup demonstrated.",
    }


def make_qiskit_circuit(
    angles: np.ndarray, reps: int = 2, entanglement: str = "linear"
) -> Any:
    """Build an optional Qiskit circuit with exactly the simulated gate order.

    Qiskit is optional and is imported only by this function.  To construct an
    overlap circuit append ``make_qiskit_circuit(y).inverse()`` to the circuit
    for x and measure every qubit.  Finite-shot execution and PSD repair of
    noisy kernels require a separate, explicitly reported hardware workflow.
    """
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim != 1 or not np.isfinite(angles).all():
        raise ValueError("angles must be one finite feature vector.")
    _validate_map(len(angles), reps, entanglement, max_qubits=12)
    edges = _pairs(len(angles), entanglement)
    rz_angles, rzz_angles = _rotation_angles(angles[None, :], edges)
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise ImportError("Circuit export requires optional dependency 'qiskit'. The NumPy kernel does not.") from exc
    circuit = QuantumCircuit(len(angles), name="schlieren_feature_map")
    for _ in range(reps):
        for j in range(len(angles)):
            circuit.h(j)
        for j, angle in enumerate(rz_angles[0]):
            circuit.rz(float(angle), j)
        for (j, k), angle in zip(edges, rzz_angles):
            circuit.rzz(float(angle[0]), j, k)
    return circuit
