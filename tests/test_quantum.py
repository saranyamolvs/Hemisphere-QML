"""Numerical checks of the actual quantum map, not experimental evidence."""

import importlib.util
import unittest

import numpy as np

from schlieren_qml.quantum import (
    feature_map_metadata,
    feature_states,
    fidelity_kernel,
    kernel_diagnostics,
    make_qiskit_circuit,
)


def dense_reference(x, reps, full=False):
    """Independent dense-gate implementation in little-endian basis order."""
    x = np.asarray(x)
    d = len(x)
    state = np.zeros(2**d, dtype=complex)
    state[0] = 1.0
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    edges = [(j, k) for j in range(d) for k in range(j + 1, d) if full or k == j + 1]
    for _ in range(reps):
        for j in range(d):
            op = np.array([[1.0]])
            for q in reversed(range(d)):
                op = np.kron(op, H if q == j else np.eye(2))
            state = op @ state
        for j in range(d):
            rz = np.diag([np.exp(-1j*x[j]), np.exp(1j*x[j])])
            op = np.array([[1.0]])
            for q in reversed(range(d)):
                op = np.kron(op, rz if q == j else np.eye(2))
            state = op @ state
        for j, k in edges:
            phi = (np.pi - x[j]) * (np.pi - x[k])
            op = np.eye(2**d, dtype=complex)
            for index in range(2**d):
                sign = (-1)**(((index >> j) & 1) + ((index >> k) & 1))
                op[index,index] = np.exp(-1j*phi*sign)
            state = op @ state
    return state


class QuantumTests(unittest.TestCase):
    def test_exact_map_against_independent_dense_gates(self):
        for d in [1, 2, 3]:
            for reps in [1, 2, 3]:
                for entanglement in ["linear", "full"]:
                    x = np.linspace(0.17, 2.43, d)
                    result = feature_states(x[None,:], reps=reps, entanglement=entanglement)[0]
                    expected = dense_reference(x, reps, full=entanglement == "full")
                    np.testing.assert_allclose(result, expected, atol=1e-13)

    def test_kernel_has_fidelity_properties(self):
        rng = np.random.default_rng(841)
        X = rng.uniform(0, np.pi, (12, 3))
        states = feature_states(X)
        np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-13)
        K = fidelity_kernel(states, batch_size=3)
        np.testing.assert_allclose(K, K.T, atol=1e-13)
        np.testing.assert_allclose(np.diag(K), 1, atol=1e-13)
        self.assertGreaterEqual(np.linalg.eigvalsh(K).min(), -1e-12)
        self.assertTrue(np.any(K < 0.8))
        self.assertTrue(kernel_diagnostics(K)["is_psd_within_tolerance"])

    def test_cross_kernel_order_and_complex_conjugation(self):
        a = np.array([[1, 1j], [1, -1j]], dtype=complex) / np.sqrt(2)
        b = np.array([[1, 1j]], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(fidelity_kernel(a, b), [[1], [0]], atol=1e-13)

    def test_single_qubit_one_layer_analytic_kernel(self):
        x, y = 0.19, 1.43
        states = feature_states(np.array([[x], [y]]), reps=1)
        self.assertAlmostEqual(fidelity_kernel(states)[0,1], np.cos(x-y)**2, places=12)

    def test_invalid_inputs_are_rejected(self):
        for X in [np.ones(4), np.ones((2,0)), np.array([[np.nan]])]:
            with self.assertRaises(ValueError):
                feature_states(X)
        with self.assertRaises(ValueError):
            feature_states(np.ones((1, 13)))
        with self.assertRaises(ValueError):
            feature_states(np.ones((2,2)), reps=0)
        with self.assertRaises(ValueError):
            feature_states(np.ones((2,2)), entanglement="nonsense")
        with self.assertRaises(ValueError):
            fidelity_kernel(np.ones((2, 4)))
        with self.assertRaises(ValueError):
            kernel_diagnostics(np.ones((2,3)))

    def test_metadata_is_explicit_about_simulation(self):
        metadata = feature_map_metadata(4)
        self.assertFalse(metadata["hardware_execution"])
        self.assertEqual(metadata["single_encoding_RZZ_count"], 6)

    def test_finite_inputs_cannot_silently_generate_nonfinite_states(self):
        for angles in ([1e200, 1e200], [1e308]):
            with self.assertRaisesRegex(ValueError, "gate rotations"):
                feature_states(np.array([angles]))
            # Validation occurs before the optional Qiskit dependency import.
            with self.assertRaisesRegex(ValueError, "gate rotations"):
                make_qiskit_circuit(np.array(angles))

    def test_simulator_memory_guard_includes_gate_temporaries(self):
        # The output alone would be about 0.5 GiB, but simulation work would
        # exceed 2 GiB. It must reject before allocating those statevectors.
        with self.assertRaisesRegex(MemoryError, "working memory"):
            feature_states(np.zeros((8193, 12)))

    @unittest.skipUnless(importlib.util.find_spec("qiskit"), "optional Qiskit is not installed")
    def test_qiskit_export_matches_numpy(self):
        from qiskit.quantum_info import Statevector
        for entanglement in ["linear", "full"]:
            x = np.array([0.32, 1.2, 2.8])
            exported = Statevector.from_instruction(make_qiskit_circuit(x, entanglement=entanglement)).data
            actual = feature_states(x[None,:], entanglement=entanglement)[0]
            np.testing.assert_allclose(exported, actual, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
