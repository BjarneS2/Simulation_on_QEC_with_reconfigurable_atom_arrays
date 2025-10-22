"""
error_models.py
===============
Defines helper functions and classes for sampling noise on qubits and gates.

Responsibilities
----------------
• Implement independent and correlated noise channels:
    - Bit-flip, phase-flip, depolarizing, and measurement noise
    - Gate-level infidelity (two-qubit Pauli faults)
• Provide utilities for Monte Carlo noise injection.

Core API
--------
def sample_pauli_channel(p_x, p_y, p_z) -> str
    Return a random Pauli error ('I','X','Y','Z').

def apply_gate_noise(q1, q2, p_gate)
    Apply stochastic two-qubit Pauli errors to a pair of qubits.

class NoiseModel:
    Stores probabilities and applies them to lattices or gates.

Used By
-------
• `core.lattice` (for physical noise)
• `gates/transversal.py` (for gate infidelity)
• `experiments/` scripts (for configurable noise sweeps)
"""
