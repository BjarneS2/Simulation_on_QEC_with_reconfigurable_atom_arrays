"""
transversal.py
==============
Implements logical-level transversal gates by coordinating many physical gates
between two or more lattices.

Responsibilities
----------------
• Implement transversal CNOT, CZ, and Hadamard between logical blocks
• Pair physical qubits across lattices (by index or mapping)
• Apply underlying physical gate to each pair
• Propagate Pauli errors and optionally inject gate infidelities

Core API
--------
def transversal_cnot(control_lattice, target_lattice, gate_error_prob=0.0)
def transversal_hadamard(logical_block)

Implementation Notes
--------------------
Start with equal-sized lattices (same d). Later generalize to different sizes.
Support correlated gate errors to simulate real-device behavior.
"""
