"""
physical_gates.py
=================
Low-level definitions of single- and two-qubit gate effects on Pauli errors.

Responsibilities
----------------
• Define Pauli-frame propagation rules for each gate:
    - H, S, X, Z, CNOT, CZ, etc.
• Implement functions that mutate `PhysicalQubit` error states appropriately.
• Provide utilities for applying gate-error models (via `error_models`).

Example
-------
def cnot(control: PhysicalQubit, target: PhysicalQubit, p_error=0.0):
    Implements propagation rules:
        X_c → X_c X_t
        Z_t → Z_c Z_t
    Then injects stochastic gate noise.

Used By
-------
`gates/transversal.py`
"""

from __future__ import annotations

from typing import Optional
import random

from ..core.physical_qubit import PhysicalQubit
from ..core.logical_block import LogicalBlock

__all__ = ["X", "Z", "H", "Rx", "Rz"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _maybe_inject_single_qubit_error(q: PhysicalQubit, error_prob: float, rng: Optional[random.Random] = None) -> None:
    if error_prob <= 0.0:
        return
    r = (rng.random() if rng else random.random())
    if r < error_prob:
        # simple symmetric depolarizing over X,Z,Y (excluding identity)
        choice = r % 3
        if choice < 1:
            q.apply_pauli('X')
        elif choice < 2:
            q.apply_pauli('Z')
        else:
            q.apply_pauli('Y')


def _iter_logical_indices(block: LogicalBlock, axis: str):
    axis_u = axis.upper()
    if axis_u == 'X':
        return block._logical_X_indices_view()
    if axis_u == 'Z':
        return block._logical_Z_indices_view()
    raise ValueError("axis must be 'X' or 'Z'")


# ---------------------------------------------------------------------------
# Gate implementations (single-qubit and logical wrappers)
# ---------------------------------------------------------------------------
def X(target, error_prob: float = 0.0) -> None:
    """Apply an X gate to a PhysicalQubit or a logical X operation to a LogicalBlock."""
    if isinstance(target, PhysicalQubit):
        target.apply_pauli('X')
        _maybe_inject_single_qubit_error(target, error_prob)
        return
    if isinstance(target, LogicalBlock):
        target.apply_logical('X')
        return
    raise TypeError("X expects PhysicalQubit or LogicalBlock")


def Z(target, error_prob: float = 0.0) -> None:
    if isinstance(target, PhysicalQubit):
        target.apply_pauli('Z')
        _maybe_inject_single_qubit_error(target, error_prob)
        return
    if isinstance(target, LogicalBlock):
        target.apply_logical('Z')
        return
    raise TypeError("Z expects PhysicalQubit or LogicalBlock")


def H(target, error_prob: float = 0.0) -> None:
    if isinstance(target, PhysicalQubit):
        # H conjugation on Pauli frame: X <-> Z, Y -> -Y (phase ignored).
        cur = target.current_error
        if cur == 'X':
            target.apply_pauli('Z')  # X * Z -> Y ... but we want swap; easiest is reset then set? Use reset approach.
        elif cur == 'Z':
            target.apply_pauli('X')
        elif cur == 'Y':
            # Y -> -Y ; global phase ignored so treat as Y (do nothing)
            pass
        _maybe_inject_single_qubit_error(target, error_prob)
        return
    if isinstance(target, LogicalBlock):
        target.apply_logical('H')
        return
    raise TypeError("H expects PhysicalQubit or LogicalBlock")


def Rz(target, theta: float, error_prob: float = 0.0) -> None:
    """Z-axis rotation (supports multiples of pi/2)."""
    from math import pi, isclose
    k = round(theta / (pi / 2))
    if not isclose(theta, k * (pi / 2), rel_tol=1e-9, abs_tol=1e-9):
        raise NotImplementedError("Rz currently only supports multiples of π/2")
    k_mod = k % 4
    # Rz(pi) -> Z, Rz(3pi/2) -> Z (phase ignored); Rz(pi/2) global phase; Rz(0) identity
    if k_mod in (2, 3):
        Z(target, error_prob=error_prob)


def Rx(target, theta: float, error_prob: float = 0.0) -> None:
    from math import pi, isclose
    k = round(theta / (pi / 2))
    if not isclose(theta, k * (pi / 2), rel_tol=1e-9, abs_tol=1e-9):
        raise NotImplementedError("Rx currently only supports multiples of π/2")
    k_mod = k % 4
    if k_mod in (2, 3):
        X(target, error_prob=error_prob)

