"""
logical_block.py
================
Provides a `LogicalBlock` class that represents an encoded logical qubit built
on top of a surface-code lattice.

Responsibilities
----------------
• Encapsulate one `Lattice` instance (the physical layer)
• Store current logical state label (|0_L⟩, |1_L⟩, |+_L⟩, etc.)
• Apply logical operations via transversal physical gates
• Provide interface to:
    - Reset / initialize logical states
    - Run stabilizer rounds (syndrome extraction)
    - Measure logical operators
    - Track logical error rate / fidelity metrics

Core API
--------
class LogicalBlock:
    lattice: Lattice
    label: str
    def initialize(self, state='0'): ...
    def apply_logical(self, gate_name: str): ...
    def apply_initialization_error(self, p_error: float): ...
    def measure_logical(self, basis='Z'): -> int
    def inject_noise(self): delegate to lattice.sample_physical_errors()
    def apply_correction(self, correction_ops: dict): ...
    def get_syndrome(self): -> dict
    def logical_error_rate(self): -> float
    def summary(self): -> str
    
Implementation Notes
--------------------
Logical operations delegate to physical-layer gates (in `gates/`).
Later you can subclass this for different codes (surface, color, Bacon-Shor, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import math

from .lattice import Lattice

_ONE_Q_LOGICAL_GATES = {"X", "Z", "H"}
_ROTATION_TOL = 1e-9


def _is_multiple_of_pi_over_2(theta: float) -> bool:
    k = theta / (math.pi / 2)
    return abs(k - round(k)) < _ROTATION_TOL


@dataclass(slots=True)
class LogicalBlock:
    """Encapsulates one surface-code logical qubit (planar square code).

    This abstraction wraps a :class:`Lattice` instance and provides:
    * Initialization to logical |0>, |+> states.
    * Application of (simplified) logical operators X, Z, H.
    * Support for rotation gate stubs (Rx, Rz) for multiples of π/2.
    * Syndrome extraction & correction hooks.
    * Logical measurements in X or Z basis.

    Limitations / Simplifications
    -----------------------------
        * Logical state label evolution tracked explicitly via transition tables
            (|0_L>, |1_L>, |+_L>, |-_L>). Global phases are discarded.
    * Rotations only support θ = n * π/2 and reduce to Pauli products.
    * No active decoder; `apply_correction` directly applies provided Pauli frame.
    * No Y stabilizers; code modeled with alternating X/Z plaquettes only.
    """

    d: int
    p_x: float = 0.0
    p_z: float = 0.0
    keep_history: bool = False
    seed: Optional[int] = None
    label: str = "|0_L>"
    lattice: Lattice = field(init=False)
    _last_syndrome: Dict[str, int] = field(init=False, default_factory=dict)
    _frame_swap: bool = field(init=False, default=False)  # still used to reinterpret parity axes after H

    def __post_init__(self) -> None:
        self.lattice = Lattice(d=self.d, p_x=self.p_x, p_z=self.p_z, keep_history=self.keep_history, seed=self.seed)
        self.initialize('0')  # default

    # ------------------------------------------------------------------
    # Initialization & State
    # ------------------------------------------------------------------
    def initialize(self, state: str = '0') -> None:
        """Initialize logical block to one of |0_L>, |1_L>, |+_L>, |-_L>.

        Resets all physical error states. Unsupported labels default to |0_L>.
        """
        # Reset physical layer
        for q in self.lattice.qubits:
            q.reset_error()
        self._last_syndrome.clear()
        norm_state = state.strip().lower()
        mapping = {
            '0': '|0_L>', 'zero': '|0_L>',
            '1': '|1_L>', 'one': '|1_L>',
            '+': '|+_L>', 'plus': '|+_L>',
            '-': '|-_L>', 'minus': '|-_L>',
        }
        self.label = mapping.get(norm_state, '|0_L>')
        self._frame_swap = False  # reset frame interpretation

    # ------------------------------------------------------------------
    # Logical Gate Application
    # ------------------------------------------------------------------
    def apply_logical(self, gate_name: str) -> None:
        """Apply a simplified logical gate: 'X', 'Z', or 'H'.

    X: Apply physical X along logical Z boundary; update label table.
    Z: Apply physical Z along logical X boundary; update label table.
    H: Toggle frame (X<->Z axis reinterpretation) & update label via table.
        """
        g = gate_name.upper()
        if g not in _ONE_Q_LOGICAL_GATES:
            raise ValueError(f"Unsupported logical gate '{gate_name}'. Supported: {_ONE_Q_LOGICAL_GATES}")
        if g == 'H':
            self._apply_logical_H()
            return
        if g == 'X':
            for i in self._logical_Z_indices_view():
                self.lattice.qubits[i].apply_pauli('X')
        elif g == 'Z':
            for i in self._logical_X_indices_view():
                self.lattice.qubits[i].apply_pauli('Z')
        self._apply_label_transition(g)

    # Rotation stubs (reduce to π/2 multiples)
    def apply_rotation(self, axis: str, theta: float) -> None:
        axis_u = axis.upper()
        if axis_u not in ('X', 'Z'):
            raise ValueError("Rotation axis must be 'X' or 'Z'.")
        if not _is_multiple_of_pi_over_2(theta):
            raise NotImplementedError("Only multiples of π/2 supported for now.")
        k = int(round(theta / (math.pi / 2))) % 4
        # Map to sequence of Pauli / frame ops
        if k == 0:
            return
        if axis_u == 'Z':
            if k == 1:  # π/2 about Z ~ phase (ignored) so treat as no-op for parity model
                return
            if k == 2:  # π -> Z
                self.apply_logical('Z')
            if k == 3:  # 3π/2 -> Z^3 = Z (phase ignored)
                self.apply_logical('Z')
        else:  # X axis
            if k == 1:
                return
            if k in (2, 3):
                self.apply_logical('X')

    def _apply_logical_H(self) -> None:
        # Toggle frame swap (affects how parity axes are interpreted)
        self._frame_swap = not self._frame_swap
        self._apply_label_transition('H')

    def _logical_X_indices_view(self) -> List[int]:
        # If frame swapped, X and Z roles reversed
        return self.lattice._logical_Z_indices if self._frame_swap else self.lattice._logical_X_indices

    def _logical_Z_indices_view(self) -> List[int]:
        return self.lattice._logical_X_indices if self._frame_swap else self.lattice._logical_Z_indices

    def _apply_label_transition(self, gate: str) -> None:
        """Update logical label deterministically for gate in {X,Z,H}."""
        transitions = {
            'X': {
                '|0_L>': '|1_L>',
                '|1_L>': '|0_L>',
                '|+_L>': '|+_L>',
                '|-_L>': '|-_L>',
            },
            'Z': {
                '|0_L>': '|0_L>',
                '|1_L>': '|1_L>',  # global phase ignored
                '|+_L>': '|-_L>',
                '|-_L>': '|+_L>',
            },
            'H': {
                '|0_L>': '|+_L>',
                '|1_L>': '|-_L>',
                '|+_L>': '|0_L>',
                '|-_L>': '|1_L>',
            },
        }
        self.label = transitions[gate][self.label]

    # Public accessor for logical label
    def logical_state(self) -> str:
        return self.label

    # ------------------------------------------------------------------
    # Noise & Syndrome
    # ------------------------------------------------------------------
    def inject_noise(self, p_x: float | None = None, p_z: float | None = None) -> None:
        self.lattice.sample_physical_errors(p_x=p_x, p_z=p_z)

    def run_syndrome_round(self, measurement_error: float = 0.0) -> Dict[str, int]:
        self._last_syndrome = self.lattice.measure_stabilizers(measurement_error=measurement_error)
        return dict(self._last_syndrome)

    def get_syndrome(self) -> Dict[str, int]:
        return dict(self._last_syndrome)

    def apply_correction(self, correction_ops: Dict[int, str]) -> None:
        """Apply a dictionary mapping physical qubit index -> Pauli label."""
        for idx, op in correction_ops.items():
            self.lattice.qubits[idx].apply_pauli(op)

    # ------------------------------------------------------------------
    # Measurement & Logical Parity
    # ------------------------------------------------------------------
    def measure_logical(self, basis: str = 'Z') -> int:
        b = basis.upper()
        if b not in ('Z', 'X'):
            raise ValueError("basis must be 'Z' or 'X'")
        parity = self.lattice.logical_parity(b)
        return parity

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def visualize_qubits(self) -> str:
        """Return ASCII grid of current physical error labels."""
        rows: List[str] = []
        for r in range(self.d):
            row = self.lattice.qubit_errors()[r * self.d:(r + 1) * self.d]
            rows.append(' '.join(row))
        return '\n'.join(rows)

    def summary(self) -> str:
        parity_z = self.lattice.logical_parity('Z')
        parity_x = self.lattice.logical_parity('X')
        syndrome_str = self._last_syndrome if self._last_syndrome else {}
        lines = [
            f"LogicalBlock(d={self.d}, state={self.label}, frame_swap={self._frame_swap})",
            f"Logical Z parity: {parity_z}  Logical X parity: {parity_x}  Syndrome: {syndrome_str}",
            "Physical error grid:",
            self.visualize_qubits(),
        ]
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"LogicalBlock(d={self.d}, label={self.label})"


__all__ = ["LogicalBlock"]

