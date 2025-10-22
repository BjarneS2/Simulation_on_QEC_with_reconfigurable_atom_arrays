"""Pauli operators with correct phase-aware multiplication.

Single-qubit Pauli multiplication rules (up to global phase):
    X * Y =  i Z
    Y * X = -i Z
    Y * Z =  i X
    Z * Y = -i X
    Z * X =  i Y
    X * Z = -i Y

All operators square to identity: P * P = I (phase +1).
Identity acts neutrally: I * P = P * I = P.

We expose a helper dataclass `PauliWithPhase` that carries an overall complex
phase in {+1, -1, +1j, -1j} alongside the resulting Pauli enum element. Chained
multiplication preserves and combines phases.

Examples:
    >>> from src.AtomLogic import Pauli
    >>> print(Pauli.X * Pauli.Y)  # iZ
    >>> prod = (Pauli.X * Pauli.Y) * Pauli.Z
    >>> print(prod)  # iI
    >>> (Pauli.X * Pauli.Z).phase  # -1j

Commutation:
Two single-qubit Paulis (excluding identity) anticommute iff they are distinct.
`Pauli.commutes_with(other)` returns True/False for the single-qubit case.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union

_PHASE_TYPE = complex

@dataclass(frozen=True)
class PauliWithPhase:
    """Container for a Pauli operator with an accumulated phase.

    Attributes:
        phase: complex in {1, -1, 1j, -1j}
        op:    Pauli enum element
    """
    phase: _PHASE_TYPE
    op: "Pauli"

    def __mul__(self, other: Union["Pauli", "PauliWithPhase"]) -> "PauliWithPhase":
        """Multiply by another Pauli or PauliWithPhase, combining phases."""
        if isinstance(other, Pauli):
            phase_delta, op_res = self.op._mul_no_wrap(other)
            return PauliWithPhase(self.phase * phase_delta, op_res)
        if isinstance(other, PauliWithPhase):
            # Multiply operators then combine phases
            phase_delta, op_res = self.op._mul_no_wrap(other.op)
            return PauliWithPhase(self.phase * other.phase * phase_delta, op_res)
        return NotImplemented

    def __rmul__(self, other: Union["Pauli", "PauliWithPhase"]) -> "PauliWithPhase":
        if isinstance(other, Pauli):
            phase_delta, op_res = other._mul_no_wrap(self.op)
            return PauliWithPhase(other._phase_identity() * self.phase * phase_delta, op_res)
        if isinstance(other, PauliWithPhase):
            phase_delta, op_res = other.op._mul_no_wrap(self.op)
            return PauliWithPhase(other.phase * self.phase * phase_delta, op_res)
        return NotImplemented

    @property
    def phase_str(self) -> str:
        if self.phase == 1:
            return ""
        if self.phase == -1:
            return "-"
        if self.phase == 1j:
            return "i"
        if self.phase == -1j:
            return "-i"
        return f"({self.phase})"

    def __str__(self) -> str:  # user-friendly
        return f"{self.phase_str}{self.op}"

    def __repr__(self) -> str:  # unambiguous
        return f"PauliWithPhase(phase={self.phase}, op={self.op})"


class Pauli(Enum):
    I = 0
    Z = 1
    X = 2
    Y = 3

    # --- Public API -----------------------------------------------------
    def __mul__(self, other: Union["Pauli", PauliWithPhase]) -> PauliWithPhase:
        """Phase-aware Pauli multiplication.

        Returns a PauliWithPhase capturing the correct complex phase.
        """
        if isinstance(other, Pauli):
            phase, op = self._mul_no_wrap(other)
            return PauliWithPhase(phase, op)
        if isinstance(other, PauliWithPhase):
            phase_delta, op_res = self._mul_no_wrap(other.op)
            return PauliWithPhase(other.phase * phase_delta, op_res)
        return NotImplemented

    def __rmul__(self, other: Union["Pauli", PauliWithPhase]):  # symmetry for right-op
        if isinstance(other, Pauli):
            phase, op = other._mul_no_wrap(self)
            return PauliWithPhase(phase, op)
        if isinstance(other, PauliWithPhase):
            phase_delta, op_res = other.op._mul_no_wrap(self)
            return PauliWithPhase(other.phase * phase_delta, op_res)
        return NotImplemented

    def commutes_with(self, other: "Pauli") -> bool:
        """Return True if single-qubit Paulis commute.

        Identity commutes with everything; distinct non-identity Paulis anticommute.
        """
        if self == Pauli.I or other == Pauli.I:
            return True
        return self == other

    # --- Internal helpers -----------------------------------------------
    @staticmethod
    def _phase_identity() -> _PHASE_TYPE:
        return 1

    def _mul_no_wrap(self, other: "Pauli") -> tuple[_PHASE_TYPE, "Pauli"]:
        """Internal raw multiplication returning (phase, Pauli)."""
        # Fast path: identical or identity => identity phase and result
        if self == other:
            return 1, Pauli.I
        if self == Pauli.I:
            return 1, other
        if other == Pauli.I:
            return 1, self

        # Explicit table for the six distinct ordered pairs that produce ±i * third.
        table: dict[tuple[Pauli, Pauli], tuple[_PHASE_TYPE, Pauli]] = {
            (Pauli.X, Pauli.Y): (1j, Pauli.Z),
            (Pauli.Y, Pauli.X): (-1j, Pauli.Z),
            (Pauli.Y, Pauli.Z): (1j, Pauli.X),
            (Pauli.Z, Pauli.Y): (-1j, Pauli.X),
            (Pauli.Z, Pauli.X): (1j, Pauli.Y),
            (Pauli.X, Pauli.Z): (-1j, Pauli.Y),
        }
        try:
            return table[(self, other)]
        except KeyError:
            raise ValueError(f"Unhandled Pauli multiplication: {self} * {other}")

    # --- Representation --------------------------------------------------
    def __repr__(self) -> str:
        return self._to_str()

    def __str__(self) -> str:
        return self._to_str()

    def _to_str(self) -> str:
        match self:
            case Pauli.I:
                return "I"
            case Pauli.Z:
                return "Z"
            case Pauli.X:
                return "X"
            case Pauli.Y:
                return "Y"
        return "?"  # should never happen

    @property
    def color(self) -> str:
        """Define color per Pauli operator for plotting."""
        match self:
            case Pauli.I:
                return "#FFFFFF"
            case Pauli.Z:
                return "#D15567"
            case Pauli.X:
                return "#E17C88"
            case Pauli.Y:
                return "#F28EBF"
    