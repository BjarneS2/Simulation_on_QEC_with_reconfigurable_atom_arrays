"""lattice.py
==============
Implements the :class:`Lattice` class for a square planar surface code of
distance ``d``. This simplified model places *data qubits* on a ``d x d``
grid and defines square plaquette stabilizers acting on 4 neighboring data
qubits. We alternate stabilizer types (X / Z) on a checkerboard pattern.

Simplifications
---------------
• Only data qubits are explicitly modeled (no separate ancilla measurement qubits).
• Stabilizer measurement outcomes are computed virtually from current Pauli
  error states without simulating circuits.
• Logical operators are simple strings across a boundary: logical X along
  the top row; logical Z along the left column (planar boundaries).
• Global phases of Pauli products are ignored - only commutation parity matters.

Conceptual Model
----------------
Each stabilizer S is product of 4 single-qubit Paulis of uniform type (all X or
all Z). A stabilizer outcome flips (i.e., becomes 1) iff an *odd* number of
data qubits beneath it carry single-qubit errors that anticommute with the
stabilizer's Pauli type:
    Z-stabilizer anticommutes with X or Y errors.
    X-stabilizer anticommutes with Z or Y errors.

Logical parity is similarly the parity of anticommutes along the chosen logical
string. Parity bit 0 => no logical flip detected; 1 => logical operator would
flip the encoded state.

Public API
----------
class Lattice:
    def __init__(d: int, p_x=0.0, p_z=0.0, keep_history=False, seed=None)
    def sample_physical_errors() -> None
    def measure_stabilizers(measurement_error: float = 0.0) -> dict[str, int]
    def logical_parity(ptype: str = 'Z') -> int
    def get_stabilizers() -> list[dict]
    def qubit_errors() -> list[str]

Edge Cases & Validation
-----------------------
• d >= 2 required (plaquettes exist). d=1 raises.
• measurement_error must be in [0,1].
• p_x, p_z validated via PhysicalQubit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random

from .physical_qubit import PhysicalQubit

_STAB_TYPES = ("X", "Z")


def _plaquette_qubits(d: int, r: int, c: int) -> List[int]:
    """Return the 4 data-qubit linear indices for top-left plaquette at (r,c)."""
    return [
        r * d + c,
        r * d + (c + 1),
        (r + 1) * d + c,
        (r + 1) * d + (c + 1),
    ]


@dataclass(slots=True)
class Lattice:
    """Square planar surface-code lattice of distance ``d``.

    Builds a grid of PhysicalQubit objects and defines square plaquette
    stabilizers. Checkerboard pattern: (r+c) % 2 == 0 -> Z-stabilizer else X.
    """

    d: int
    p_x: float = 0.0
    p_z: float = 0.0
    keep_history: bool = False
    seed: Optional[int] = None
    qubits: List[PhysicalQubit] = field(init=False, default_factory=list)
    stabilizers: List[Dict] = field(init=False, default_factory=list)
    _logical_X_indices: List[int] = field(init=False, default_factory=list)
    _logical_Z_indices: List[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.d < 2:
            raise ValueError("Lattice distance d must be >= 2 for plaquettes.")
        if self.seed is not None:
            random.seed(self.seed)
        self._build_qubits()
        self._build_stabilizers()
        self._define_logical_strings()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_qubits(self) -> None:
        self.qubits = [
            PhysicalQubit(p_x=self.p_x, p_z=self.p_z, label=f"({r},{c})", keep_history=self.keep_history)
            for r in range(self.d) for c in range(self.d)
        ]

    def _build_stabilizers(self) -> None:
        stabs: List[Dict] = []
        for r in range(self.d - 1):
            for c in range(self.d - 1):
                stype = "Z" if (r + c) % 2 == 0 else "X"
                qubits = _plaquette_qubits(self.d, r, c)
                stabs.append({
                    "id": f"P_{r}_{c}",
                    "type": stype,
                    "qubits": qubits,
                    "coord": (r, c),
                })
        self.stabilizers = stabs

    def _define_logical_strings(self) -> None:
        # Logical X along top row (acts with X on each qubit). Logical Z along left column.
        self._logical_X_indices = [0 * self.d + c for c in range(self.d)]
        self._logical_Z_indices = [r * self.d + 0 for r in range(self.d)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample_physical_errors(self, p_x: float | None = None, p_z: float | None = None) -> None:
        """Sample and apply independent single-qubit errors on all data qubits.

        Optional overrides p_x, p_z apply uniformly this round.
        """
        for q in self.qubits:
            q.sample_error(p_x=p_x, p_z=p_z)

    def measure_stabilizers(self, measurement_error: float = 0.0, rng: Optional[random.Random] = None) -> Dict[str, int]:
        """Return dict of stabilizer outcomes {stab_id: bit}.

        Bit semantics: 0 => even parity (commutes), 1 => odd parity (flip).
        Measurement error flips the bit with given probability independently.
        """
        if not (0.0 <= measurement_error <= 1.0):
            raise ValueError("measurement_error must be in [0,1].")
        rand = rng.random if rng is not None else random.random
        outcomes: Dict[str, int] = {}
        for stab in self.stabilizers:
            stype = stab["type"]
            indices = stab["qubits"]
            parity = 0
            for idx in indices:
                err = self.qubits[idx].current_error
                if stype == "Z":  # anticommute if X or Y present
                    if err in ("X", "Y"):
                        parity ^= 1
                else:  # X stabilizer anticommutes with Z or Y
                    if err in ("Z", "Y"):
                        parity ^= 1
            # Apply measurement error
            if measurement_error > 0 and rand() < measurement_error:
                parity ^= 1
            outcomes[stab["id"]] = parity
        return outcomes

    def logical_parity(self, ptype: str = "Z") -> int:
        """Compute logical operator parity.

        ptype='Z' uses logical Z string along left column - parity of X/Y errors.
        ptype='X' uses logical X string along top row - parity of Z/Y errors.
        Returns 0 (no flip) or 1 (flip).
        """
        if ptype not in ("Z", "X"):
            raise ValueError("ptype must be 'Z' or 'X'.")
        indices = self._logical_Z_indices if ptype == "Z" else self._logical_X_indices
        parity = 0
        for idx in indices:
            err = self.qubits[idx].current_error
            if ptype == "Z":
                if err in ("X", "Y"):
                    parity ^= 1
            else:
                if err in ("Z", "Y"):
                    parity ^= 1
        return parity

    def get_stabilizers(self) -> List[Dict]:
        """Return list of stabilizer definitions."""
        return list(self.stabilizers)

    def qubit_errors(self) -> List[str]:
        """Return list of current single-qubit error labels."""
        return [q.current_error for q in self.qubits]

    # ------------------------------------------------------------------
    # Convenience / representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Lattice(d={self.d}, qubits={len(self.qubits)}, stabilizers={len(self.stabilizers)})"

    def summary(self) -> str:
        lines = [repr(self), "Stabilizers:"]
        for s in self.stabilizers:
            lines.append(f"  {s['id']} {s['type']} {s['qubits']}")
        lines.append("Logical X indices: " + str(self._logical_X_indices))
        lines.append("Logical Z indices: " + str(self._logical_Z_indices))
        return "\n".join(lines)


__all__ = ["Lattice"]

