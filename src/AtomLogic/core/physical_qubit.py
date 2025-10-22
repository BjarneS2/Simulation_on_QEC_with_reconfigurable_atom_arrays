"""physical_qubit.py
=================
Implements the :class:`PhysicalQubit` primitive used by the lattice layer.

A physical qubit tracks an accumulated single-qubit Pauli error drawn from
``{I, X, Y, Z}`` together with per-qubit independent bit-/phase-flip
probabilities ``p_x`` and ``p_z``. These probabilities are used to *sample*
stochastic errors (Monte Carlo) but the qubit itself only stores the *current
effective Pauli error* (up to global phase) for propagation through stabilizer
measurements and transversal gates.

Design Principles
-----------------
• Represent Pauli errors internally using two parity bits (x_bit, z_bit)
  so composition is a constant-time XOR.
• Provide an ergonomic string / enum view for other layers.
• Keep this component dependency-light; higher layers decide when to sample.
• Ignore global phases (±1, ±i) because they do not affect stabilizer outcomes.

Public API (stable)
-------------------
class PhysicalQubit(
    p_x: float = 0.0,
    p_z: float = 0.0,
    label: str | None = None,
    keep_history: bool = False
):
    .apply_pauli(pauli: str | 'Pauli') -> str
        Compose a new Pauli onto the stored error.
    .sample_error(p_x: float | None = None, p_z: float | None = None) -> str
        Stochastically apply an error using (possibly overridden) probabilities.
    .reset_error() -> None
        Clear accumulated error to identity.
    .current_error -> str
        Property returning 'I','X','Y','Z'.
    .has_error() -> bool
        True iff current_error != 'I'.
    .history -> list[str]
        Optional chronological error states (only if keep_history=True).

Extension Hooks (future)
------------------------
• Leakage flag / T1/T2 tracking
• Measurement error channel
• Gate-specific error accumulation

NOTE: We import ``Pauli`` enum from ``src.AtomLogic.pauli`` for convenience.
The qubit however stores *only* the effective Pauli symbol; phaseful products
are handled at the gate / Pauli layer when needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import random

try:  # Local import; allow soft failure if Pauli not yet implemented.
    from src.AtomLogic.pauli import Pauli  # type: ignore
except Exception:  # pragma: no cover - Pauli may not exist during early dev
    Pauli = None  # fallback; string interface still works

_PAULI_STRS = ("I", "X", "Y", "Z")


def _encode_pauli(label: str) -> Tuple[int, int]:
    """Encode Pauli label into parity bits (x_bit, z_bit).

    Mapping (ignoring global phase):
        I -> (0,0)
        X -> (1,0)
        Z -> (0,1)
        Y -> (1,1)
    """
    match label.upper():
        case "I":
            return 0, 0
        case "X":
            return 1, 0
        case "Z":
            return 0, 1
        case "Y":
            return 1, 1
        case _:
            raise ValueError(f"Invalid Pauli label '{label}'. Expected one of {_PAULI_STRS}.")


def _decode_pauli(x_bit: int, z_bit: int) -> str:
    """Inverse of :func:`_encode_pauli`."""
    if (x_bit, z_bit) == (0, 0):
        return "I"
    if (x_bit, z_bit) == (1, 0):
        return "X"
    if (x_bit, z_bit) == (0, 1):
        return "Z"
    if (x_bit, z_bit) == (1, 1):
        return "Y"
    raise ValueError(f"Invalid bits {(x_bit, z_bit)} for Pauli decode.")


@dataclass(slots=True)
class PhysicalQubit:
    """A single surface-code data qubit tracking its Pauli error state.

    Parameters
    ----------
    p_x : float
        Independent probability of a bit-flip (X) error per sampling event.
    p_z : float
        Independent probability of a phase-flip (Z) error per sampling event.
    label : str | None
        Optional human-readable identifier (e.g. lattice coordinate).
    keep_history : bool
        If True, record each new error state after application / sampling.

    Internal Representation
    -----------------------
    We store current error using two parity bits (x_bit, z_bit). Composition of
    a new Pauli simply XORs these bits. Global phases are intentionally dropped.
    """

    p_x: float = 0.0
    p_z: float = 0.0
    label: Optional[str] = None
    keep_history: bool = False
    _x_bit: int = field(init=False, repr=False, default=0)
    _z_bit: int = field(init=False, repr=False, default=0)
    _history: List[str] = field(init=False, repr=False, default_factory=list)

    # ------------------------------------------------------------------
    # Core state helpers
    # ------------------------------------------------------------------
    @property
    def current_error(self) -> str:
        """Return current accumulated Pauli error label."""
        return _decode_pauli(self._x_bit, self._z_bit)

    def has_error(self) -> bool:
        """True iff current error isn't identity."""
        return (self._x_bit | self._z_bit) != 0

    @property
    def history(self) -> List[str]:  # read-only external view
        return list(self._history)

    # ------------------------------------------------------------------
    # Mutation operations
    # ------------------------------------------------------------------
    def reset_error(self) -> None:
        """Reset to the identity (clean) state."""
        self._x_bit = 0
        self._z_bit = 0
        if self.keep_history:
            self._history.append(self.current_error)

    def apply_pauli(self, pauli: Union[str, object]) -> str:
        """Compose a Pauli error onto this qubit.

        The input may be a string label or a ``Pauli`` enum value. Global phase
        is discarded. Returns the updated error label.
        """
        if Pauli is not None and isinstance(pauli, Pauli):  # enum path
            label = str(pauli)
        else:
            if not isinstance(pauli, str):
                raise TypeError("apply_pauli expects str or Pauli enum.")
            label = pauli.upper()

        x_new, z_new = _encode_pauli(label)
        # XOR composition (mod 2 addition)
        self._x_bit ^= x_new
        self._z_bit ^= z_new
        if self.keep_history:
            self._history.append(self.current_error)
        return self.current_error

    def sample_error(self, p_x: Optional[float] = None, p_z: Optional[float] = None, rng: Optional[random.Random] = None) -> str:
        """Stochastically sample & apply a fresh physical error.

        Independent sampling of X and Z faults; if both occur we record a Y.
        Probabilities may be overridden per call; defaults fall back to the
        qubit's stored ``p_x`` and ``p_z``.

        Returns the Pauli label actually applied ('I','X','Y','Z').
        """
        # Use provided Random instance or global functions.
        rand_func = (rng.random if rng is not None else random.random)
        p_x_eff = self.p_x if p_x is None else p_x
        p_z_eff = self.p_z if p_z is None else p_z

        if not (0.0 <= p_x_eff <= 1.0 and 0.0 <= p_z_eff <= 1.0):
            raise ValueError("Probabilities p_x, p_z must lie in [0,1].")

        flip_x = rand_func() < p_x_eff
        flip_z = rand_func() < p_z_eff
        if not flip_x and not flip_z:
            applied = "I"
        elif flip_x and not flip_z:
            applied = "X"
        elif not flip_x and flip_z:
            applied = "Z"
        else:
            applied = "Y"

        # Apply sampled error operator
        self.apply_pauli(applied)
        return applied

    # ------------------------------------------------------------------
    # Utility / representation
    # ------------------------------------------------------------------
    def set_probabilities(self, p_x: float | None = None, p_z: float | None = None) -> None:
        """Update per-qubit error probabilities in place."""
        if p_x is not None:
            if not (0.0 <= p_x <= 1.0):
                raise ValueError("p_x must be in [0,1].")
            self.p_x = p_x
        if p_z is not None:
            if not (0.0 <= p_z <= 1.0):
                raise ValueError("p_z must be in [0,1].")
            self.p_z = p_z

    def __repr__(self) -> str:  # for debugging
        lbl = f"[{self.label}]" if self.label else ""
        return f"PhysicalQubit{lbl}(error={self.current_error}, p_x={self.p_x}, p_z={self.p_z})"

    def __str__(self) -> str:
        return self.__repr__()


__all__ = ["PhysicalQubit"]



