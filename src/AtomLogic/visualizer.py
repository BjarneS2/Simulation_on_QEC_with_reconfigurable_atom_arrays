"""A module for representing a view of a surface code lattice.

This module provides a `LatticeView` class that represents a view of a
surface code lattice. A lattice view is a subset of the qubits in a lattice,
and it can be used to apply quantum gates to specific qubits.

Typical usage example:

  >>> from surfq import Lattice
  >>>
  >>> lattice = Lattice(3)
  >>> view = lattice[0, 0]
  >>> view.X()
"""

from typing import final
import importlib.resources
from logging import warning
from pathlib import Path, PosixPath

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch, Polygon, Wedge
from numpy.typing import NDArray

from .pauli import Pauli

sns.set_style("darkgrid")
font_resource = importlib.resources.files("surfq").joinpath("fonts/Roboto-Regular.ttf")

with importlib.resources.as_file(font_resource) as font_path:
    roboto_font = fm.FontProperties(fname=PosixPath(Path(font_path)))
    fm.fontManager.addfont(str(font_path)) 
mpl.rcParams.update(  
    {
        "font.family": roboto_font.get_name(),
        "font.size": 12,
        "grid.color": "0.5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "xtick.color": "black",
        "ytick.color": "black",
    }
)

QubitMask = NDArray[np.uint64]

QubitAxisIndex = int | slice | list[int] | QubitMask

QubitIndex = (
    QubitAxisIndex | tuple[QubitAxisIndex, QubitAxisIndex] | list[tuple[int, int]]
)


@final
class LatticeView:
    """A class representing a view of a surface code lattice.

    A lattice view is a subset of the qubits in a lattice, and it can be used
    to apply quantum gates to specific qubits.

    Attributes:
        L: The size of the original LxL lattice.
        n: The number of qubits in the original lattice.
        paulis: The Pauli operators applied on the original lattice's qubits.
        tableau: The tableau of the stabilizers of the original lattice.
        mask: A mask of indexes that selects the qubits in the view prior to operations.
    """

    def __init__(
        self,
        L: int,
        paulis: NDArray[np.uint8],
        tableau: NDArray[np.uint8],
        qubits: QubitIndex,
    ):
        """Initialise a new lattice view.

        Args:
            L: The size of the original LxL lattice.
            paulis: The Pauli operators applied on the original lattice's qubits.
            tableau: The tableau of the stabilizers.
            qubits: The qubits in the view.
        """
        self.L = L
        self.n = L**2
        self.paulis = paulis
        self.tableau = tableau
        self.mask = self._build_mask(qubits)

    def _build_mask(self, qubits: QubitIndex) -> QubitMask:
        """
        Transform qubit indices into a valid mask.

        Qubit indices can be specified using either single- or double-axis notation:

        - Single-axis: Qubits are indexed over the full lattice, i.e., [0, n)
        - Double-axis: Qubits are indexed along x and y axes, i.e., [0, L) for each axis

        Each index may be one of:
        - int: A single qubit within the respective range
        - list[int]: Multiple qubits within the respective range
        - slice: Multiple qubits within the respective range
        - np.ndarray: Multiple qubits within the respective range
        - list[tuple[int, int]]: Explicit list of (x, y) coordinates

        The lattice is read from left to right and top to bottom.

        Examples:
        - Single qubit:
            - lattice[6]           : 7th qubit in the lattice
            - lattice[2, 5]        : Qubit at column 3, row 6

        - Slicing:
            - lattice[:]            : All qubits in the lattice
            - lattice[:, :]         : All qubits in the lattice
            - lattice[6, :]         : All qubits in column 7
            - lattice[:, 2]         : All qubits in row 3
            - lattice[::2, 1::2]    : Qubits at odd rows and even columns

        - Lists and arrays:
            - lattice[[1, 5, 6]]               : Qubits 2, 6, and 7
            - lattice[np.array([1, 5, 6])]     : Qubits 2, 6, and 7
            - lattice[[1, 5, 6], 2]            : Qubits 2, 6, and 7 in row 3
            - lattice[:, np.array([1, 5, 6])]  : Qubits 2, 6, and 7 in each column

        - Coordinates list:
            - lattice[[(1, 0), (5, 2)]] : Qubits at coordinates (1, 0) and (5, 2)
        """
        if isinstance(qubits, tuple):
            xs = qubits[0]
            if isinstance(xs, slice):
                start = 0 if xs.start is None else int(xs.start)  
                stop = self.L if xs.stop is None else int(xs.stop)  
                step = 1 if xs.step is None else int(xs.step)  
                xs = np.arange(start, stop, step, dtype=np.uint64)

            ys = qubits[1]
            if isinstance(ys, slice):
                start = 0 if ys.start is None else int(ys.start)  
                stop = self.L if ys.stop is None else int(ys.stop)  
                step = 1 if ys.step is None else int(ys.step)  
                ys = np.arange(start, stop, step, dtype=np.uint64)

            xs_len = 1 if isinstance(xs, int) else len(xs)
            ys_len = 1 if isinstance(ys, int) else len(ys)

            return (np.tile(xs, ys_len) + np.repeat(ys, xs_len) * self.L).astype(
                np.uint64
            )
        elif isinstance(qubits, slice):
            start = 0 if qubits.start is None else qubits.start  
            stop = self.n if qubits.stop is None else qubits.stop  
            step = 1 if qubits.step is None else qubits.step  
            return np.arange(start, stop, step, dtype=np.uint64)
        elif isinstance(qubits, int):
            return np.array([qubits], dtype=np.uint64)
        elif isinstance(qubits, list):
            if all(isinstance(item, int) for item in qubits):
                return np.array(qubits, dtype=np.uint64)
            else:
                return np.array([x + y * self.L for x, y in qubits])  
        else:
            return qubits

    def H(self):
        """Apply a Hadamard gate to the qubits in the view."""
        self.paulis[self.mask] = (self.paulis[self.mask] >> 1) + (
            self.paulis[self.mask] << 1
        ) % 4

        mask = self.mask
        self.tableau[:, mask], self.tableau[:, mask + self.n] = (
            self.tableau[:, mask + self.n].copy(),
            self.tableau[:, mask].copy(),
        )
        self.tableau[:, -1] ^= np.bitwise_xor.reduce(
            self.tableau[:, mask] & self.tableau[:, mask + self.n], axis=1
        )
        return self

    def S(self):
        """Apply an S gate to the qubits in the view."""
        mask = self.mask
        self.tableau[:, mask + self.n] ^= self.tableau[:, mask]

        self.tableau[:, -1] ^= np.bitwise_xor.reduce(
            self.tableau[:, mask] & self.tableau[:, mask + self.n], axis=1
        )
        # self.tableau[:, -1] ^= self.tableau[:, mask] & self.tableau[:, mask + self.n]
        return self

    def CX(self, target: "LatticeView"):
        """Apply a CNOT gate to the qubits in the view.

        Args:
            target: The target qubits for the CNOT gate.

        Raises:
            ValueError: If the number of control and target qubits do not match.
        """
        c_mask = self.mask
        t_mask = target.mask
        if len(c_mask) != len(t_mask):
            raise ValueError(
                f"dimension mismatch between control and target indexes: \
                {self.mask} and {target.mask}"
            )
        self.tableau[:, t_mask] ^= self.tableau[:, c_mask]
        self.tableau[:, self.n + c_mask] ^= self.tableau[:, self.n + t_mask]
        self.tableau[:, -1] ^= np.bitwise_xor.reduce(
            self.tableau[:, c_mask]
            & self.tableau[:, self.n + t_mask]
            & (self.tableau[:, t_mask] ^ self.tableau[:, self.n + c_mask] ^ 1),
            axis=1,
        )
        return self

    def Z(self):
        """Apply a Z gate to the qubits in the view."""
        self.paulis[self.mask] ^= Pauli.Z.value
        return self.S().S()

    def X(self):
        """Apply an X gate to the qubits in the view."""
        return self.H().Z().H()

    def Y(self):
        """Apply a Y gate to the qubits in the view."""
        return self.X().Z()

    def show(self):
        """Plot the original lattice."""
        return plot_lattice(self.L, self.tableau, self.paulis)
    



def plot_lattice(L: int, tableau: NDArray[np.uint8], paulis: NDArray[np.uint8]) -> None:
    """Plot a surface code lattice.

    This function plots a surface code lattice, including the qubits, the
    stabilizers, and the Pauli operators.

    Args:
        L: The size of the lattice.
        tableau: The tableau of the stabilizers.
        paulis: The Pauli operators on the qubits.
    """
    n = L**2

    if L >= 20:
        warning("Lattice dimension is too big to show")
        return

    s = 60 if L <= 10 else 25
    fontsize = 10 if L <= 10 else 7

    x_grid, y_grid = np.meshgrid(np.arange(L), np.arange(L))
    grid_x, grid_y = x_grid.flatten(), y_grid.flatten()
    ax = plt.gca() 

    def draw_polygon_or_wedge(
        indices: NDArray[np.intp],
        color: str,
    ) -> None:
        x = indices % L
        y = L - 1 - indices // L

        if len(indices) == 4:
            pts = np.column_stack((x, y))
            pts[[-2, -1]] = pts[[-1, -2]]
            patch = Polygon(
                list(pts),
                facecolor=color,
                edgecolor="k",
                linewidth=0.75,
            )
        elif len(indices) == 2:
            cx, cy = float(np.mean(x)), float(np.mean(y))
            if y[0] == y[1] == 0:
                theta1, theta2 = (180, 0)
            elif y[0] == y[1] == L - 1:
                theta1, theta2 = (0, 180)
            elif x[0] == x[1] == 0:
                theta1, theta2 = (90, -90)
            else:
                theta1, theta2 = (-90, 90)
            patch = Wedge(
                center=(cx, cy),
                r=0.5,
                theta1=theta1,
                theta2=theta2,
                facecolor=color,
                edgecolor="k",
                linewidth=0.75,
            )
        else:
            return

        ax.add_patch(patch)  

    for stab in tableau[: (n - 1) // 2]: 
        indices = np.where(stab[:n] == 1)[0] 
        if stab[-1] != 1:
            draw_polygon_or_wedge(indices, "#6188b2")
        else:
            draw_polygon_or_wedge(indices, "#D15567")

    for stab in tableau[(n - 1) // 2 :]: 
        indices = np.where(stab[n:-1] == 1)[0] 
        if stab[-1] != 1:
            draw_polygon_or_wedge(indices, "#87b1d3")
        else:
            draw_polygon_or_wedge(indices, "#E17C88")

    ax.scatter(grid_x, grid_y, s=s, color="white", edgecolor="black")  

    ax.set_aspect("equal")  
    ax.set_yticks(np.arange(L))  
    ax.set_xticks(np.arange(L))  
    ax.set_xlim(-1, L)  
    ax.set_ylim(-1, L)  
    ax.set_yticklabels(np.arange(L)[::-1])  
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, labelsize=fontsize)  
    ax.tick_params(axis="y", labelsize=fontsize)  
    ax.grid(True)  

    ax.legend(  
        handles=[
            Patch(
                facecolor="#6188b2",
                edgecolor="black",
                label="X-Stabiliser",
            ),
            Patch(
                facecolor="#87b1d3",
                edgecolor="black",
                label="Z-Stabiliser",
            ),
            Patch(
                facecolor="#D15567",
                edgecolor="black",
                label="X-Stabiliser Syndrome",
            ),
            Patch(
                facecolor="#E17C88",
                edgecolor="black",
                label="Z-Stabiliser Syndrome",
            ),
        ],
        handlelength=1,
        handleheight=1,
        borderpad=0.5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        fontsize=10,
        markerscale=2.0,
        ncol=2,
    )

    for q, op in enumerate(paulis): 
        if not op:
            continue
        p = Pauli(op)
        x, y = q % L, L - q // L - 1
        _ = plt.scatter(x, y, color=p.color, edgecolor="black", s=s)
        _ = plt.text(  
            x + 0.1,
            y + 0.1,
            f"$\\mathrm{{{p}}}_{{{q}}}$",
            ha="left",
            va="bottom",
            fontsize=fontsize,
        )

    plt.show()  


