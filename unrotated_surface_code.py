"""
This is just a visualization module for the unrotated surface code.
It follows a very similar structure to the other classes. 
"""

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from typing import List, Tuple, Dict, Union, Literal, Optional

class UnrotatedSurfaceCode:

    def __init__(self, distance: int):
        self.d = distance
        self.qubit_coords, self.x_stabilisers_coords, self.z_stabilisers_coords = self.create(distance)
        self.stabilisers_coords = self.x_stabilisers_coords + self.z_stabilisers_coords
        
        self.index_mapping = self.get_index_mapping()
        self.inverse_mapping = self.get_inverse_mapping()
        
        self.stab_indices = sorted(
            [self.inverse_mapping[c] for c in self.stabilisers_coords],
            key=lambda idx: (self.index_mapping[idx][0][1], self.index_mapping[idx][0][0])
        )

    def get_data_qubits(self, _as: Literal["coord", "idx"] = "coord") -> Union[List[Tuple[int, int]], List[int]]:
        if _as == "coord":
            return self.qubit_coords
        else:
            return [self.inverse_mapping[c] for c in self.qubit_coords]

    def get_stabilisers(self, _as: Literal["coord", "idx"] = "coord") -> Union[List[Tuple[int, int]], List[int]]:
        if _as == "coord":
            return self.stabilisers_coords
        else:
            return [self.inverse_mapping[c] for c in self.stabilisers_coords]

    def get_all_qubits(self, _as: Literal["coord", "idx"] = "coord") -> Union[List[Tuple[int, int]], List[int]]:
        all_coords = self.qubit_coords + self.stabilisers_coords
        if _as == "coord":
            return all_coords
        else:
            return [self.inverse_mapping[c] for c in all_coords]

    @staticmethod
    def create(d: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        qubit_coords = []
        x_stab_coords = []
        z_stab_coords = []
        
        extent = 2 * d - 1
        
        for x in range(extent):
            for y in range(extent):
                if (x + y) % 2 == 0:
                    qubit_coords.append((x, y))
                else:
                    if x % 2 == 1: 
                        x_stab_coords.append((x, y))
                    else:
                        z_stab_coords.append((x, y))
                        
        return qubit_coords, x_stab_coords, z_stab_coords

    def get_index_mapping(self) -> Dict[int, Tuple[Tuple[int, int], str]]:
        mapping = {}
        all_coords = self.qubit_coords + self.x_stabilisers_coords + self.z_stabilisers_coords
        all_coords.sort(key=lambda c: (c[1], c[0]))
        
        for i, coord in enumerate(all_coords):
            if coord in self.x_stabilisers_coords:
                qtype = "X_stab"
            elif coord in self.z_stabilisers_coords:
                qtype = "Z_stab"
            else:
                qtype = "data"
            mapping[i] = (coord, qtype)
        return mapping

    def get_inverse_mapping(self) -> Dict[Tuple[int, int], int]:
        return {v[0]: k for k, v in self.index_mapping.items()}

    def get_surrounding_data_qubits(self, stab_coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = stab_coord
        potential = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        neighbors = []
        for nx, ny in potential:
            if (nx, ny) in self.inverse_mapping:
                idx = self.inverse_mapping[(nx, ny)]
                if self.index_mapping[idx][1] == "data":
                    neighbors.append((nx, ny))
        return neighbors

    def get_surrounding_ancilla_qubits(self, data_coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = data_coord
        potential = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        neighbors = []
        for nx, ny in potential:
            if (nx, ny) in self.inverse_mapping:
                idx = self.inverse_mapping[(nx, ny)]
                if self.index_mapping[idx][1] in ["X_stab", "Z_stab"]:
                    neighbors.append((nx, ny))
        return neighbors

    def get_syndrome(self, x_errors: List[Union[int, Tuple[float, float]]] = [], z_errors: List[Union[int, Tuple[float, float]]] = []) -> np.ndarray:
        def to_coords(errors):
            coords = set()
            for e in errors:
                if isinstance(e, int):
                    if e in self.index_mapping:
                        coords.add(self.index_mapping[e][0])
                elif isinstance(e, tuple):
                    coords.add(e)
            return coords

        x_err_coords = to_coords(x_errors)
        z_err_coords = to_coords(z_errors)
        
        syndrome = []
        
        x_stabs = [idx for idx in self.stab_indices if self.index_mapping[idx][1] == "X_stab"]
        z_stabs = [idx for idx in self.stab_indices if self.index_mapping[idx][1] == "Z_stab"]
        
        for idx in x_stabs:
            coord = self.index_mapping[idx][0]
            neighbors = self.get_surrounding_data_qubits(coord)
            parity = 0
            for n in neighbors:
                if n in z_err_coords:
                    parity += 1
            syndrome.append(parity % 2)
            
        for idx in z_stabs:
            coord = self.index_mapping[idx][0]
            neighbors = self.get_surrounding_data_qubits(coord)
            parity = 0
            for n in neighbors:
                if n in x_err_coords:
                    parity += 1
            syndrome.append(parity % 2)
            
        return np.array(syndrome, dtype=int)

    def visualize(self, result: Optional[np.ndarray] = None, 
                  show_ancillas: bool = True, 
                  show_index: bool = False, 
                  x_err: List[Union[int, Tuple[int, int]]] = [], 
                  z_err: List[Union[int, Tuple[int, int]]] = [],
                  show_legend: bool = True) -> None:
        
        sns.set_style("darkgrid")
        mpl.rcParams.update(  
            {
                "font.size": 12,
                "grid.color": "0.5",
                "grid.linestyle": "--",
                "grid.linewidth": 0.6,
                "xtick.color": "black",
                "ytick.color": "black",
            }
        )

        if result is None:
            if x_err or z_err:
                result = self.get_syndrome(x_errors=x_err, z_errors=z_err) # type: ignore
            else:
                x_stabs_count = len(self.x_stabilisers_coords)
                z_stabs_count = len(self.z_stabilisers_coords)
                result = np.zeros(x_stabs_count + z_stabs_count, dtype=int)

        _, ax = plt.subplots(figsize=(8, 8))
        
        colors = {
            'X_stab': {0: "#79ABD4", 1: "#F07084"},
            'Z_stab': {0: "#385B8B", 1: "#9D212F"}
        }
        
        x_stabs_indices = [idx for idx in self.stab_indices if self.index_mapping[idx][1] == "X_stab"]
        z_stabs_indices = [idx for idx in self.stab_indices if self.index_mapping[idx][1] == "Z_stab"]
        
        x_results = result[:len(x_stabs_indices)]
        z_results = result[len(x_stabs_indices):]
        
        def to_coords(errors):
            coords = set()
            for e in errors:
                if isinstance(e, int):
                    if e in self.index_mapping:
                        coords.add(self.index_mapping[e][0])
                elif isinstance(e, tuple):
                    coords.add(e)
            return coords

        x_error_coords = to_coords(x_err)
        z_error_coords = to_coords(z_err)

        def plot_stabilizer_patch(coord, qtype, val):
            neighbors = self.get_surrounding_data_qubits(coord)
            color = colors[qtype][val]
            
            if len(neighbors) > 1:
                center_x = sum(p[0] for p in neighbors) / len(neighbors)
                center_y = sum(p[1] for p in neighbors) / len(neighbors)
                
                sorted_neighbors = sorted(
                    neighbors, 
                    key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x)
                )
                poly = Polygon(sorted_neighbors, facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
                ax.add_patch(poly)

        for i, idx in enumerate(x_stabs_indices):
            coord = self.index_mapping[idx][0]
            val = x_results[i]
            plot_stabilizer_patch(coord, 'X_stab', val)
            if show_index:
                 ax.text(coord[0], coord[1], str(idx), color='black', ha='center', va='center', fontsize=8, fontweight='bold')

        for i, idx in enumerate(z_stabs_indices):
            coord = self.index_mapping[idx][0]
            val = z_results[i]
            plot_stabilizer_patch(coord, 'Z_stab', val)
            if show_index:
                 ax.text(coord[0], coord[1], str(idx), color='white', ha='center', va='center', fontsize=8, fontweight='bold')

        data_x = [c[0] for c in self.qubit_coords]
        data_y = [c[1] for c in self.qubit_coords]
        
        data_colors = []
        for coord in self.qubit_coords:
            has_x = coord in x_error_coords
            has_z = coord in z_error_coords
            
            if has_x and has_z:
                data_colors.append('cyan') 
            elif has_x:
                data_colors.append('gold')
            elif has_z:
                data_colors.append('palegreen')
            else:
                data_colors.append('#F0F0F0')
                
        ax.scatter(data_x, data_y, s=100, c=data_colors, edgecolor='black', zorder=10, label='Data Qubit')
        
        if show_ancillas:
             x_anc_x = [c[0] for c in self.x_stabilisers_coords]
             x_anc_y = [c[1] for c in self.x_stabilisers_coords]
             ax.scatter(x_anc_x, x_anc_y, s=75, c='silver', marker='o', edgecolor="black", zorder=11, label='X Ancilla')
             
             z_anc_x = [c[0] for c in self.z_stabilisers_coords]
             z_anc_y = [c[1] for c in self.z_stabilisers_coords]
             ax.scatter(z_anc_x, z_anc_y, s=75, c='silver', marker='o', edgecolor="black", zorder=11, label='Z Ancilla')
        
        if show_index:
            for coord in self.qubit_coords:
                 idx = self.inverse_mapping[coord]
                 ax.text(coord[0], coord[1], str(idx), color='black', ha='center', va='center', fontsize=8, zorder=12)
            for coord in self.x_stabilisers_coords + self.z_stabilisers_coords:
                 idx = self.inverse_mapping[coord]
                 ax.text(coord[0], coord[1], str(idx), color='black', ha='center', va='center', fontsize=8, zorder=12)

        ax.set_aspect('equal')
        extent = 2 * self.d - 1
        ax.set_xlim(-1, extent)
        ax.set_ylim(-1, extent)
        ax.set_xticks(range(extent))
        ax.set_yticks(range(extent))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        ax.set_title(f"Unrotated Surface Code (d={self.d})")
        
        if show_legend:
            legend_elements = [
                Patch(facecolor=colors['X_stab'][0], edgecolor='k', label='X-Stab (Ok)'),
                Patch(facecolor=colors['X_stab'][1], edgecolor='k', label='X-Syndrome (Error)'),
                Patch(facecolor=colors['Z_stab'][0], edgecolor='k', label='Z-Stab (Ok)'),
                Patch(facecolor=colors['Z_stab'][1], edgecolor='k', label='Z-Syndrome (Error)'),
                Line2D([0], [0], marker='o', color='w', label='Data Qubit (No Error)',
                              markerfacecolor='#F0F0F0', markersize=10, markeredgecolor='k'),
                Line2D([0], [0], marker='s', color='w', label='Ancilla Qubit',
                              markerfacecolor='silver', markersize=10, markeredgecolor='k'),
                Line2D([0], [0], marker='o', color='w', label='X Error (On Data)',
                              markerfacecolor='gold', markersize=10, markeredgecolor='k'),
                Line2D([0], [0], marker='o', color='w', label='Z Error (On Data)',
                              markerfacecolor='palegreen', markersize=10, markeredgecolor='k'),
                Line2D([0], [0], marker='o', color='w', label='Y Error (On Data)',
                              markerfacecolor='cyan', markersize=10, markeredgecolor='k'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.5, 1))

        plt.tight_layout()
        if ax is None:
             plt.show()
