"""
I want to create a class that represents a surface code for qec.
I want to use the Surface Code class to initialize a surface code of a given distance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge, Patch
from typing import List, Tuple, Dict, Any, Union, Optional
import matplotlib as mpl
import seaborn as sns
import stim


sns.set_style("darkgrid")
mpl.rcParams.update(  # pyright: ignore[reportUnknownMemberType]
    {
        "font.size": 12,
        "grid.color": "0.5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "xtick.color": "black",
        "ytick.color": "black",
    }
)


class SurfaceCode:

    def __init__(self, distance: int):
        if distance < 0 or distance % 1 != 0 :
            raise ValueError("Distance must be an odd positive integer.")

        self.d = distance
        self.number_of_qubits = distance ** 2
        # qubit coordinates is the list of data qubits and stabilisers coordinates the list of ancillas
        self.qubit_coords, self.x_stabilisers_coords, self.z_stabilisers_coords = self.create(distance)
        self.stabilisers_coords = self.x_stabilisers_coords + self.z_stabilisers_coords
        self.index_mapping = self.get_index_mapping()
        self.inverse_mapping = self.get_inverse_mapping()

    @staticmethod
    def create(L: int) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]], List[Tuple[float, float]]]:
        data_bits: List[Tuple[int, int]] = []
        for r in range(L):
            for c in range(L):
                data_bits.append((c, r))

        x_ancilla: List[Tuple[float, float]] = []
        # Set up inner X stabilisers
        for row in range(L - 1):
            for col in range((L - 1) // 2):
                j = 2 * col + (0 if row % 2 == 0 else 1)
                x_ancilla.append((j + 0.5, row + 0.5))

        # Set up top X stabilisers
        for col in range((L - 1) // 2):
            j = 2 * col + 1
            x_ancilla.append((j + 0.5, -0.5))

        # Set up bottom X stabilisers
        for col in range((L - 1) // 2):
            j = 2 * col
            x_ancilla.append((j + 0.5, L - 1 + 0.5))

        z_ancilla: List[Tuple[float, float]] = []
        # Set up inner Z stabilisers
        for row in range(L - 1):
            for col in range((L - 1) // 2):
                j = 2 * col + (0 if row % 2 == 1 else 1)
                z_ancilla.append((j + 0.5, row + 0.5))

        # Set up left Z stabilisers
        for row in range((L - 1) // 2):
            z_ancilla.append((-0.5, 2 * row + 0.5))

        # Set up right Z stabilisers
        for row in range((L - 1) // 2):
            i = (2 * row + 1) * L
            z_ancilla.append((L - 1 + 0.5, 2 * row + 1 + 0.5))

        return data_bits, x_ancilla, z_ancilla

    def get_index_mapping(self) -> Dict[int, Tuple[Tuple[Union[int, float], Union[int, float]], str]]:
        """
         {0: ((coord_x, coord_y), "data"), 1: ((coord_x, coord_y), "X_stab"), ...}
        """
        all_qubits = []
        for coord in self.qubit_coords:
            all_qubits.append({'coord': coord, 'type': 'data', 'sort_key': (coord[1], coord[0])})

        for coord in self.x_stabilisers_coords:
            all_qubits.append({'coord': coord, 'type': 'X_stab', 'sort_key': (coord[1], coord[0])})

        for coord in self.z_stabilisers_coords:
            all_qubits.append({'coord': coord, 'type': 'Z_stab', 'sort_key': (coord[1], coord[0])})
        
        # Sort qubits: primarily by y-coordinate, secondarily by x-coordinate.
        # This creates the row-by-row ordering.
        all_qubits.sort(key=lambda q: q['sort_key'])

        index_mapping: Dict[int, Tuple[Tuple[Union[int, float], Union[int, float]], str]] = {}
        for i, qubit in enumerate(all_qubits):
            index_mapping[i] = (qubit['coord'], qubit['type'])

        return index_mapping
    
    def get_inverse_mapping(self) -> Dict[Tuple[Union[int, float], Union[int, float]], int]:
        """inverses the mapping of self.index_mapping"""
        inverse_mapping: Dict[Tuple[Union[int, float], Union[int, float]], int] = {}
        
        for idx, (coord, _) in self.index_mapping.items():
            inverse_mapping[coord] = idx

        return inverse_mapping

    def get_surrounding_data_qubits(self, stab_coord: Tuple[float, float]) -> List[Tuple[int, int]]:
        """ From the index map we can get the X stabilizers and the Z stabilizers """
        """ Given a stabilizer coordinate we can just look for qubits at position """
        """ (x +/- 0.5, y +/- 0.5) to find the data qubits it acts on given these """
        """ are in the index mapping contained (edge stabilizers got 2 neighbors) """
        surrounding_qubits: List[Tuple[int, int]] = []
        x, y = stab_coord
        deltas = [(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)]

        for dx, dy in deltas:
            neighbor_coord: Tuple[int, int] = (int(x + dx), int(y + dy))

            if neighbor_coord in self.qubit_coords:
                surrounding_qubits.append(neighbor_coord)

        return surrounding_qubits

    def plot(self):

        # Surface code plotting, to see if the structure is correct
        plt.figure(figsize=(self.d, self.d))
        
        data_x = [coord[0] for coord in self.qubit_coords]
        data_y = [coord[1] for coord in self.qubit_coords]
        plt.scatter(data_x, data_y, color='blue', label='Data Qubits', s=65)

        anc_x = [coord[0] for coord in self.stabilisers_coords]
        anc_y = [coord[1] for coord in self.stabilisers_coords]
        
        x_stab_x = [coord[0] for coord in self.x_stabilisers_coords]
        x_stab_y = [coord[1] for coord in self.x_stabilisers_coords]
        
        z_stab_x = [coord[0] for coord in self.z_stabilisers_coords]
        z_stab_y = [coord[1] for coord in self.z_stabilisers_coords]

        plt.scatter(x_stab_x, x_stab_y, color='red', marker='s', label='X Stabilizers', s=65)
        plt.scatter(z_stab_x, z_stab_y, color='green', marker='s', label='Z Stabilizers', s=65)

        """ Create a number indexing for each qubit and add the number 
            on the markers of the plot in the respective positions """
        for index, (coord, qtype) in self.index_mapping.items():
            plt.text(coord[0], coord[1], str(index), color='white', 
                     fontsize=8, ha='center', va='center')

        plt.xlim(-1, self.d)
        plt.ylim(-1, self.d)
        plt.xticks(range(self.d))
        plt.yticks(range(self.d))
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.legend()
        plt.title(f'Surface Code d={self.d}')
        plt.grid(True)
        plt.show()
        
    def build_in_stim_idea(self):
        """ 
            I wanna translate/build this in the stim framework. 
            for this I need to create the stim circuit and then
            insert all the qubits and stabilizers/ancillas.

            I should first reset all qubits, and then think of 
            how to measure the stabilizers. Meaning which operations
            from data qubits to stabilizers I need to apply.
            
            For this I need to find the surrounding data qubits for 
            each stabilizer and then apply respective CNOTs/Hadamards,
            depending on the stabilizer type. Finally I need to measure
            the stabilizers and repeat this for a number of rounds.

            For measuring Z Errors: 
                - Apply Hadamard on the stabilizer
                - Apply CNOT from data qubits to stabilizer
                - Apply Hadamard on the stabilizer
                - Measure the stabilizer
            For measuring X Errors:
                - Apply CNOT from data qubits to stabilizer
                - Measure the stabilizer

        """
        pass

    def build_in_stim(self, rounds: int = 1) -> stim.Circuit:
        circuit = stim.Circuit()

        for idx, (coord, _) in self.index_mapping.items():
            circuit.append("QUBIT_COORDS", [idx], [coord[0], coord[1]])

        data_indices = [k for k, v in self.index_mapping.items() if v[1] == 'data']
        x_stab_indices = [k for k, v in self.index_mapping.items() if v[1] == 'X_stab']
        z_stab_indices = [k for k, v in self.index_mapping.items() if v[1] == 'Z_stab']
        all_ancillas = x_stab_indices + z_stab_indices

        circuit.append("R", self.index_mapping.keys())  # type: ignore
        circuit.append("TICK")  # type: ignore

        loop_body = stim.Circuit()

        loop_body.append("R", all_ancillas)  # type: ignore
        
        # X-Stabilizers (Measure X parity -> Detect Z Errors)
        # Requirement: Apply H on stabilizer
        if x_stab_indices:
            loop_body.append("H", x_stab_indices)  # type: ignore
        
        # Apply CNOTs
        # We iterate over every ancilla, find its neighbors, and apply the CNOT.
        # Note: In a real QEC, the order of CNOTs (N, E, W, S) matters for hook errors.
        # Here we perform them in the order found in 'get_surrounding_data_qubits'.
        
        for ancilla_idx in all_ancillas:
            coord, qtype = self.index_mapping[ancilla_idx]
            neighbors = self.get_surrounding_data_qubits(coord) # type: ignore
            
            for neighbor_coord in neighbors:
                neighbor_idx = self.inverse_mapping[neighbor_coord]
                
                if qtype == 'X_stab':
                    # Measuring X: H(Anc) -> CNOT(Anc, Data) -> H(Anc)
                    # This propagates X from Ancilla to Data (checking X parity)
                    loop_body.append("CNOT", [ancilla_idx, neighbor_idx])  # type: ignore
                elif qtype == 'Z_stab':
                    # Measuring Z: CNOT(Data, Anc)
                    # This propagates Z from Data to Ancilla (checking Z parity)
                    loop_body.append("CNOT", [neighbor_idx, ancilla_idx])  # type: ignore

        # X-Stabilizers: Apply closing H on stabilizer
        if x_stab_indices:
            loop_body.append("H", x_stab_indices)  # type: ignore

        # --- C. Measure Ancillas ---
        loop_body.append("M", all_ancillas)  # type: ignore
        loop_body.append("TICK")  # type: ignore

        # Add loop to main circuit
        circuit += loop_body * rounds

        return circuit
        
    def run_simulation(self, circuit: stim.Circuit, shots: int = 1000) -> np.ndarray:
        sampler = circuit.compile_sampler()
        results = sampler.sample(shots=shots)
        return results
    
    def visualize_results_sketch(self, result: np.ndarray):
        """ Visualize the measurement results one the surface code """
        """ Therefore I would liket to plot all of the data qubits """
        """ but this time color the plaquetes instead of individual"""
        """ ancillas according to the measurement results. """  
    
    def visualize_results(self, result: np.ndarray):
        """ 
        Visualize the measurement results on the surface code.
        Plots data qubits as circles and colors the plaquettes (stabilizers)
        based on the measurement outcome (Syndrome vs Normal).
        """
        
        # 1. Define Color Palette
        colors = {
            'X_stab': {0: '#A3C1DA', 1: '#D15567'},  # Muted Blue vs Muted Red
            'Z_stab': {0: '#6C8EBF', 1: '#E17C88'}   # Darker Blue vs Lighter Red
        }

        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 2. Map results back to specific stabilizers
        x_indices = [k for k, v in self.index_mapping.items() if v[1] == 'X_stab']
        z_indices = [k for k, v in self.index_mapping.items() if v[1] == 'Z_stab']
        
        if len(result) != (len(x_indices) + len(z_indices)):
            raise ValueError(f"Result length {len(result)} does not match number of stabilizers.")

        x_results = result[:len(x_indices)]
        z_results = result[len(x_indices):]

        # Helper to plot a single stabilizer
        def plot_stabilizer_patch(coord, neighbors, qtype, val):
            color = colors[qtype][val]
            cx, cy = coord
            
            # Sort neighbors angularly around the center to ensure correct polygon drawing
            neighbors = sorted(neighbors, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
            
            if len(neighbors) == 4:
                # Bulk Stabilizer -> Polygon
                poly = Polygon(neighbors, facecolor=color, edgecolor='black', alpha=0.9, linewidth=1)
                ax.add_patch(poly)
            
            elif len(neighbors) == 2:
                # Boundary Stabilizer -> Wedge (Half-circle)
                
                # --- START OF FIX: SHIFTING CENTER ---
                shift_x = 0
                shift_y = 0
                L = self.d # Distance/Size of the code grid
                
                # Determine inward shift based on the coordinate being outside the grid [0, L-1]
                # Left edge: cx = -0.5 (shift +0.5)
                if cx < 0:
                    shift_x = 0.5
                # Right edge: cx = L - 0.5 (shift -0.5)
                elif cx > L - 1:
                    shift_x = -0.5
                
                # Top edge: cy = -0.5 (shift +0.5)
                if cy < 0:
                    shift_y = 0.5
                # Bottom edge: cy = L - 0.5 (shift -0.5)
                elif cy > L - 1:
                    shift_y = -0.5
                
                shifted_cx = cx + shift_x # Line where cx is shifted
                shifted_cy = cy + shift_y # Line where cy is shifted
                # --- END OF FIX: SHIFTING CENTER ---
                
                # Vector from original center (cx, cy) to midpoint of neighbors (mx, my)
                mx, my = (neighbors[0][0] + neighbors[1][0])/2, (neighbors[0][1] + neighbors[1][1])/2
                vec_x, vec_y = mx - cx, my - cy
                
                # Calculate the angle of the vector pointing from the center to the midpoint
                pointing_angle = np.degrees(np.arctan2(vec_y, vec_x))
                
                # Flip the angle so the wedge points into the code
                pointing_angle = (pointing_angle + 180) % 360  
                
                # Wedge spans 180 degrees centered on that new pointing angle
                theta1 = pointing_angle - 90
                theta2 = pointing_angle + 90
                
                # Use the shifted center and the correct radius (0.5)
                wedge = Wedge((shifted_cx, shifted_cy), r=0.5, theta1=theta1, theta2=theta2, 
                              facecolor=color, edgecolor='black', alpha=0.9, linewidth=1)
                ax.add_patch(wedge)

        # 3. Plot X Stabilizers
        for i, idx in enumerate(x_indices):
            coord, qtype = self.index_mapping[idx]
            neighbors = self.get_surrounding_data_qubits(coord)
            plot_stabilizer_patch(coord, neighbors, qtype, x_results[i])

        # 4. Plot Z Stabilizers
        for i, idx in enumerate(z_indices):
            coord, qtype = self.index_mapping[idx]
            neighbors = self.get_surrounding_data_qubits(coord)
            plot_stabilizer_patch(coord, neighbors, qtype, z_results[i])

        # 5. Plot Data Qubits (White/Grey Circles)
        data_x = [c[0] for c in self.qubit_coords]
        data_y = [c[1] for c in self.qubit_coords]
        ax.scatter(data_x, data_y, s=200, color='#F0F0F0', edgecolor='black', zorder=10, label='Data Qubit')

        # 6. Formatting
        ax.set_aspect('equal')
        ax.set_xticks(range(self.d))
        ax.set_yticks(range(self.d))
        ax.set_xlim(-1, self.d)
        ax.set_ylim(-1, self.d)
        ax.set_title(f"Syndrome Measurement (d={self.d})")
        
        # Custom Legend
        legend_elements = [
            Patch(facecolor=colors['X_stab'][0], edgecolor='k', label='X-Stab (OK)'),
            Patch(facecolor=colors['X_stab'][1], edgecolor='k', label='X-Syndrome (Error)'),
            Patch(facecolor=colors['Z_stab'][0], edgecolor='k', label='Z-Stab (OK)'),
            Patch(facecolor=colors['Z_stab'][1], edgecolor='k', label='Z-Syndrome (Error)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.tight_layout()
        plt.show()
