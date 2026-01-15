"""
MultiLogicalSurfaceCode.py
A module for simulating and visualizing multiple logical qubits
using the rotated surface code. The code in this current version
does not support full functionality and no decoding nor useful
simuations will run. This is a current work in progress.

"""
from typing import List, Dict, Any, Tuple, Union, Optional
from matplotlib.patches import Polygon, Patch, Wedge
from surface_code import SurfaceCode
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pymatching as pm
import seaborn as sns
import numpy as np
import stim

class MultiLogicalSurfaceCode:
    def __init__(self, distance: int, num_logical_qubits: int, seed: int = 42):
        self.distance = distance
        self.num_logical_qubits = num_logical_qubits
        self.seed = seed
        self.circuit = stim.Circuit()
        self.quantum_algorithm: Optional[stim.Circuit] = None

        self.logical_cnot_history: List[Tuple[int, int]] = []
        self.after_transversal_cnot = False
        self.stab_measurement_history: Dict[int, List[Any]] = {}
        
        self._initialize_lattice()
        
    def _initialize_lattice(self):
        self.logical_qubits = {}
        self.index_mapping = {}
        self.inverse_mapping = {}
        self.all_qubits_coords = []
        
        current_idx = 0
        
        for i in range(self.num_logical_qubits):
            # Create coordinates using SurfaceCode static method
            data_coords, x_stab_coords, z_stab_coords = SurfaceCode.create(self.distance)
            print(data_coords, x_stab_coords, z_stab_coords)
            
            # Shift coordinates
            shift_x = i * (self.distance + 1)
            
            shifted_data = [(x + shift_x, y) for x, y in data_coords]
            shifted_x_stab = [(x + shift_x, y) for x, y in x_stab_coords]
            shifted_z_stab = [(x + shift_x, y) for x, y in z_stab_coords]
            
            # Collect qubits for a patch & sort
            patch_qubits = []
            for coord in shifted_data:
                patch_qubits.append({'coord': coord, 'type': 'data', 'sort_key': (coord[1], coord[0])})
            for coord in shifted_x_stab:
                patch_qubits.append({'coord': coord, 'type': 'X_stab', 'sort_key': (coord[1], coord[0])})
            for coord in shifted_z_stab:
                patch_qubits.append({'coord': coord, 'type': 'Z_stab', 'sort_key': (coord[1], coord[0])})
            
            # Sort primarily by y, then x (as in SurfaceCode)
            patch_qubits.sort(key=lambda q: q['sort_key'])
            
            patch_indices = []
            patch_data_indices = []
            patch_stab_indices = []
            patch_index_mapping = {}
            
            for q in patch_qubits:
                idx = current_idx
                current_idx += 1
                
                coord = q['coord']
                qtype = q['type']
                
                self.index_mapping[idx] = (coord, qtype)
                self.inverse_mapping[coord] = idx
                
                patch_indices.append(idx)
                patch_index_mapping[idx] = (coord, qtype)
                
                if qtype == 'data':
                    patch_data_indices.append(idx)
                else:
                    patch_stab_indices.append(idx)
            
            # Sort stabilizer indices for deterministic ordering
            patch_stab_indices.sort(key=lambda anc: (
                self.index_mapping[anc][1], # type
                self.index_mapping[anc][0][0], # x
                self.index_mapping[anc][0][1]  # y
            ))

            self.logical_qubits[str(i)] = {
                "qubit_coords": shifted_data,
                "x_stabilisers_coords": shifted_x_stab,
                "z_stabilisers_coords": shifted_z_stab,
                "data_indices": patch_data_indices,
                "stab_indices": patch_stab_indices,
                "all_indices": patch_indices,
                "index_mapping": patch_index_mapping
            }
            
            self.all_qubits_coords.extend(shifted_data + shifted_x_stab + shifted_z_stab)
            
        self.num_physical_qubits = current_idx

    def get_surrounding_data_qubits(self, stab_coord: Tuple[float, float]) -> List[Tuple[int, int]]:
        # Reusing logic from SurfaceCode
        surrounding_qubits: List[Tuple[int, int]] = []
        x, y = stab_coord
        
        if stab_coord not in self.inverse_mapping:
            return []
            
        idx = self.inverse_mapping[stab_coord]
        _, qtype = self.index_mapping[idx]
        
        deltaZ = [(0.5, -0.5), (0.5, 0.5), (-0.5, -0.5), (-0.5, 0.5)]
        deltaX = [(0.5, -0.5), (-0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        deltas = deltaZ if qtype == 'Z_stab' else deltaX
        
        for dx, dy in deltas:
            neighbor_coord = (x + dx, y + dy)
            if neighbor_coord in self.inverse_mapping:
                # Check if it is a data qubit also acts 
                # like a safety type check for the stab coord
                n_idx = self.inverse_mapping[neighbor_coord]
                if self.index_mapping[n_idx][1] == 'data': 
                    surrounding_qubits.append(neighbor_coord) # type: ignore

        return surrounding_qubits

    def measure_local_stabilizers(self, round0:bool=False) -> stim.Circuit:
        circ = stim.Circuit()
        all_stabs = []

        for i in range(self.num_logical_qubits):
            all_stabs.extend(self.logical_qubits[str(i)]["stab_indices"])

        circ.append("R", all_stabs)  # type: ignore

        for i in range(self.num_logical_qubits):
            for anc in self.logical_qubits[str(i)]["stab_indices"]:
                coord, qtype = self.index_mapping[anc]
                neighbors = self.get_surrounding_data_qubits(coord)

                if qtype == "X_stab":
                    circ.append("H", [anc])  # type: ignore

                for n in neighbors:
                    nidx = self.inverse_mapping[n]
                    if qtype == "X_stab":
                        circ.append("CNOT", [anc, nidx])  # type: ignore
                    else:
                        circ.append("CNOT", [nidx, anc])  # type: ignore

                if qtype == "X_stab":
                    circ.append("H", [anc])  # type: ignore

        circ.append("M", all_stabs)  # type: ignore

        for k, anc in enumerate(all_stabs):
            rec = stim.target_rec(-(len(all_stabs) - k))

            # Check if there's a previous measurement in the history for this stabilizer
            if anc in self.stab_measurement_history and self.stab_measurement_history[anc]:
                prev = self.stab_measurement_history[anc][-1]
                if not round0:
                    circ.append("DETECTOR", [rec, prev])  # type: ignore

            self.stab_measurement_history.setdefault(anc, []).append(rec)

        circ.append("TICK") # type: ignore
        return circ

    def measure_joint_stabilizers_after_cnot(self, control: int, target: int) -> stim.Circuit:
        circ = stim.Circuit()

        ctrl = self.logical_qubits[str(control)]
        tgt = self.logical_qubits[str(target)]

        shift = self.distance + 1

        # X@X joint stabilizers
        for anc in ctrl["stab_indices"]:
            coord, qtype = self.index_mapping[anc]
            if qtype != "X_stab":
                continue

            neigh_c = self.get_surrounding_data_qubits(coord)
            neigh_t = [(x + shift, y) for x, y in neigh_c]

            circ.append("R", [anc])  # type: ignore
            circ.append("H", [anc])  # type: ignore

            for c, t in zip(neigh_c, neigh_t):
                circ.append("CNOT", [anc, self.inverse_mapping[c]])  # type: ignore
                circ.append("CNOT", [anc, self.inverse_mapping[t]])  # type: ignore

            circ.append("H", [anc])  # type: ignore
            circ.append("M", [anc])  # type: ignore

            rec_joint = stim.target_rec(-1)
            prev_ctrl = self.stab_measurement_history[anc][-1]

            # target X
            tgt_coord = (coord[0] + shift, coord[1])
            tgt_anc = self.inverse_mapping[tgt_coord]
            prev_tgt = self.stab_measurement_history[tgt_anc][-1]

            # print("DETECTOR", [rec_joint, prev_ctrl, prev_tgt])
            circ.append("DETECTOR", [rec_joint, prev_ctrl, prev_tgt])  # type: ignore
            self.stab_measurement_history[anc].append(rec_joint)

        # now for ZZ
        for anc in tgt["stab_indices"]:
            coord, qtype = self.index_mapping[anc]
            if qtype != "Z_stab":
                continue

            neigh_t = self.get_surrounding_data_qubits(coord)
            neigh_c = [(x - shift, y) for x, y in neigh_t]

            circ.append("R", [anc]) # type: ignore

            for c, t in zip(neigh_c, neigh_t):
                circ.append("CNOT", [self.inverse_mapping[c], anc]) # type: ignore
                circ.append("CNOT", [self.inverse_mapping[t], anc]) # type: ignore

            circ.append("M", [anc]) # type: ignore

            rec_joint = stim.target_rec(-1)

            prev_tgt = self.stab_measurement_history[anc][-1]

            ctrl_coord = (coord[0] - shift, coord[1])
            ctrl_anc = self.inverse_mapping[ctrl_coord]
            prev_ctrl = self.stab_measurement_history[ctrl_anc][-1]

            # Detector: ZZ+Zc+Zt
            circ.append("DETECTOR", [rec_joint, prev_ctrl, prev_tgt])  # type: ignore

            self.stab_measurement_history[anc].append(rec_joint)

        circ.append("TICK")  # type: ignore
        return circ

    def apply_logical_gate(self, gate: str, control: int, target: int):
        if gate != "CNOT":
            raise NotImplementedError

        if self.quantum_algorithm is None:
            self.quantum_algorithm = stim.Circuit()

        ctrl = self.logical_qubits[str(control)]["data_indices"]
        tgt = self.logical_qubits[str(target)]["data_indices"]

        # To check if X error gets propagated to target
        # mid_idx = len(ctrl) // 2
        # self.quantum_algorithm.append("X", [ctrl[mid_idx]]) # type: ignore

        for c, t in zip(ctrl, tgt):
            self.quantum_algorithm.append("CNOT", [c, t]) # type: ignore

        self.after_transversal_cnot = True
        self.logical_cnot_history.append((control, target))

    def build(self, rounds: int = 3):
        """
        Essentially the same as in SurfaceCode.py but now a bit 
        shorter and modularized since we have more helper 
        functipns. 
        """

        for q, (coord, _) in self.index_mapping.items():
            self.circuit.append("QUBIT_COORDS", [q], coord) 

        for q, (coord, qtype) in self.index_mapping.items():
            if 'data' == qtype:
                self.circuit.append("R", q) # type: ignore

        self.circuit += self.measure_local_stabilizers(round0=True)
        
        if self.quantum_algorithm is not None:
            self.circuit += self.quantum_algorithm

        if self.after_transversal_cnot:
            for c, t in self.logical_cnot_history:
                self.circuit += self.measure_joint_stabilizers_after_cnot(c, t)

        for _ in range(rounds - 1):
            self.circuit += self.measure_local_stabilizers()

    def run_with_pymatching(self, shots=1000, joint_decoding=True):
        """
        Using stims detector error model and pymatching for decoding. 
        """

        model = self.circuit.detector_error_model(decompose_errors=True)

        if not joint_decoding:
            model = model.filter_detectors( # type: ignore
                lambda d: d.coords is None or len(d.coords) < 3
            )

        matching = pm.Matching.from_detector_error_model(model)
        sampler = self.circuit.compile_detector_sampler()
        syndrome, obs = sampler.sample(shots, separate_observables=True)

        preds = matching.decode_batch(syndrome)
        print(syndrome)
        return np.mean(np.any(preds != obs, axis=1))

    def visualize_results(self, result: np.ndarray, show_ancillas: bool = False, show_indices: bool = False):
        """
        Visualize a syndrome result array.
        
        0 -> no error (stabilizer == +1)
        1 -> error (stabilizer == -1)"""

        sns.set_style("darkgrid")
        mpl.rcParams.update({
            "font.size": 12,
            "grid.color": "0.5",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "xtick.color": "black",
            "ytick.color": "black",
        })
        
        colors = {
            'X_stab': {0: "#79ABD4", 1: "#F07084"},
            'Z_stab': {0: "#385B8B", 1: "#9D212F"}
        }
        
        total_width = self.num_logical_qubits * (self.distance + 1)
        fig, ax = plt.subplots(figsize=(total_width, self.distance + 2))
        
        all_stab_indices = []
        for i in range(self.num_logical_qubits):
            all_stab_indices.extend(self.logical_qubits[str(i)]["stab_indices"])
            
        def plot_stabilizer_patch(coord, neighbors, qtype, val):
            center_x, center_y = coord
            neighbors = sorted(neighbors, key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x))
            
            if len(neighbors) == 4:
                poly = Polygon(neighbors, closed=True, facecolor=colors[qtype][val], edgecolor='black', alpha=0.7)
                ax.add_patch(poly)
            elif len(neighbors) == 2:
                mx = (neighbors[0][0] + neighbors[1][0]) / 2
                my = (neighbors[0][1] + neighbors[1][1]) / 2
                vec_x = mx - center_x
                vec_y = my - center_y
                angle = np.degrees(np.arctan2(vec_y, vec_x))
                pointing_angle = (angle + 180) % 360
                theta1 = pointing_angle - 90
                theta2 = pointing_angle + 90
                wedge = Wedge((mx, my), r=0.5, theta1=theta1, theta2=theta2,
                              facecolor=colors[qtype][val], edgecolor='black', alpha=0.7)
                ax.add_patch(wedge)

        for i, idx in enumerate(all_stab_indices):
            if i >= len(result): 
                break
            val = result[i]
            coord, qtype = self.index_mapping[idx]
            neighbors = self.get_surrounding_data_qubits(coord)
            plot_stabilizer_patch(coord, neighbors, qtype, val)
            
            if show_indices:
                ax.text(coord[0], coord[1], str(idx), color='white', 
                        ha='center', va='center', fontsize=9, fontweight='bold', zorder=20)
            
        data_coords = [c for c in self.all_qubits_coords if self.index_mapping[self.inverse_mapping[c]][1] == 'data']
        data_x = [c[0] for c in data_coords]
        data_y = [c[1] for c in data_coords]
        
        ax.scatter(data_x, data_y, s=100, color='#F0F0F0', edgecolor='black', zorder=10, label='Data Qubit')
        
        if show_indices:
            for c in data_coords:
                idx = self.inverse_mapping[c]
                ax.text(c[0], c[1], str(idx), color='black', 
                        ha='center', va='center', fontsize=8, zorder=21)
        
        if show_ancillas:
            anc_coords = [c for c in self.all_qubits_coords if 'stab' in self.index_mapping[self.inverse_mapping[c]][1]]
            anc_x = [c[0] for c in anc_coords]
            anc_y = [c[1] for c in anc_coords]
            ax.scatter(anc_x, anc_y, s=50, color='gray', marker='s', zorder=11)

            if show_indices:
                active_indices = set(all_stab_indices[:len(result)])
                for c in anc_coords:
                    idx = self.inverse_mapping[c]
                    if idx not in active_indices:
                         ax.text(c[0], c[1], str(idx), color='white', 
                                 ha='center', va='center', fontsize=8, zorder=21)

        for i in range(self.num_logical_qubits):
            shift_x = i * (self.distance + 1)
            rect_patch = plt.Rectangle( # type: ignore
                (shift_x - 0.8, -0.8), 
                self.distance + 0.6, 
                self.distance + 0.6,
                fill=False,
                edgecolor='purple',
                linewidth=3,
                linestyle='--'
            )
            ax.add_patch(rect_patch)
            ax.text(shift_x + self.distance/2 - 0.5, -1.2, f"L{i}", color='purple', fontsize=14, fontweight='bold')

        ax.set_aspect('equal')
        # ax.invert_yaxis()
        # ax.set_title(f"Multi-Logical Qubit Surface Code (d={self.distance}, N={self.num_logical_qubits})")
        plt.tight_layout()
        plt.show()

    def run_simulation(self, shots: int = 1000) -> np.ndarray:
        """simply run the circuit and return the results"""
        sampler = self.circuit.compile_detector_sampler()
        results:np.ndarray = sampler.sample(shots=shots) # type: ignore
        print(results)
        return results

    def diagram(self, *args, **kwargs):
        print(self.circuit.diagram(*args, **kwargs))

    def get_syndrome(self, X_errors: Dict[str, List[Union[int, Tuple[float, float]]]] = {}, Z_errors: Dict[str, List[Union[int, Tuple[float, float]]]] = {}) -> np.ndarray:
        """
        Calculate the syndrome given dictionaries of X and Z errors.
        Keys are logical qubit indices as strings ("0", "1", ...).
        Values are lists of error locations (indices or coordinates).
        """
        full_syndrome = []

        for i in range(self.num_logical_qubits):
            key = str(i)
            # Get errors for this patch (default to empty list if not present)
            x_errs = X_errors.get(key, [])
            z_errs = Z_errors.get(key, [])

            # Helper to convert to coordinates specific to this patch
            def to_coords(errors):
                coords = set()
                for e in errors:
                    if isinstance(e, int):
                        if e in self.index_mapping:
                            coords.add(self.index_mapping[e][0])
                    elif isinstance(e, tuple):
                        coords.add(e)
                return coords

            x_error_coords = to_coords(x_errs)
            z_error_coords = to_coords(z_errs)
            
            patch_stabs = self.logical_qubits[key]["stab_indices"]
            
            # Filter X and Z stabilizers for this patch
            x_stabs = [s for s in patch_stabs if self.index_mapping[s][1] == "X_stab"]
            z_stabs = [s for s in patch_stabs if self.index_mapping[s][1] == "Z_stab"]
            
            # 1. X Stabilizers (detect Z errors)
            for anc in x_stabs:
                coord, _ = self.index_mapping[anc]
                neighbors = self.get_surrounding_data_qubits(coord)
                parity = 0
                for n_coord in neighbors:
                    if n_coord in z_error_coords:
                        parity += 1
                full_syndrome.append(parity % 2)

            # 2. Z Stabilizers (detect X errors)
            for anc in z_stabs:
                coord, _ = self.index_mapping[anc]
                neighbors = self.get_surrounding_data_qubits(coord)
                parity = 0
                for n_coord in neighbors:
                    if n_coord in x_error_coords:
                        parity += 1
                full_syndrome.append(parity % 2)
                
        return np.array(full_syndrome, dtype=int)

    def visualize(self, show_ancillas: bool = False, show_indices: bool = False,
                  X_errors: Dict[str, List[Union[int, Tuple[float, float]]]] = {}, 
                  Z_errors: Dict[str, List[Union[int, Tuple[float, float]]]] = {}):
        """
        Visualizing the code with some syndrome given X and Z errors. 

        Example usage:
        multi_code = MultiLogicalSurfaceCode(distance=7, num_logical_qubits=2)
        X_errors = {
            "0": [46, 47, 48],
            "1": [144, 145]
        }
        Z_errors = {
            "0": [48],
            "1": [144]
        }
        multi_code.visualize(X_errors=X_errors, Z_errors=Z_errors)
        """
        
        result = self.get_syndrome(X_errors, Z_errors)
        
        sns.set_style("darkgrid")
        mpl.rcParams.update({
            "font.size": 12,
            "grid.color": "0.5",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "xtick.color": "black",
            "ytick.color": "black",
        })
        
        colors = {
            'X_stab': {0: "#79ABD4", 1: "#F07084"},
            'Z_stab': {0: "#385B8B", 1: "#9D212F"}
        }
        
        total_width = self.num_logical_qubits * (self.distance + 1)
        _, ax = plt.subplots(figsize=(total_width, self.distance + 2))
        
        all_x_error_coords = set()
        all_z_error_coords = set()
        
        for _, errs in X_errors.items():
            for e in errs:
                if isinstance(e, int) and e in self.index_mapping:
                    all_x_error_coords.add(self.index_mapping[e][0])
                elif isinstance(e, tuple):
                    all_x_error_coords.add(e)
                    
        for _, errs in Z_errors.items():
            for e in errs:
                if isinstance(e, int) and e in self.index_mapping:
                    all_z_error_coords.add(self.index_mapping[e][0])
                elif isinstance(e, tuple):
                    all_z_error_coords.add(e)

        current_res_idx = 0
        
        def plot_stabilizer_patch(coord, neighbors, qtype, val):
            center_x, center_y = coord
            neighbors = sorted(neighbors, key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x))
            
            if len(neighbors) == 4:
                poly = Polygon(neighbors, closed=True, facecolor=colors[qtype][val], edgecolor='black', alpha=0.9, linewidth=1)
                ax.add_patch(poly)
            elif len(neighbors) == 2:
                # Boundary Stabilizer -> Wedge
                mx = (neighbors[0][0] + neighbors[1][0]) / 2
                my = (neighbors[0][1] + neighbors[1][1]) / 2
                vec_x = mx - center_x
                vec_y = my - center_y
                angle = np.degrees(np.arctan2(vec_y, vec_x))
                pointing_angle = (angle + 180) % 360
                theta1 = pointing_angle - 90
                theta2 = pointing_angle + 90
                
                wedge = Wedge((mx, my), r=0.5, theta1=theta1, theta2=theta2,
                              facecolor=colors[qtype][val], edgecolor='black', alpha=0.9, linewidth=1)
                ax.add_patch(wedge)

        for i in range(self.num_logical_qubits):
            patch_stabs = self.logical_qubits[str(i)]["stab_indices"]
            # Split into X and Z to match get_syndrome order
            x_stabs = [s for s in patch_stabs if self.index_mapping[s][1] == "X_stab"]
            z_stabs = [s for s in patch_stabs if self.index_mapping[s][1] == "Z_stab"]
            
            ordered_stabs = x_stabs + z_stabs
            
            for idx in ordered_stabs:
                if current_res_idx >= len(result): 
                    break
                val = result[current_res_idx]
                current_res_idx += 1
                
                coord, qtype = self.index_mapping[idx]
                neighbors = self.get_surrounding_data_qubits(coord)
                plot_stabilizer_patch(coord, neighbors, qtype, val)
                
                if show_indices:
                    t_color = 'black'
                    ax.text(coord[0], coord[1], str(idx), color=t_color, 
                            ha='center', va='center', fontsize=9, fontweight='bold', zorder=20)

        data_coords = [c for c in self.all_qubits_coords if self.index_mapping[self.inverse_mapping[c]][1] == 'data']
        data_x = [c[0] for c in data_coords]
        data_y = [c[1] for c in data_coords]
        
        data_colors = []
        for c in data_coords:
            has_x = c in all_x_error_coords
            has_z = c in all_z_error_coords
            
            if has_x and has_z:
                data_colors.append('cyan')
            elif has_x:
                data_colors.append('gold')
            elif has_z:
                data_colors.append('palegreen')
            else:
                data_colors.append('#F0F0F0')

        ax.scatter(data_x, data_y, s=200, c=data_colors, edgecolor='black', zorder=10, label='Data Qubit')
        
        if show_indices:
            for c in data_coords:
                idx = self.inverse_mapping[c]
                ax.text(c[0], c[1], str(idx), color='black', 
                        ha='center', va='center', fontsize=8, zorder=21)
        
        if show_ancillas:
            anc_coords = [c for c in self.all_qubits_coords if 'stab' in self.index_mapping[self.inverse_mapping[c]][1]]
            anc_x = [c[0] for c in anc_coords]
            anc_y = [c[1] for c in anc_coords]
            ax.scatter(anc_x, anc_y, s=150, color='silver', marker='o', edgecolor='black', zorder=11)

            if show_indices:
                # Get set of indices already plotted to avoid overlap (active stabs are covered by patches but we might want to see indices)
                # But here we are plotting the background dots.
                for c in anc_coords:
                    idx = self.inverse_mapping[c]
                    # We already plotted indices for active stabilizers on top of patches
                    # This might overlap if we plot again.
                    pass 


        for i in range(self.num_logical_qubits):
            shift_x = i * (self.distance + 1)
            rect_patch = plt.Rectangle( # type: ignore
                (shift_x - 0.8, -0.8), 
                self.distance + 0.6, 
                self.distance + 0.6,
                fill=False,
                edgecolor='purple',
                linewidth=3,
                linestyle='--'
            )
            ax.add_patch(rect_patch)
            ax.text(shift_x + self.distance/2 - 0.5, -1.2, f"L{i}", color='purple', fontsize=14, fontweight='bold')

        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.invert_yaxis()
        
        legend_elements = [
            Patch(facecolor=colors['X_stab'][0], edgecolor='k', label='X-Stab (Ok)'),
            Patch(facecolor=colors['X_stab'][1], edgecolor='k', label='X-Syndrome (Error)'),
            Patch(facecolor=colors['Z_stab'][0], edgecolor='k', label='Z-Stab (Ok)'),
            Patch(facecolor=colors['Z_stab'][1], edgecolor='k', label='Z-Syndrome (Error)'),
            Line2D([0], [0], marker='o', color='w', label='Data Qubit (No Error)',
                          markerfacecolor='#F0F0F0', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='Ancilla Qubit',
                          markerfacecolor='silver', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='X Error (On Data)',
                          markerfacecolor='gold', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='Z Error (On Data)',
                          markerfacecolor='palegreen', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='Y Error (On Data)',
                          markerfacecolor='cyan', markersize=10, markeredgecolor='k'),
            
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.175, 1))
        plt.tight_layout()
        plt.show()

        