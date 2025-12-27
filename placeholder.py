"""
This class will be an extended version of the surface code. 
It will implement one essential additional feature:
the ability to encode multiple logical qubits simulataneously.
This means in one circuit run, we can have several logical qubits.
With this I will be able to test logical operations between multiple qubits,
such as logical CNOTs + Hadamards to create entangled states like Bell pairs.

On these I will then be able to perform MWPM and other sophisticated decoding techniques
such as the correlated decoder.

I need implements for: 
    - all data qubits for the whole lattice as well as for each logical qubit
    - all stabilizers (X and Z) for the entire lattice, as well as for each logical qubit
    - logical operators for each logical qubit
    - methods to perform logical operations (CNOT, Hadamard, etc.) between logical qubits
    - methods to simulate error generation and syndrome extraction
    - methods to apply MWPM and correlated decoding (for now MWPM is sufficient and could 
      be used from old code)
      --> for MWPM I need to extract syndromes from the INDIVIDUAL lattices and create a matching graph
          This is important because this allows for individual logical qubit error correction
          This will then be also useful for the correlated decoder later on!

"""
from surface_code import SurfaceCode
import stim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch, Wedge
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import pymatching as pm
from typing import List, Dict, Any, Tuple, Union, Optional, Literal

class MultiLogicalSurfaceCode:
    def __init__(self, distance: int, num_logical_qubits: int, seed: int = 42):
        self.distance = distance
        self.num_logical_qubits = num_logical_qubits
        self.seed = seed
        self.circuit = stim.Circuit()
        self.quantum_algorithm: Optional[stim.Circuit] = None
        
        # Initialize lattices
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
            
            # Shift coordinates
            shift_x = i * (self.distance + 1)
            
            shifted_data = [(x + shift_x, y) for x, y in data_coords]
            shifted_x_stab = [(x + shift_x, y) for x, y in x_stab_coords]
            shifted_z_stab = [(x + shift_x, y) for x, y in z_stab_coords]
            
            # Collect all qubits for this patch to sort them
            patch_qubits = []
            for coord in shifted_data:
                patch_qubits.append({'coord': coord, 'type': 'data', 'sort_key': (coord[1], coord[0])})
            for coord in shifted_x_stab:
                patch_qubits.append({'coord': coord, 'type': 'X_stab', 'sort_key': (coord[1], coord[0])})
            for coord in shifted_z_stab:
                patch_qubits.append({'coord': coord, 'type': 'Z_stab', 'sort_key': (coord[1], coord[0])})
            
            # Sort primarily by y, then x (same as SurfaceCode)
            patch_qubits.sort(key=lambda q: q['sort_key'])
            
            # Assign indices
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
        
        deltaZ = [(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)]
        deltaX = [(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]
        deltas = deltaZ if qtype == 'Z_stab' else deltaX
        
        for dx, dy in deltas:
            neighbor_coord = (x + dx, y + dy)
            if neighbor_coord in self.inverse_mapping:
                # Check if it is a data qubit
                n_idx = self.inverse_mapping[neighbor_coord]
                if self.index_mapping[n_idx][1] == 'data':
                    surrounding_qubits.append(neighbor_coord) # type: ignore

        return surrounding_qubits

    def loop_body_noisy(self, noise_params: Dict[str, float] = {}) -> stim.Circuit:
        p_init = noise_params.get("p_init", 0.0)
        p_meas = noise_params.get("p_meas", 0.0)
        p_gate1 = noise_params.get("p_gate1", 0.0)
        p_gate2 = noise_params.get("p_gate2", 0.0)
        p_idle = noise_params.get("p_idle", 0.0)

        loop = stim.Circuit()
        
        # Collect all stabilizer indices in order
        all_stab_indices = []
        for i in range(self.num_logical_qubits):
            all_stab_indices.extend(self.logical_qubits[str(i)]["stab_indices"])

        # 1. Initialization
        loop.append("R", all_stab_indices) # type: ignore
        if p_init > 0:
            loop.append("X_ERROR", all_stab_indices, p_init)

        # 2. Syndrome Extraction
        for i in range(self.num_logical_qubits):
            lq = self.logical_qubits[str(i)]
            stabs = lq["stab_indices"]
            
            for ancilla_idx in stabs:
                coord, qtype = self.index_mapping[ancilla_idx]
                neighbors = self.get_surrounding_data_qubits(coord)
                
                if qtype == 'X_stab':
                    loop.append("H", [ancilla_idx]) # type: ignore
                    if p_gate1 > 0:
                        loop.append("DEPOLARIZE1", [ancilla_idx], p_gate1)
                
                for neighbor_coord in neighbors:
                    neighbor_idx = self.inverse_mapping[neighbor_coord]
                    
                    if qtype == 'X_stab':
                        targets = [ancilla_idx, neighbor_idx]
                    elif qtype == 'Z_stab':
                        targets = [neighbor_idx, ancilla_idx]
                    
                    loop.append("CNOT", targets) # type: ignore
                    if p_gate2 > 0:
                        loop.append("DEPOLARIZE2", targets, p_gate2) # type: ignore
                
                if qtype == 'X_stab':
                    loop.append("H", [ancilla_idx]) # type: ignore
                    if p_gate1 > 0:
                        loop.append("DEPOLARIZE1", [ancilla_idx], p_gate1)

        # 3. Measurement
        if p_meas > 0:
            print("here adding errs")
            loop.append("X_ERROR", all_stab_indices, p_meas)
        loop.append("M", all_stab_indices) # type: ignore

        # 4. Idle Error
        if p_idle > 0:
            all_data_indices = []
            for i in range(self.num_logical_qubits):
                all_data_indices.extend(self.logical_qubits[str(i)]["data_indices"])
            loop.append("DEPOLARIZE1", all_data_indices, p_idle)

        loop.append("TICK") # type: ignore
        return loop

    def build_in_stim_noisy(self, rounds: int = 1, logical_basis: Union[List[Literal["X", "Z"]], Literal["X", "Z"]] = "Z", 
                            noise_params: Optional[Dict[str, float]] = None) -> None:
        if noise_params is None:
            noise_params = {}

        if isinstance(logical_basis, str) or len(logical_basis) != self.num_logical_qubits:
            logic_basis = [logical_basis] * self.num_logical_qubits
        elif len(logical_basis) != self.num_logical_qubits:
            raise ValueError("Length of logical_basis list must match num_logical_qubits.")
        else:
            logic_basis = logical_basis

        # Qubit Coords
        for idx, (coord, _) in self.index_mapping.items():
            self.circuit.append("QUBIT_COORDS", [idx], [coord[0], coord[1]])
            
        # Collect all data indices
        all_data_indices = []
        for i in range(self.num_logical_qubits):
            all_data_indices.extend(self.logical_qubits[str(i)]["data_indices"])
            
        # Collect all stab indices for dt calculation
        all_stab_indices = []
        for i in range(self.num_logical_qubits):
            all_stab_indices.extend(self.logical_qubits[str(i)]["stab_indices"])
        dt = len(all_stab_indices)
        
        p_init = noise_params.get("p_init", 0.0)
        p_gate1 = noise_params.get("p_gate1", 0.0)
        p_meas = noise_params.get("p_meas", 0.0)

        # Logical State Prep
        self.circuit.append("R", all_data_indices) # type: ignore
        
        if p_init > 0:
            self.circuit.append("X_ERROR", all_data_indices, p_init)

        for i, basis in enumerate(logic_basis):  # initialize in logical |0> or |+> depending on logic_basis
            lq_data_indices = self.logical_qubits[str(i)]["data_indices"]
            if basis == "X":
                self.circuit.append("H", lq_data_indices) # type: ignore
        
        if p_init > 0:
            self.circuit.append("DEPOLARIZE1", all_data_indices) # type: ignore


        self.circuit.append("TICK") # type: ignore
        
        # Rounds ~ Loop
        loop_body = self.loop_body_noisy(noise_params)
        self.circuit += loop_body
        
        # After initial round -> add the quantum algorithm that you want to simulate
        if self.quantum_algorithm is not None:
            self.circuit += self.quantum_algorithm

        t = dt
        for rnd in range(1, rounds):
            self.circuit.append("SHIFT_COORDS", [], [0, 0, 1])
            self.circuit += loop_body
            
            prev = t - dt
            curr = t
            
            # Add detectors for all stabilizers
            global_stab_idx = 0
            for i in range(self.num_logical_qubits):
                stabs = self.logical_qubits[str(i)]["stab_indices"]
                for anc in stabs:
                    coord, _ = self.index_mapping[anc]
                    lookback_prev = (prev + global_stab_idx) - (t + dt)
                    lookback_curr = (curr + global_stab_idx) - (t + dt)
                    
                    self.circuit.append(
                        "DETECTOR",
                        [stim.target_rec(lookback_prev), stim.target_rec(lookback_curr)],
                        [coord[0], coord[1], rnd]
                    )
                    global_stab_idx += 1
            
            t += dt
            
        for i, basis in enumerate(logic_basis):
            lq_data_indices = self.logical_qubits[str(i)]["data_indices"]
            
            if basis == "X":
                self.circuit.append("H", lq_data_indices) # type: ignore
                if p_gate1 >0:
                    self.circuit.append("DEPOLARIZE1", lq_data_indices) # type: ignore
            if p_meas > 0:
                self.circuit.append("X_ERROR", all_data_indices, p_meas)

            self.circuit.append("M", all_data_indices) # type: ignore
            
        # Observables -- NOTE: Why this so complicated here? Was it to differentiate 
                             # the multiple surface codes for decoding?
        current_data_meas_offset = 0
        for i in range(self.num_logical_qubits):
            lq_data_indices = self.logical_qubits[str(i)]["data_indices"]
            num_lq_data = len(lq_data_indices)
            
            targets = []
            for k in range(num_lq_data):
                rec_idx = (current_data_meas_offset + k) - len(all_data_indices)
                targets.append(stim.target_rec(rec_idx))
            
            self.circuit.append("OBSERVABLE_INCLUDE", targets, i)
            current_data_meas_offset += num_lq_data

    def visualize_results(self, result: np.ndarray, show_ancillas: bool = False):
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
                # Boundary Stabilizer -> Wedge
                # Midpoint of neighbors is the center of the wedge
                mx = (neighbors[0][0] + neighbors[1][0]) / 2
                my = (neighbors[0][1] + neighbors[1][1]) / 2
                
                # Vector from Ancilla (coord) to Midpoint
                vec_x = mx - center_x
                vec_y = my - center_y
                
                # Angle of this vector
                angle = np.degrees(np.arctan2(vec_y, vec_x))
                
                # We want the wedge to point FROM midpoint TO ancilla (outwards)
                # So we flip the angle by 180 degrees
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
            
        data_x = [c[0] for c in self.all_qubits_coords if self.index_mapping[self.inverse_mapping[c]][1] == 'data']
        data_y = [c[1] for c in self.all_qubits_coords if self.index_mapping[self.inverse_mapping[c]][1] == 'data']
        ax.scatter(data_x, data_y, s=100, color='#F0F0F0', edgecolor='black', zorder=10, label='Data Qubit')
        
        if show_ancillas:
            anc_x = [c[0] for c in self.all_qubits_coords if 'stab' in self.index_mapping[self.inverse_mapping[c]][1]]
            anc_y = [c[1] for c in self.all_qubits_coords if 'stab' in self.index_mapping[self.inverse_mapping[c]][1]]
            ax.scatter(anc_x, anc_y, s=50, color='gray', marker='s', zorder=11)

        for i in range(self.num_logical_qubits):
            shift_x = i * (self.distance + 1)
            rect_patch = plt.Rectangle(
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
        ax.set_title(f"Multi-Logical Qubit Surface Code (d={self.distance}, N={self.num_logical_qubits})")
        plt.show()

    def run_with_pymatching(self, shots: int = 1000):
        model = self.circuit.detector_error_model(decompose_errors=True)
        matching = pm.Matching.from_detector_error_model(model)
        sampler = self.circuit.compile_detector_sampler()
        
        syndrome, obs = sampler.sample(
            shots=shots,
            separate_observables=True
        )
        
        preds = matching.decode_batch(syndrome)
        print(np.array(preds).shape, preds)
        errors = np.any(preds != obs, axis=1)
        logical_error_rate = np.mean(errors)
        
        return logical_error_rate

    def run_simulation(self, shots: int = 1000) -> np.ndarray:
        sampler = self.circuit.compile_detector_sampler()
        results:np.ndarray = sampler.sample(shots=shots) # type: ignore
        return results

    def diagram(self) -> None:
        print(self.circuit.diagram())

    def apply_logical_gate(self, gate: str, control: int, target: int, noise_params: Dict[str, float] = {}) -> None:

        if self.quantum_algorithm is None:
            self.quantum_algorithm = stim.Circuit()
        
        p_gate2 = noise_params.get("p_gate2", 0.0)

        # should be able to perform logical gates, e.g. CNOT between logical qubits
        if gate == "CNOT":
            if control >= self.num_logical_qubits or target >= self.num_logical_qubits:
                raise ValueError("Control or target logical qubit index out of range.")

            control_data_indices = self.logical_qubits[str(control)]["data_indices"]
            target_data_indices = self.logical_qubits[str(target)]["data_indices"]
            for c_idx, t_idx in zip(control_data_indices, target_data_indices):
                self.quantum_algorithm.append("CNOT", [c_idx, t_idx]) # type: ignore
                if p_gate2 > 0:
                    self.quantum_algorithm.append("DEPOLARIZE2", [c_idx, t_idx], p_gate2) # type: ignore
        else:
            raise NotImplementedError(f"Logical gate {gate} not implemented.")

        pass

    # TODO:
    """
    Verify correctness:
    Initialize both in Z basis (0) and add a logical X error on one logical qubit.
    After performing a logical CNOT, measure both in Z basis and verify that
    the error has been mapped correctly to the second logical qubit.

    Do this with some qubits or more errors and see if syndromes match expectations.
    Just go through the indices again and see if everything is correct.
    Cross-check with single logical qubit surface code!!!

    Check state initialization of states as well:
    https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-022-04566-8/MediaObjects/41586_2022_4566_MOESM1_ESM.pdf
    this is example for distance 3 surface code with one logical qubit
    if I have verified this then I obviously can extend this to multiple and larger surface codes.
    
    ---
    verify that stabilizer measurements work correctly : Also that error mapping is correct.
    for that initialized and add some errors and see if syndromes are correct.
    e.g. only one logical qubit (which should be control) will have errors and syndromes
    this should be visible and mapped onto the second logical qubit (target) after CNOT.
    ---

    Implement correlated decoder that can handle multiple logical qubits.
    """