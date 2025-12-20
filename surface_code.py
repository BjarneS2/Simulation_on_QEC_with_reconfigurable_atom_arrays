"""
I want to create a class that represents a surface code for qec.
I want to use the Surface Code class to initialize a surface code of a given distance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge, Patch
from typing import List, Literal, Tuple, Dict, Any, Union, Optional
import matplotlib as mpl
import seaborn as sns
import pymatching as pm
import stim
from copy import deepcopy 



INSTRUCTION_KEYS = ["operation", "operation_error", "initialization_error", "measurement_error", "idle_error"]
GATE_TYPES = ["H", "CNOT", "X", "Y", "Z"]
QubitSpec = Union[List[Union[Tuple[int|float, int|float], int]], List[Union[Tuple[Tuple[int|float, int|float], Tuple[int|float, int|float]], Tuple[float, float]]], str]
InstructionDict = Dict[str, Any]

class SurfaceCode:

    def __init__(self, distance: int, seed : int = 42):
        if distance < 0 or distance % 1 != 0 :
            raise ValueError("Distance must be an odd positive integer.")

        self.d = distance
        self.number_of_qubits = distance ** 2
        # qubit coordinates is the list of data qubits and stabilisers coordinates the list of ancillas
        self.qubit_coords, self.x_stabilisers_coords, self.z_stabilisers_coords = self.create(distance)
        self.stabilisers_coords = self.x_stabilisers_coords + self.z_stabilisers_coords
        self.index_mapping = self.get_index_mapping()
        self.inverse_mapping = self.get_inverse_mapping()
        self.seed = seed
        self.circuit = stim.Circuit()
        
        stab_indices: List[int] = self.get_stabilisers(_as="idx") # type: ignore
        self.stab_indices = sorted(  # otherwise Z and X stabilizers are measured in random order!
            stab_indices,
            key=lambda anc: (
                self.index_mapping[anc][1],        # 'X_stab' or 'Z_stab'
                self.index_mapping[anc][0][0],     # x-coordinate
                self.index_mapping[anc][0][1],     # y-coordinate
            )
        )
        np.random.seed(self.seed)

    def get_data_qubits(self, _as: Literal["coord", "idx"] = "coord") -> List[Tuple[int, int]]|List[int]:
        """Returns the data qubits coordinates or indices based on the _as parameter."""
        if _as == "coord":
            return deepcopy(self.qubit_coords)
        elif _as == "idx":
            data_qubit_indices = []
            for idx, (coord, qtype) in self.index_mapping.items():
                if qtype == "data":
                    data_qubit_indices.append(idx)
            return data_qubit_indices
        else:
            raise ValueError("Invalid value for _as. Use 'coord' or 'idx'.")
    
    def get_stabilisers(self, _as: Literal["coord", "idx"] = "coord") -> List[Tuple[float, float]]|List[int]:
        """Returns the stabilisers coordinates or indices based on the _as parameter."""
        if _as == "coord":
            return deepcopy(self.stabilisers_coords)
        elif _as == "idx":
            stabiliser_indices = []
            for idx, (coord, qtype) in self.index_mapping.items():
                if qtype in ["X_stab", "Z_stab"]:
                    stabiliser_indices.append(idx)
            return stabiliser_indices
        else:
            raise ValueError("Invalid value for _as. Use 'coord' or 'idx'.")
        
    def get_all_qubits(self, _as: Literal["coord", "idx"] = "coord") -> List[Tuple[Union[int, float], Union[int, float]]]|List[int]:
        if _as == "coord":
            return list(self.qubit_coords + self.stabilisers_coords)
        elif _as == "idx":
            return list(self.index_mapping.keys())
        else:
            raise ValueError("Invalid value for _as. Use 'coord' or 'idx'.")
        
    def get_seed(self) -> int:
        return self.seed
    
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
        
        """
    # ====================================================================================================== #
        From stim/src/... and paper: https://arxiv.org/pdf/2409.14765
            // Define interaction orders so that hook errors run against the error grain instead of with it.
            std::vector<surface_coord> z_order{
                {1, 1},
                {1, -1},
                {-1, 1},
                {-1, -1},
            };
            std::vector<surface_coord> x_order{
                {1, 1},
                {-1, 1},
                {1, -1},
                {-1, -1},
            };
        """
        
        surrounding_qubits: List[Tuple[int, int]] = []
        x, y = stab_coord
        deltaZ = [(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)]
        deltaX = [(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]
        deltas = deltaZ if stab_coord in self.z_stabilisers_coords else deltaX
        for dx, dy in deltas:
            neighbor_coord: Tuple[int, int] = ((x + dx), (y + dy)) # type: ignore

            if neighbor_coord in self.qubit_coords:
                surrounding_qubits.append(neighbor_coord)

        return surrounding_qubits

    def plot(self):
        sns.set_style("dark")
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

        # Surface code plotting, to see if the structure is correct
        plt.figure(figsize=(self.d, self.d))
        
        data_x = [coord[0] for coord in self.qubit_coords]
        data_y = [coord[1] for coord in self.qubit_coords]
        plt.scatter(data_x, data_y, color='blue', label='Data Qubits', s=65)
        
        x_stab_x = [coord[0] for coord in self.x_stabilisers_coords]
        x_stab_y = [coord[1] for coord in self.x_stabilisers_coords]
        
        z_stab_x = [coord[0] for coord in self.z_stabilisers_coords]
        z_stab_y = [coord[1] for coord in self.z_stabilisers_coords]

        plt.scatter(x_stab_x, x_stab_y, color='red', marker='s', label='X Stabilizers', s=65)
        plt.scatter(z_stab_x, z_stab_y, color='green', marker='s', label='Z Stabilizers', s=65)

        """ Create a number indexing for each qubit and add the number 
            on the markers of the plot in the respective positions 
        """
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

    def loop_body(self, meas_errors = 0.0) -> stim.Circuit:
            """ Stabilizer Measurement Loop Body """
            loop_body = stim.Circuit()
            # Ancillas in Z basis
            loop_body.append("R", self.stab_indices)  # type: ignore

            for ancilla_idx in self.stab_indices:
                coord, qtype = self.index_mapping[ancilla_idx]
                neighbors = self.get_surrounding_data_qubits(coord) # type: ignore
                
                if qtype == 'X_stab':
                    # Measuring X: H(Anc) -> CNOT(Anc, Data) -> H(Anc)
                    loop_body.append("H", [ancilla_idx])  # type: ignore

                for neighbor_coord in neighbors:
                    neighbor_idx = self.inverse_mapping[neighbor_coord]
                    
                    if qtype == 'X_stab':
                        # Measuring X: H(Anc) -> CNOT(Anc, Data) -> H(Anc)
                        loop_body.append("CNOT", [ancilla_idx, neighbor_idx])  # type: ignore

                    elif qtype == 'Z_stab':
                        # Measuring Z: CNOT(Data, Anc)
                        loop_body.append("CNOT", [neighbor_idx, ancilla_idx])  # type: ignore

                if qtype == 'X_stab':
                    loop_body.append("H", [ancilla_idx])  # type: ignore
                
            # Add measurement errors in form of depolarizing error on ancillas
            # if meas_errors > 0.0:
            #     loop_body.append("X_ERROR", self.stab_indices, meas_errors)  # type: ignore

            loop_body.append("M", self.stab_indices)  # type: ignore
            loop_body.append("TICK")  # type: ignore
            return loop_body

    def loop_body_noisy(self, noise_params: Dict[str, float] = {}) -> stim.Circuit:
            """
            Stabilizer Measurement Loop Body with granular error injection.
            noise_params keys: 'p_init', 'p_meas', 'p_gate1', 'p_gate2', 'p_idle'
            """
            # Default to 0.0 if not specified
            p_init = noise_params.get("p_init", 0.0)
            p_meas = noise_params.get("p_meas", 0.0)
            p_gate1 = noise_params.get("p_gate1", 0.0)
            p_gate2 = noise_params.get("p_gate2", 0.0)
            p_idle = noise_params.get("p_idle", 0.0)

            loop = stim.Circuit()

            # 1. Initialization (Reset Ancillas)
            loop.append("R", self.stab_indices) # type: ignore
            if p_init > 0:
                loop.append("X_ERROR", self.stab_indices, p_init) # Flip error after reset

            # 2. Syndrome Extraction
            for ancilla_idx in self.stab_indices:
                coord, qtype = self.index_mapping[ancilla_idx]
                neighbors = self.get_surrounding_data_qubits(coord)
                
                # X-Stabilizers need Basis Change (Hadamard)
                if qtype == 'X_stab':
                    loop.append("H", [ancilla_idx]) # type: ignore
                    if p_gate1 > 0: 
                        loop.append("DEPOLARIZE1", [ancilla_idx], p_gate1)

                # CNOT sequence
                for neighbor_coord in neighbors:
                    neighbor_idx = self.inverse_mapping[neighbor_coord]
                    
                    # Determine Control/Target based on Stabilizer type
                    if qtype == 'X_stab':
                        targets = [ancilla_idx, neighbor_idx] # Ancilla is Control
                    elif qtype == 'Z_stab':
                        targets = [neighbor_idx, ancilla_idx] # Data is Control
                    
                    loop.append("CNOT", targets)  # type: ignore
                    if p_gate2 > 0:
                        loop.append("DEPOLARIZE2", targets, p_gate2) # type: ignore

                # Basis Change back for X-Stabilizers
                if qtype == 'X_stab':
                    loop.append("H", [ancilla_idx])  # type: ignore
                    if p_gate1 > 0: 
                        loop.append("DEPOLARIZE1", [ancilla_idx], p_gate1)

            # 3. Measurement
            # Apply measurement error (probability of flipping the result)
            if p_meas > 0:
                loop.append("X_ERROR", self.stab_indices, p_meas)
            loop.append("M", self.stab_indices) # type: ignore

            # 4. Physical Idle Error
            # Data qubits sit idle while ancillas are measured/reset. 
            # We apply decoherence (Identity error) to all DATA qubits here.
            if p_idle > 0:
                data_qubits = self.get_data_qubits(_as="idx")
                loop.append("DEPOLARIZE1", data_qubits, p_idle) # type: ignore

            loop.append("TICK") # type: ignore
            return loop

    def build_in_stim(self, rounds: int = 1, logical_basis: Literal["X", "Z"] = "Z", depolarize_prob: float = 0.0) -> None:

        for idx, (coord, _) in self.index_mapping.items():
            self.circuit.append("QUBIT_COORDS", [idx], [coord[0], coord[1]])

        data_indices = [k for k, v in self.index_mapping.items() if v[1] == 'data']

        dt = len(self.stab_indices)
        if dt == 0:
            raise RuntimeError("No stabilizers found in the surface code.")

        # |0>L or |+>L
        if logical_basis == "Z":
            self.circuit.append("R", data_indices) # type: ignore
        elif logical_basis == "X":
            self.circuit.append("R", data_indices) # type: ignore
            self.circuit.append("H", data_indices) # type: ignore
        else:
            raise ValueError("logical_basis must be 'Z' or 'X'")

        self.circuit.append("TICK") # type: ignore 
        
        loop_body = self.loop_body()
        # One-shot stabilizer projection
        self.circuit += loop_body
        
        t = dt # "time index"

        
        for rnd in range(1, rounds):
            # Shift coordinates 
            self.circuit.append("SHIFT_COORDS", [], [0, 0, 1])  # type: ignore

            if depolarize_prob > 0.0:
                self.circuit.append("DEPOLARIZE1", data_indices, depolarize_prob)  # type: ignore
                # self.circuit.append("X_ERROR", data_indices, depolarize_prob)  # type: ignore
                # self.circuit.append("Y_ERROR", data_indices, depolarize_prob)  # type: ignore
                # self.circuit.append("Z_ERROR", data_indices, depolarize_prob)  # type: ignore
                
            self.circuit += self.loop_body(meas_errors=depolarize_prob/10)

            prev = t - dt
            curr = t
            for j, anc in enumerate(self.stab_indices):
                coord, _ = self.index_mapping[anc]
                # target the time j produced by the last M operation.

                lookback_prev = (prev+j) - (t+dt)# - (dt-j) # (prev+j) - (t+dt) # look back to prev
                lookback_curr = (curr+j) - (t+dt) # - (2*dt-j) # (curr+j) - (t+dt) # look back to curr

                self.circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(lookback_prev), # (prev + j) | stim.target_record_bit, 
                        stim.target_rec(lookback_curr), # (curr + j) | stim.target_record_bit
                    ],
                    [coord[0], coord[1], rnd]
                )

            t +=  dt

        if logical_basis == "Z":
            self.circuit.append("M", data_indices) # type: ignore
        else:
            # measure X by rotating to Z then measuring
            self.circuit.append("H", data_indices) # type: ignore
            self.circuit.append("M", data_indices) # type: ignore
            
        logical_targets = [stim.target_rec(-k) for k in range(1, len(data_indices)+1)]
        self.circuit.append("OBSERVABLE_INCLUDE", logical_targets, 0)

    def build_in_stim_noisy(self, rounds: int = 1, logical_basis: Literal["X", "Z"] = "Z", 
                        noise_params: Optional[Dict[str, float]] = None) -> None:
            
            if noise_params is None:
                noise_params = {}

            # --- Qubit Coordinates Setup (Same as before) ---
            for idx, (coord, _) in self.index_mapping.items():
                self.circuit.append("QUBIT_COORDS", [idx], [coord[0], coord[1]])

            data_indices = self.get_data_qubits(_as="idx")
            dt = len(self.stab_indices)

            # --- Logical State Prep ---
            # Note: We apply errors to these initial operations too if desired
            p_init = noise_params.get("p_init", 0.0)
            p_gate1 = noise_params.get("p_gate1", 0.0)

            self.circuit.append("R", data_indices)  # type: ignore
            if p_init > 0: 
                self.circuit.append("X_ERROR", data_indices, p_init)  # type: ignore

            if logical_basis == "X":
                self.circuit.append("H", data_indices)  # type: ignore
                if p_gate1 > 0: 
                    self.circuit.append("DEPOLARIZE1", data_indices, p_gate1)  # type: ignore
            
            self.circuit.append("TICK")  # type: ignore
            
            # --- Rounds Loop ---
            # Generate the loop body ONCE with noise parameters baked in
            loop_body_circuit = self.loop_body_noisy(noise_params)
            
            # Add initial round
            self.circuit += loop_body_circuit

            t = dt 
            
            # Subsequent rounds
            for rnd in range(1, rounds):
                self.circuit.append("SHIFT_COORDS", [], [0, 0, 1])
                self.circuit += loop_body_circuit

                # Detector declaration (Same as before)
                prev = t - dt
                curr = t
                for j, anc in enumerate(self.stab_indices):
                    coord, _ = self.index_mapping[anc]
                    lookback_prev = (prev+j) - (t+dt)
                    lookback_curr = (curr+j) - (t+dt)
                    self.circuit.append(
                        "DETECTOR",
                        [stim.target_rec(lookback_prev), stim.target_rec(lookback_curr)],
                        [coord[0], coord[1], rnd]
                    )
                t += dt

            # --- Logical Measurement ---
            p_meas = noise_params.get("p_meas", 0.0)
            
            if logical_basis == "X":
                self.circuit.append("H", data_indices)  # type: ignore
                if p_gate1 > 0: 
                    self.circuit.append("DEPOLARIZE1", data_indices, p_gate1)  # type: ignore

            if p_meas > 0:
                self.circuit.append("X_ERROR", data_indices, p_meas)  # type: ignore
                
            self.circuit.append("M", data_indices)  # type: ignore
            
            logical_targets = [stim.target_rec(-k) for k in range(1, len(data_indices)+1)]
            self.circuit.append("OBSERVABLE_INCLUDE", logical_targets, 0)

    def build_in_stim_noisy2(self, rounds: int = 1, logical_basis: Literal["X", "Z"] = "Z", 
                                noise_params: Optional[Dict[str, float]] = None) -> None:
            
            if noise_params is None: noise_params = {}
            
            # [Coordinates setup ... same as before]
            for idx, (coord, _) in self.index_mapping.items():
                self.circuit.append("QUBIT_COORDS", [idx], [coord[0], coord[1]])

            data_indices = self.get_data_qubits(_as="idx")
            # Map data qubit indices to their position in the measurement array for easier lookup later
            data_idx_to_meas_offset = {idx: i for i, idx in enumerate(data_indices)}
            
            dt = len(self.stab_indices)

            # [Logical State Prep ... same as before]
            p_init = noise_params.get("p_init", 0.0)
            p_gate1 = noise_params.get("p_gate1", 0.0)
            self.circuit.append("R", data_indices)
            if p_init > 0: self.circuit.append("X_ERROR", data_indices, p_init)
            if logical_basis == "X":
                self.circuit.append("H", data_indices)
                if p_gate1 > 0: self.circuit.append("DEPOLARIZE1", data_indices, p_gate1)
            self.circuit.append("TICK")
            
            # [Rounds Loop ... same as before]
            loop_body_circuit = self.loop_body_noisy(noise_params)
            self.circuit += loop_body_circuit # Round 0
            t = dt 
            
            for rnd in range(1, rounds):
                self.circuit.append("SHIFT_COORDS", [], [0, 0, 1])
                self.circuit += loop_body_circuit
                
                # Bulk Detectors (Comparing Ancilla t vs Ancilla t-1)
                prev = t - dt
                curr = t
                for j, anc in enumerate(self.stab_indices):
                    coord, _ = self.index_mapping[anc]
                    lookback_prev = (prev+j) - (t+dt)
                    lookback_curr = (curr+j) - (t+dt)
                    self.circuit.append("DETECTOR",
                        [stim.target_rec(lookback_prev), stim.target_rec(lookback_curr)],
                        [coord[0], coord[1], rnd]
                    )
                t += dt

            # --- LOGICAL MEASUREMENT & FINAL BOUNDARY ---
            p_meas = noise_params.get("p_meas", 0.0)
            
            # 1. Measure Data Qubits
            if logical_basis == "X":
                self.circuit.append("H", data_indices)
                if p_gate1 > 0: self.circuit.append("DEPOLARIZE1", data_indices, p_gate1)

            if p_meas > 0:
                self.circuit.append("X_ERROR", data_indices, p_meas)
                
            self.circuit.append("M", data_indices)

            # 2. Add "End of Time" Detectors
            # We need to compare the LAST ancilla measurement to the stabilizers 
            # reconstructed from the data measurements.
            
            # Number of measurements in the final block (all data qubits)
            num_data_meas = len(data_indices)
            
            # Determine which stabilizer type we can reconstruct (Z-stabs if measured in Z, X-stabs if X)
            relevant_stab_type = "Z_stab" if logical_basis == "Z" else "X_stab"

            for j, anc in enumerate(self.stab_indices):
                coord, qtype = self.index_mapping[anc]
                
                # Only add detectors for the stabilizers compatible with the measurement basis
                if qtype == relevant_stab_type:
                    
                    # Get the record index of the last ancilla measurement for this stabilizer
                    # The ancilla measurements happened before the data measurements
                    # So we look back past the data measurements (num_data_meas) to the ancilla block
                    ancilla_lookback = j - dt - num_data_meas 
                    
                    # Construct the "pseudo-stabilizer" from data measurements
                    # This involves XORing the measurements of the data qubits involved in this stabilizer
                    neighbors = self.get_surrounding_data_qubits(coord)
                    rec_targets = [stim.target_rec(ancilla_lookback)]
                    
                    for n_coord in neighbors:
                        n_idx = self.inverse_mapping[n_coord]
                        # Find where this data qubit is in the measurement record (0 to -num_data_meas)
                        meas_offset = data_idx_to_meas_offset[n_idx] - num_data_meas
                        rec_targets.append(stim.target_rec(meas_offset))
                    
                    self.circuit.append("DETECTOR", rec_targets, [coord[0], coord[1], rounds])

            # 3. Logical Observable (Same as before)
            logical_targets = [stim.target_rec(-k) for k in range(1, len(data_indices)+1)]
            self.circuit.append("OBSERVABLE_INCLUDE", logical_targets, 0)

    def run_with_pymatching(self, shots: int = 1000):
        model = self.circuit.detector_error_model(decompose_errors=True)
        matching = pm.Matching.from_detector_error_model(model)
        sampler = self.circuit.compile_detector_sampler()

        syndrome, obs = sampler.sample(
            shots=shots,
            separate_observables=True
        )

        preds = matching.decode_batch(syndrome)

        # logical error happens when observable prediction != actual
        errors = np.any(preds != obs, axis=1)
        logical_error_rate = np.mean(errors)

        return logical_error_rate, syndrome, obs, preds

    def run_simulation(self, circuit: Optional[stim.Circuit] = None, shots: int = 1000) -> np.ndarray:
        if circuit is not None:
            sampler = circuit.compile_sampler()
        else:
            sampler = self.circuit.compile_sampler()
        results = sampler.sample(shots=shots)
        return results
 
    def diagram(self, *args, **kwargs) -> None:
        """Prints a text diagram of the circuit."""
        print(self.circuit.diagram(*args, **kwargs))

    def visualize_results(self, result: np.ndarray, show_ancillas: bool = False) -> None:
        """ 
        adapted from github/jfoliveira/surfq
        Visualize the measurement results on the surface code.
        Plots data qubits as circles and colors the plaquettes (stabilizers)
        based on the measurement outcome (Syndrome vs Normal).
        results: Array of measurement (list of bools or 0/1) for all stabilizers.
        show_ancillas: If True, shows ancilla qubits as squares on the plot.
        """
        
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
        
        colors = {
            'X_stab': {0: "#79ABD4", 1: "#F07084"},  # Muted Blue vs Muted Red
            'Z_stab': {0: "#385B8B", 1: "#9D212F"}   # Darker Blue vs Lighter Red
        }

        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Split into X and Z based on the correct ordering
        x_indices = [anc for anc in self.stab_indices if self.index_mapping[anc][1] == 'X_stab']
        z_indices = [anc for anc in self.stab_indices if self.index_mapping[anc][1] == 'Z_stab']
        # Now match the result array exactly
        x_results = result[:len(x_indices)]
        z_results = result[len(x_indices):]

        if len(result) != (len(x_indices) + len(z_indices)):
            raise ValueError(f"Result length {len(result)} does not match number of stabilizers.")

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
                # Boundary Stabilizer -> wedge (half circle)
                shift_x = 0
                shift_y = 0
                L = self.d 
                
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
                # Vector from original center (cx, cy) to midpoint of neighbors (mx, my)
                mx, my = (neighbors[0][0] + neighbors[1][0])/2, (neighbors[0][1] + neighbors[1][1])/2
                vec_x, vec_y = mx - cx, my - cy
                pointing_angle = np.degrees(np.arctan2(vec_y, vec_x))
                pointing_angle = (pointing_angle + 180) % 360
                theta1 = pointing_angle - 90
                theta2 = pointing_angle + 90
                
                wedge = Wedge((shifted_cx, shifted_cy), r=0.5, theta1=theta1, theta2=theta2, 
                              facecolor=color, edgecolor='black', alpha=0.9, linewidth=1)
                ax.add_patch(wedge)

        for i, idx in enumerate(x_indices):
            coord, qtype = self.index_mapping[idx]
            neighbors = self.get_surrounding_data_qubits(coord)
            plot_stabilizer_patch(coord, neighbors, qtype, x_results[i])

        for i, idx in enumerate(z_indices):
            coord, qtype = self.index_mapping[idx]
            neighbors = self.get_surrounding_data_qubits(coord)
            plot_stabilizer_patch(coord, neighbors, qtype, z_results[i])

        data_x = [c[0] for c in self.qubit_coords]
        data_y = [c[1] for c in self.qubit_coords]
        ax.scatter(data_x, data_y, s=200, color='#F0F0F0', edgecolor='black', zorder=10, label='Data Qubit')

        if show_ancillas:
            x_stab_x = [coord[0] for coord in self.x_stabilisers_coords]
            x_stab_y = [coord[1] for coord in self.x_stabilisers_coords]
            z_stab_x = [coord[0] for coord in self.z_stabilisers_coords]
            z_stab_y = [coord[1] for coord in self.z_stabilisers_coords]

            ax.scatter(x_stab_x, x_stab_y, marker='s', s=100, color='tab:orange', label='X Ancilla', zorder=11)
            ax.scatter(z_stab_x, z_stab_y, marker='s', s=100, color='tab:orange', label='Z Ancilla', zorder=11)

        ax.set_aspect('equal')
        ax.set_xticks(range(self.d))
        ax.set_yticks(range(self.d))
        ax.set_xlim(-1, self.d)
        ax.set_ylim(-1, self.d)
        ax.set_title(f"Syndrome Measurement (d={self.d})")
        
        legend_elements = [
            Patch(facecolor=colors['X_stab'][0], edgecolor='k', label='X-Stab (Ok)'),
            Patch(facecolor=colors['X_stab'][1], edgecolor='k', label='X-Syndrome (Error)'),
            Patch(facecolor=colors['Z_stab'][0], edgecolor='k', label='Z-Stab (Ok)'),
            Patch(facecolor=colors['Z_stab'][1], edgecolor='k', label='Z-Syndrome (Error)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.tight_layout()
        plt.show()
        
    def is_valid_coordinate(self, coord: Tuple[int|float, int|float]) -> bool:
        """ Check if the given coordinate is valid within the surface code. """
        x, y = coord
        if not (isinstance(x, int) and isinstance(y, int) or isinstance(x, float) and isinstance(y, float)):
            return False
        # Now I need to check if the coordinates are in the inverse mapping items
        return (coord in self.inverse_mapping.keys())
    
    def is_valid_index(self, index: int) -> bool:
        """ Check if the given index is valid within the surface code. """
        return index in self.index_mapping.keys()
        
    def validate_qubit_spec(self, qubit_spec: Union[Tuple[int, int], int, Tuple[Tuple[int|float, int|float]]], operation: str) -> bool:
        """ Validate if the qubit specification is valid for the given operation. 
            qubit spec can be:
            1. single qubit coordinate: (x, y)
            2. single qubit index: int
            3. CNOT qubit coordinates: ((control_x, control_y), (target_x, target_y))
            4. CNOT qubit indices: (control_index, target_index)
            indices are integers, while coordinates are tuples of two integers, or floats."""

        if operation in ['H', 'X', 'Y', 'Z']:
            # Check for single qubit operations

            # if qubit_spec is given as coordinates (x,y) <- can be int or float
            if isinstance(qubit_spec, tuple) and len(qubit_spec) == 2 and all(isinstance(coord, int) for coord in qubit_spec):
                return self.is_valid_coordinate(qubit_spec)
            
            # if qubit_spec is given as index int
            elif isinstance(qubit_spec, int):
                return self.is_valid_index(qubit_spec)
            
            else:
                print("Warning: Invalid qubit specification {qubits} for operation {operation}. Skipping this instruction.")
            
        elif operation == 'CNOT':
            # Check for two qubit operations (CNOT)

            # if qubit_spec is given as coordinates ((control_x, control_y), (target_x, target_y))
            if (isinstance(qubit_spec, tuple) and len(qubit_spec) == 2 and
                all(isinstance(part, tuple) and len(part) == 2 and all(isinstance(coord, int) for coord in part) for part in qubit_spec)):
                return self.is_valid_coordinate(qubit_spec[0]) and self.is_valid_coordinate(qubit_spec[1])  # type: ignore
            
            # if qubit_spec is given as indices (control_index, target_index)
            elif (isinstance(qubit_spec, tuple) and len(qubit_spec) == 2 and
                all(isinstance(part, int) for part in qubit_spec)):
                return self.is_valid_index(qubit_spec[0]) and self.is_valid_index(qubit_spec[1])
            
            else: 
                print("Warning: Invalid qubit specification {qubits} for operation {operation}. Skipping this instruction.")

        return False

    def parse_instructions_new(self, instructions: List[Dict[str, Any]]):
            """
            Used Like this:
            SurfaceCode.parse_instructions([
                {'operation': 'H', 'qubits': [(1,2), (3,4)]},
                {'operation': 'CNOT', 'qubits': [((1,2), (3,4)), ((5,6), (7,8))]},
                {'operation': 'X', 'qubits': [(2,3)]},
                {'operation': 'Z', 'qubits': [(4,5)]},
            ])
            Firstly just saves a set of instructions to be applied to the circuit.
            And saves it under self.intructions.
            If it is called multiple times it appends to the existing instructions.

            e.g.:
            sc = SurfaceCode(distance=3)
            sc.parse_instructions([...])
            sc.parse_instructions([...])

            It is also supposed to handle errors in the future - the form of:
            sc.parse_instructions([
                {'operation_error': 'X', 'qubits': [(2,3)], 'probability': 0.01},
                {'operation_error': 'Z', 'qubits': [(4,5)], 'probability': 0.02},
                {...},
                {'idle_error': ...}, # potentially adding this
                {initialization_error': 'X', 'qubits': [(1,1)], 'probability': 0.05},
                {...},
                {'measurement_error': 'Z', 'qubits': [(3,3)], 'probability': 0.03},
                {...},
            ])
            
            This should make it possible to insert errors in the circuit after
            initialization, before measurement, and during operations. I also want
            to be able to make it recognize if error values are given multiple times
            for the same qubit and operation, in which case it is overwritten with the
            last given value. I also want to make sure that the probabilities are 
            between 0 and 1. if the instructions are given as 
            [{'what instruction': 'type of gate', 'qubits': list|str, probability: float},...]
            then the instructions should be literals: 
                    - operation, operation_error, measurement_error, initialization_error
            for the type of gate it should be literals:
                    - H, CNOT, X, Y, Z
            for the qubits it should be either a list of coordinates:
                    - [(x1,y1), (x2,y2), ...]
                or in case of CNOT a list of tuples:
                    - [((control_x, control_y), (target_x, target_y)), ...]
                or just the index instead of coordinates:
                    - [index1, index2, ...]
                respectively for CNOT:
                    - [(control_index, target_index), ...]
                or even a string 'all' to apply to all qubits of that type.

            With the functions/methods defined above I should be able to retrieve
            the indices of the qubits from their coordinates and vice versa.
            To get the list of all qubits I can use get_all_qubits(_as="idx"|"coord")

            should print a warning if error instruction has overridden a previous one.
            should print a warning if probability is out of bounds [0, 1] and skip it.
            should print a warning if qubit specification (out of surface code) is invalid and skip that instruction.
            print("Warning: Overriding previous instruction for qubit {qubit} and operation {operation}.\n"
                + "Before: {old_probability}, After: {new_probability}")
            print("Warning: Probability {probability} for qubit {qubit} and operation {operation} is out of bounds [0, 1]. Clipping to valid range."))
            print("Warning: Invalid qubit specification {qubits} for operation {operation}. Skipping this instruction.")

            
            Of course I want to enable to apply these instructions one after another
            So if I wanna do e.g.: H as logical gate (all data qubits), then apply errors,
            then do CNOTs between data and ancillas, then apply more errors, 
            then measure ancillas with potential measurement errors, I should be able to do so.
            So the instructions should be stored in a list of dictionaries, where each dictionary
            represents a single instruction or errors at a specific point in the circuit. 
            
            I should maybe also think of a way to have X,Z,... errors applied every time after 
            a certain operation, e.g. after every H, after every CNOT, ... Maybe I can do this
            by having a special key in the dictionary like 'after_operation': 'H' and then the error
            is applied after every H operation in the circuit. This would require parsing the circuit
            and inserting the errors at the right places. This could be done in the build_in_stim method.

            0. Always reset all bits. 
            1. go through the instructions list and apply the operations/errors.
            2. do rounds of stabilizer measurements.
            3. measure all ancillas at the end.
            4. return the circuit.
            5. run the simulation and get the results.
            6. visualize the results on the surface code.
            For the full QEC cycle I would need to include as well:
            (7. apply decoding algorithm to correct errors.
            8. visualize the corrected results on the surface code.
            9. Maybe enable how the error was found/corrected. Visualize
            the actual errors that occurred vs the detected ones and what corrected them.
            10. Calculate the logical error rate over multiple shots.)

            """

            if not hasattr(self, 'instructions'):
                self.instructions: List[Dict[str, Any]] = []
            
            checked_instructions: List[Dict[str, Any]] = []
            for instr in instructions:
                valid_qubits = []
                operation = instr['operation']
                qubits = instr['qubits']
                assert operation in GATE_TYPES, f"Invalid operation {operation}. Must be one of {GATE_TYPES}."
                
                match operation, qubits:
                    # any gate checked for all qubits
                    case _, 'all':
                        if operation in ['H', 'X', 'Y', 'Z']:
                            # apply to all data qubits
                            valid_qubits = [coord for coord, qtype in self.index_mapping.values() if qtype == 'data']

                        else: # CNOT case
                            raise ValueError("Cannot apply 'all' to CNOT operation without specifying control and target qubits.")
                    
                    # CNOT gate with list of qubits
                    case 'CNOT', list() as qubit_list:
                        # Case that it is a CNOT operation with a list of qubit specifications
                        for control_target in qubit_list:
                            if self.validate_qubit_spec(control_target, operation):
                                valid_qubits.append(control_target)
                            else:
                                print(f"Warning: Invalid qubit specification {control_target} for operation {operation}. Skipping this qubit.")
                    
                    # Single qubit gate with list of qubits
                    case _, list() as qubit_list:
                        # Case that it is a single qubit operation with a list of qubit specifications
                        for qubit in qubit_list:
                            if self.validate_qubit_spec(qubit, operation):
                                valid_qubits.append(qubit)
                            else:
                                print(f"Warning: Invalid qubit specification {qubit} for operation {operation}. Skipping this qubit.")
                    
                    # Invalid instruction
                    case _:
                        print(f"Warning: Invalid instruction for qubit specification {qubits} and operation {operation}. Skipping this instruction.")
                
                # still missing the error cases but for now just store the valid instructions
                if valid_qubits:
                    checked_instructions.append({'operation': operation, 'qubits': valid_qubits})

            self.instructions.extend(checked_instructions)

    def get_instructions(self) -> List[Dict[str, Any]]:
        """ Returns the list of parsed instructions. """
        if hasattr(self, 'instructions'):
            return deepcopy(self.instructions)
        else:
            return []
        
