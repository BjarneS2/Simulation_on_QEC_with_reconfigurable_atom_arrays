"""
I want to create a class that represents a surface code for qec.
I want to use the Surface Code class to initialize a surface code of a given distance.
"""

from tabnanny import check
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge, Patch
from typing import List, Literal, Tuple, Dict, Any, Union, Literal
import matplotlib as mpl
import seaborn as sns
import stim
from copy import deepcopy 


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

INSTRUCTION_KEYS = Literal["operation", "operation_error", "initialization_error", "measurement_error", "idle_error"]
GATE_TYPES = Literal["H", "CNOT", "X", "Y", "Z"]
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
    
    def parse_instructions(self, instructions: List[Dict[str, Any]]) -> dict:
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
            valid_qubits: List[Union[Tuple[int, int], int, Tuple[Tuple[int, int], Tuple[int, int]]]] = []

            if 'operation' in instr:  ### NOTE: checking operations
                operation = instr['operation']
                qubits = instr['qubits']

                if operation == "CNOT" and isinstance(qubits[0], tuple) and isinstance(qubits[0][0], tuple):
                    for control_coord, target_coord in qubits:
                        if (control_coord in self.qubit_coords or control_coord in self.stabilisers_coords) and \
                           (target_coord in self.qubit_coords or target_coord in self.stabilisers_coords):
                            valid_qubits.append((control_coord, target_coord))
                        
                        else:
                            print(f"Warning: Invalid qubit specification {(control_coord, target_coord)} for operation {operation}. Skipping this qubit.")
                    
                    qubits = valid_qubits

                if qubits == 'all': ### NOTE: all qubits
                    qubits = self.get_all_qubits(_as="coord")

                elif qubits and isinstance(qubits[0], tuple): ### NOTE: if coordinates provided
                    # Coordinates provided, validate them
                    valid_qubits = []
                    for coord in qubits:
                        if coord in self.qubit_coords or coord in self.stabilisers_coords:
                            valid_qubits.append(coord)
                        else:
                            print(f"Warning: Invalid qubit specification {coord} for operation {operation}. Skipping this qubit.")
                    qubits = valid_qubits

                elif qubits and isinstance(qubits[0], int):
                    # Indices provided, validate them
                    valid_qubits = []
                    for idx in qubits:
                        if idx in self.index_mapping:
                            valid_qubits.append(idx)
                        else:
                            print(f"Warning: Invalid qubit specification {idx} for operation {operation}. Skipping this qubit.")
                    qubits = valid_qubits

                checked_instructions.append({'operation': operation, 'qubits': qubits})

            elif 'operation_error' in instr:
                operation = instr['operation_error']
                qubits = instr['qubits']
                probability = instr.get('probability', 0.0)
                
                if not (0.0 <= probability <= 1.0):
                    print(f"Warning: Probability {probability} for qubit {qubits} and operation {operation} is out of bounds [0, 1]. Clipping to valid range.")
                    probability = max(0.0, min(1.0, probability))
                
                checked_instructions.append({'operation_error': operation, 'qubits': qubits, 'probability': probability})
            
            elif 'initialization_error' in instr:
                operation = instr['initialization_error']
                qubits = instr['qubits']
                probability = instr.get('probability', 0.0)
                
                if not (0.0 <= probability <= 1.0):
                    print(f"Warning: Probability {probability} for qubit {qubits} and operation {operation} is out of bounds [0, 1]. Clipping to valid range.")
                    probability = max(0.0, min(1.0, probability))
                
                checked_instructions.append({'initialization_error': operation, 'qubits': qubits, 'probability': probability})
            
            elif 'measurement_error' in instr:
                operation = instr['measurement_error']
                qubits = instr['qubits']
                probability = instr.get('probability', 0.0)
                
                if not (0.0 <= probability <= 1.0):
                    print(f"Warning: Probability {probability} for qubit {qubits} and operation {operation} is out of bounds [0, 1]. Clipping to valid range.")
                    probability = max(0.0, min(1.0, probability))
                
                checked_instructions.append({'measurement_error': operation, 'qubits': qubits, 'probability': probability})
            
            else:
                print(f"Warning: Invalid instruction format {instr}. Skipping this instruction.")


    def build_in_stim_hadamard_tracker(self, rounds: int = 1) -> stim.Circuit:
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

            In addition to this I need to build a parser that can 
            take a list of operations,qubits and convert them into
            stim operations. but this should only run on the circuit
            before the measurement rounds. So this is where I will
            insert Hadamards and CNOTs as operations and Errors in the 
            future. I will need to also consider the amount of logical
            Hadamards I apply to the code since I need to track them 
            to switch the stabilizer types accordingly.   

        """
        circuit = stim.Circuit()

        for idx, (coord, _) in self.index_mapping.items():
            circuit.append("QUBIT_COORDS", [idx], [coord[0], coord[1]])

        data_indices = [k for k, v in self.index_mapping.items() if v[1] == 'data']
        x_stab_indices = [k for k, v in self.index_mapping.items() if v[1] == 'X_stab']
        z_stab_indices = [k for k, v in self.index_mapping.items() if v[1] == 'Z_stab']
        all_ancillas = x_stab_indices + z_stab_indices

        circuit.append("R", self.index_mapping.keys())  # type: ignore
        circuit.append("TICK")  # type: ignore
        """
        insert error in one qubit for testing
        """
        circuit.append("X", data_indices[:])  # type: ignore
        circuit.append("Z", data_indices[:])  # type: ignore

        loop_body = stim.Circuit()

        loop_body.append("R", all_ancillas)  # type: ignore
        
        for ancilla_idx in all_ancillas:
            coord, qtype = self.index_mapping[ancilla_idx]
            neighbors = self.get_surrounding_data_qubits(coord) # type: ignore
            
            for neighbor_coord in neighbors:
                neighbor_idx = self.inverse_mapping[neighbor_coord]
                
                if qtype == 'X_stab':
                    loop_body.append("H", [neighbor_idx])  # type: ignore
                    loop_body.append("CNOT", [ancilla_idx, neighbor_idx])  # type: ignore
                    loop_body.append("H", [neighbor_idx])  # type: ignore

                elif qtype == 'Z_stab':
                    loop_body.append("CNOT", [neighbor_idx, ancilla_idx])  # type: ignore

        loop_body.append("M", all_ancillas)  # type: ignore
        loop_body.append("TICK")  # type: ignore

        circuit += loop_body * rounds

        return circuit

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
        """
        insert error in one qubit for testing
        """
        circuit.append("X", data_indices[:])  # type: ignore
        circuit.append("Z", data_indices[:])  # type: ignore

        loop_body = stim.Circuit()

        loop_body.append("R", all_ancillas)  # type: ignore
        
        # X-Stabilizers (Measure X parity -> Detect Z Errors)
        # Requirement: Apply H on stabilizer
        # if x_stab_indices:
        #     loop_body.append("H", x_stab_indices)  # type: ignore
        
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
                    loop_body.append("H", [neighbor_idx])  # type: ignore
                    loop_body.append("CNOT", [ancilla_idx, neighbor_idx])  # type: ignore
                    loop_body.append("H", [neighbor_idx])  # type: ignore

                elif qtype == 'Z_stab':
                    # Measuring Z: CNOT(Data, Anc)
                    # This propagates Z from Data to Ancilla (checking Z parity)
                    loop_body.append("CNOT", [neighbor_idx, ancilla_idx])  # type: ignore

        # X-Stabilizers: Apply closing H on stabilizer
        # if x_stab_indices:
        #     loop_body.append("H", x_stab_indices)  # type: ignore

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



        
    def is_valid_coordinate(self, coord: Tuple[int|float, int|float]) -> bool:
        """ Check if the given coordinate is valid within the surface code. """
        x, y = coord
        if not (isinstance(x, int) and isinstance(y, int) or isinstance(x, float) and isinstance(y, float)):
            return False
        # Now I need to check if the coordinates are in the inverse mapping items
        return (coord in self.inverse_mapping().keys())
    
    def is_valid_index(self, index: int) -> bool:
        """ Check if the given index is valid within the surface code. """
        return index in self.index_mapping.keys()
        
    def validate_qubit_spec(qubit_spec: Union[Tuple[int|float, int|float], int, Tuple[Tuple[int|float, int|float]]], operation: str) -> bool:
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
                return self.is_valid_coordinate(qubit_spec[0]) and self.is_valid_coordinate(qubit_spec[1])
            
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

                    case _, 'all':
                        if operation in ['H', 'X', 'Y', 'Z']:
                            # apply to all data qubits
                            valid_qubits = [coord for coord, qtype in self.index_mapping.values() if qtype == 'data']

                        else: # CNOT case
                            raise ValueError("Cannot apply 'all' to CNOT operation without specifying control and target qubits.")

                    case 'CNOT', list() as qubit_list:

                        pass

                    case _, list() as qubit_list:
                        
                        for qubit in qubit_list:
                            if self.validate_qubit_spec(qubit, operation):
                                valid_qubits.append(qubit)
                            else:
                                print(f"Warning: Invalid qubit specification {qubit} for operation {operation}. Skipping this qubit.")

