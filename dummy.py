from typing import List, Dict, Any, Tuple, Union, Literal

INSTRUCTION_KEYS = Literal["operation", "operation_error", "initialization_error", "measurement_error", "idle_error"]
GATE_TYPES = Literal["H", "CNOT", "X", "Y", "Z"]
QubitSpec = Union[List[Union[Tuple[int|float, int|float], int]], List[Union[Tuple[Tuple[int|float, int|float], Tuple[int|float, int|float]], Tuple[float, float]]], str]
InstructionDict = Dict[str, Any]

self = ... # type: ignore
self.index_mapping: Dict[int, Tuple[Tuple[Union[int, float], Union[int, float]], str]] = self.get_index_mapping() # type: ignore
self.inverse_mapping: Dict[Tuple[Union[int, float], Union[int, float]], int] = self.get_inverse_mapping() # type: ignore

class PlaceHolder: # type: ignore

    def __init__(self):


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
        

