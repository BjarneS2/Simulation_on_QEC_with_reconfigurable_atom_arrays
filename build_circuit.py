import stim


def create_clean_circuit(distance: int, rounds: int) -> stim.Circuit:
    """Create a clean surface code circuit."""
    return stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                distance=distance,
                rounds=rounds,
                after_clifford_depolarization=0,
                after_reset_flip_probability=0,
                before_measure_flip_probability=0,
                before_round_data_depolarization=0
            )

def apply_noise(circuit, p_gate: float, p_meas: float, p_idle: float, p_cnot: float):
    """Apply noise to the clean circuit."""
    noisy_circuit = stim.Circuit()
    
    # Define target sets for convenience
    single_qubit_gates = {"H"}
    two_qubit_gates = {"CX", "CNOT"}
    measure_gates = {"M", "MX", "MY", "MZ"}
    
    for instruction in circuit:
        noisy_circuit.append(instruction)
        name = instruction.name
        targets = instruction.targets_copy()

        if name in measure_gates:
            if p_meas > 0:
                noisy_circuit.append("X_ERROR", targets, p_meas)
                # and on all qubits that are non targets apply idle error
                for qubit in range(circuit.num_qubits):
                    if qubit not in targets and p_idle > 0:
                        noisy_circuit.append("DEPOLARIZE1", [qubit], p_idle)
                    

        elif name in single_qubit_gates:
            if p_gate > 0:
                noisy_circuit.append("DEPOLARIZE1", targets, p_gate)

        elif name in two_qubit_gates:
            if p_cnot > 0:
                noisy_circuit.append("DEPOLARIZE2", targets, p_cnot)
        
        return noisy_circuit

def build_circuit(distance, rounds:int, p_gate:float, p_meas:float, p_idle:float, 
                  p_cnot:float, instructions: list[tuple]) -> stim.Circuit:
    """Build the custom circuit and apply the noise on it."""
    clean_circuit: stim.Circuit = create_clean_circuit(distance, rounds)
    print("Successfully created clean circuit...")
    clean_circuit.diagram('timeline-svg')

    for gate, targets in instructions:
        clean_circuit.append(gate, targets=targets)  # type: ignore
    
    noisy_circuit = clean_circuit
    print("Successfully applied noise to the circuit...")
    noisy_circuit.diagram('timeline-svg')
    
    # noisy_circuit = apply_noise(
    #     clean_circuit, 
    #     p_gate=p_gate, 
    #     p_meas=p_meas, 
    #     p_idle=p_idle, 
    #     p_cnot=p_cnot
    # )
    return noisy_circuit