from typing import Optional
import stim

def combine_circuits_parallel_deprecated(circ1: stim.Circuit, circ2: stim.Circuit, 
                              instructions: Optional[list[tuple[str, float|int]]]) -> stim.Circuit:
    """
    Combines two circuits to run simultaneously side-by-side.
    """
    circ1 = circ1.flattened()
    circ2 = circ2.flattened()

    shift = circ1.num_qubits
    
    # Calculate the Observable shift
    obs_shift = 0
    for instr in circ1:
        if instr.name == "OBSERVABLE_INCLUDE":
            args = instr.gate_args_copy()  # type: ignore
            if args:
                obs_shift = max(obs_shift, int(args[0]) + 1)

    # Re-build circ2 with shifted indices
    circ2_shifted = stim.Circuit()
    for instr in circ2:
        new_targets = []
        for t in instr.targets_copy():  # type: ignore
            if t.is_qubit_target:
                new_targets.append(t.value + shift)
            elif t.is_x_target:
                new_targets.append(stim.target_x(t.value + shift))
            elif t.is_y_target:
                new_targets.append(stim.target_y(t.value + shift))
            elif t.is_z_target:
                new_targets.append(stim.target_z(t.value + shift))
            elif t.is_combiner:
                new_targets.append(stim.target_combiner())
            else:
                new_targets.append(t)
        
        # Shift the observable index
        new_args = instr.gate_args_copy()  # type: ignore
        if instr.name == "OBSERVABLE_INCLUDE":
            if new_args:
                new_args[0] += obs_shift
                
        circ2_shifted.append(instr.name, new_targets, new_args)

    # Merge them TICK by TICK
    full_circuit = stim.Circuit()
    iter1 = iter(circ1)
    iter2 = iter(circ2_shifted)
    
    # Simple zipper merge
    while True:
        chunk1 = stim.Circuit()
        try:
            while True:
                op = next(iter1)
                chunk1.append(op)
                if op.name == "TICK": 
                    break
        except StopIteration:
            pass

        chunk2 = stim.Circuit()
        try:
            while True:
                op = next(iter2)
                chunk2.append(op)
                if op.name == "TICK": 
                    break
        except StopIteration:
            pass

        """Exit the loop."""
        if len(chunk1) == 0 and len(chunk2) == 0:
            break
            
        for op in chunk1:
            if op.name != "TICK": 
                full_circuit.append(op)
        for op in chunk2:
            if op.name != "TICK": 
                full_circuit.append(op)
            
        full_circuit.append("TICK")  # type: ignore

    return full_circuit

def shift_target(t: stim.GateTarget, shift: int) -> stim.GateTarget|int:
    """Helper to shift the qubit index of various target types."""
    if t.is_qubit_target:
        # Check for combiner first (which has a special target value of -1)
        if t.is_combiner:
            return stim.target_combiner()
        
        # This handles simple qubits (like T_REC 0, T_REC 1, etc. in string form)
        # For a simple qubit target (target_t), the value is just the index.
        # We return the new integer index here.
        return t.value + shift
        
    elif t.is_x_target:
        return stim.target_x(t.value + shift)
    elif t.is_y_target:
        return stim.target_y(t.value + shift)
    elif t.is_z_target:
        return stim.target_z(t.value + shift)
    
    # Leave non-qubit/non-pauli targets (like rec[-k], sweep bits) unshifted here.
    return t


def combine_circuits_parallel(circ1: stim.Circuit, circ2: stim.Circuit) -> stim.Circuit:
    """
    Combines two circuits side-by-side, fixing:
    1. Qubit indices (Logic)
    2. Measurement record indices (Logic)
    3. Coordinate data (Visualization/Layout)
    """
    circ1 = circ1.flattened()
    circ2 = circ2.flattened()

    # --- 1. Calculate Index Shifts ---
    shift = circ1.num_qubits
    
    obs_shift = 0
    for instr in circ1:
        if instr.name == "OBSERVABLE_INCLUDE":
            args = instr.gate_args_copy()  # type: ignore
            if args:
                obs_shift = max(obs_shift, int(args[0]) + 1)

    # --- 2. Calculate Coordinate Shifts (Visual Layout) ---
    # Find the bounding box of circ1 to place circ2 "below" it.
    max_y = 0.0
    for instr in circ1:
        if instr.name in ["QUBIT_COORDS", "DETECTOR"]:
            args = instr.gate_args_copy()  # type: ignore
            if len(args) >= 2:
                max_y = max(max_y, args[1])
    
    # Gap between the two circuits visually (shift Y by max_y + padding)
    y_shift = max_y + 4.0 

    # --- 3. Prepare Iterators ---
    iter1 = iter(circ1)
    iter2 = iter(circ2)
    
    full_circuit = stim.Circuit()
    
    # Track measurement indices
    meas_map_1 = [] 
    meas_map_2 = []
    
    current_global_meas = 0

    while True:
        # --- Collect Chunk 1 ---
        chunk1 = []
        try:
            while True:
                op = next(iter1)
                if op.name == "TICK":
                    break
                chunk1.append(op)
        except StopIteration:
            pass

        # --- Collect Chunk 2 ---
        chunk2 = []
        try:
            while True:
                op = next(iter2)
                if op.name == "TICK":
                    break
                chunk2.append(op)
        except StopIteration:
            pass

        if not chunk1 and not chunk2:
            break

        # --- Process Chunk 1 ---
        for op in chunk1:
            if op.name in ["M", "MR", "MX", "MY", "MZ"]:
                for _ in op.targets_copy():
                    meas_map_1.append(current_global_meas)
                    current_global_meas += 1
                full_circuit.append(op)
            
            elif op.name in ["DETECTOR", "OBSERVABLE_INCLUDE"]:
                new_targets = []
                for t in op.targets_copy():
                    if t.is_measurement_record_target:
                        local_idx = len(meas_map_1) + t.value
                        global_idx = meas_map_1[local_idx]
                        new_offset = global_idx - current_global_meas
                        new_targets.append(stim.target_rec(new_offset))
                    else:
                        new_targets.append(t)
                full_circuit.append(op.name, new_targets, op.gate_args_copy())
            else:
                full_circuit.append(op)

        # --- Process Chunk 2 (Apply Shifts) ---
        for op in chunk2:
            new_targets = []
            new_args = op.gate_args_copy()

            # A. Shift Visual Coordinates (Y-axis)
            if op.name in ["QUBIT_COORDS", "DETECTOR"]:
                if len(new_args) >= 2:
                    new_args[1] += y_shift

            # B. Shift Observable Index
            if op.name == "OBSERVABLE_INCLUDE":
                 if new_args:
                    new_args[0] += obs_shift
            
            # C. Handle Targets
            # Case: Measurement Definition
            if op.name in ["M", "MR", "MX", "MY", "MZ"]:
                for t in op.targets_copy():
                    new_targets.append(shift_target(t, shift))
                    meas_map_2.append(current_global_meas)
                    current_global_meas += 1
            
            # Case: Measurement Reference (Detector/Observable)
            elif op.name in ["DETECTOR", "OBSERVABLE_INCLUDE"]:
                 for t in op.targets_copy():
                    if t.is_measurement_record_target:
                        # Remap Rec
                        local_idx = len(meas_map_2) + t.value
                        global_idx = meas_map_2[local_idx]
                        new_offset = global_idx - current_global_meas
                        new_targets.append(stim.target_rec(new_offset))
                    else:
                        new_targets.append(shift_target(t, shift))
            
            # Case: Standard Instructions (Gates, Qubit Coords)
            else:
                 for t in op.targets_copy():
                    new_targets.append(shift_target(t, shift))

            full_circuit.append(op.name, new_targets, new_args)

        full_circuit.append("TICK")  # type: ignore

    return full_circuit
