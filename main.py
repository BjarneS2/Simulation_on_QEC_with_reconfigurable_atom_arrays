#%%

from src.AtomLogic import PhysicalQubit, Lattice, LogicalBlock
from src.AtomLogic.gates.physical_gates import Z, H, Rx
# %%
def main():
    print("=== PhysicalQubit demo ===")
    qbit = PhysicalQubit(p_x=0.3, p_z=0.4, keep_history=True)
    print(f"Initial error: {qbit.current_error}")
    qbit.apply_pauli('X')
    print(f"Apply X -> {qbit.current_error}")
    qbit.apply_pauli('Z')
    print(f"Apply Z -> {qbit.current_error}")
    sampled = qbit.sample_error()
    print(f"Sample error ({sampled}) -> {qbit.current_error}")
    print(f"History: {qbit.history}\n")

    print("=== Lattice demo (d=3) ===")
    lat = Lattice(d=3, p_x=0.1, p_z=0.15, keep_history=False, seed=42)
    print(lat.summary())
    print("Qubit errors (initial):", lat.qubit_errors())
    lat.sample_physical_errors()  # sample with default per-qubit probabilities
    print("Qubit errors (after sampling):", lat.qubit_errors())
    synd = lat.measure_stabilizers(measurement_error=0.0)
    print("Stabilizer outcomes:", synd)
    print("Logical Z parity:", lat.logical_parity('Z'))
    print("Logical X parity:", lat.logical_parity('X'))

    # Show another round with measurement error
    lat.sample_physical_errors(p_x=0.2, p_z=0.2)
    synd2 = lat.measure_stabilizers(measurement_error=0.05)
    print("Stabilizer outcomes (round 2, with meas noise):", synd2)
    print("Logical parities round 2 -> Z:", lat.logical_parity('Z'), "X:", lat.logical_parity('X'))

    print("\n=== LogicalBlock demo (d=5) ===")
    lb = LogicalBlock(d=5, p_x=0.02, p_z=0.03, keep_history=False, seed=7)
    print(lb.summary())
    print("-- Apply logical X --")
    lb.apply_logical('X')
    print(lb.summary())
    print("-- Apply logical Z via gate helper --")
    Z(lb)
    print(lb.summary())
    print("-- Apply logical H (frame swap) --")
    H(lb)
    print(lb.summary())
    print("-- Inject noise and run syndrome round --")
    lb.inject_noise()
    lb.run_syndrome_round()
    print(lb.summary())
    print("-- Apply rotation Rx(pi) via reduction --")
    Rx(lb, theta=3.141592653589793)
    print(lb.summary())
    print("-- Measure logical Z and X --")
    mz = lb.measure_logical('Z')
    mx = lb.measure_logical('X')
    print(f"Logical measurement results: Z={mz} X={mx}")

# %%
if __name__ == "__main__":
    main()

# %%
