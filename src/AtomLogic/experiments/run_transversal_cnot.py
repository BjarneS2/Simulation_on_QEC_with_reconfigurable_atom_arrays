"""
run_transversal_cnot.py
=======================
Script to run Monte Carlo simulations of a transversal logical CNOT between
two logical qubits and estimate logical error or Bell-state fidelity.

Workflow
--------
1. Initialize two logical blocks (control, target)
2. Inject physical noise into both lattices
3. Measure stabilizers and decode
4. Apply transversal CNOT
5. Measure logical operators
6. Repeat for many trials and compute statistics

Functions
---------
def run_one_experiment(d, p_data_X, p_data_Z, gate_err, meas_err) -> bool
    Runs a single noisy transversal-CNOT experiment and returns True if
    a logical error occurred.

def run_scaling_study(distances, n_trials, noise_params)
    Runs many trials for various code distances and plots logical error vs. d.

Outputs
-------
• Logical error rates, Bell-pair fidelities
• Optionally: plots saved under `data/figures/`
"""
