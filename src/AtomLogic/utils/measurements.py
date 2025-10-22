"""
measurements.py
===============
Utility functions for converting stabilizer outcomes into logical measurement results.

Responsibilities
----------------
• Aggregate stabilizer measurements across rounds
• Compute parity, correlations, and post-selection criteria
• Generate summary metrics (logical fidelity, success probability, etc.)

Core API
--------
def compute_logical_fidelity(results) -> float
def postselect(data, criteria) -> filtered_data
def pretty_print_syndrome(syndrome: dict)

Used By
-------
`experiments/` scripts.
"""
