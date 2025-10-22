"""
simple_decoder.py
=================
Implements a basic (non-optimal) decoder for surface-code syndromes.

Responsibilities
----------------
• Map syndrome bits to likely error corrections
• Provide minimal viable correction so the pipeline runs before MWPM is added
• Useful for unit testing and early development

Decoder Algorithm (Prototype)
-----------------------------
1. Identify stabilizers with parity 1 (defects)
2. Pair up nearest defects on the lattice
3. Apply a correction chain between them (flip X or Z on path)

Core API
--------
def decode(lattice: Lattice, syndrome: dict) -> list[(qid, pauli)]
    Return list of Pauli corrections to apply.

Used By
-------
`experiments/` and `core/logical_block` when running decoding rounds.
"""
