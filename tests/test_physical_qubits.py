# tests/test_physical_qubit.py
from src.AtomLogic.core.physical_qubit import PhysicalQubit

def test_apply_sequence():
    q = PhysicalQubit()
    assert q.current_error == 'I'
    q.apply_pauli('X'); assert q.current_error == 'X'
    q.apply_pauli('Z'); assert q.current_error == 'Y'
    q.apply_pauli('X'); assert q.current_error == 'Z'
    q.apply_pauli('Z'); assert q.current_error == 'I'

def test_sampling_extremes():
    q = PhysicalQubit(p_x=1.0, p_z=0.0); q.sample_error(); assert q.current_error == 'X'
    q.reset_error()
    q.set_probabilities(p_x=0.0, p_z=1.0); q.sample_error(); assert q.current_error == 'Z'
    q.reset_error()
    q.set_probabilities(p_x=1.0, p_z=1.0); q.sample_error(); assert q.current_error == 'Y'

def test_history():
    q = PhysicalQubit(keep_history=True)
    q.apply_pauli('X'); q.apply_pauli('Z')
    assert q.history[-1] == 'Y'