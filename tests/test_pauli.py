from src.AtomLogic import Pauli, PauliWithPhase


def test_basic_products():
    xy = Pauli.X * Pauli.Y
    assert isinstance(xy, PauliWithPhase)
    assert xy.phase == 1j
    assert xy.op == Pauli.Z
    assert str(xy) == "iZ"

    yx = Pauli.Y * Pauli.X
    assert yx.phase == -1j
    assert yx.op == Pauli.Z
    assert str(yx) == "-iZ"

    xx = Pauli.X * Pauli.X
    assert xx.phase == 1
    assert xx.op == Pauli.I
    assert str(xx) == "I"


def test_chain_multiplication():
    prod = (Pauli.X * Pauli.Y) * Pauli.Z  # (iZ) * Z = iI
    assert prod.phase == 1j
    assert prod.op == Pauli.I
    assert str(prod) == "iI"


def test_commutation():
    assert Pauli.X.commutes_with(Pauli.X)
    assert Pauli.X.commutes_with(Pauli.I)
    assert not Pauli.X.commutes_with(Pauli.Z)
    assert not Pauli.Y.commutes_with(Pauli.X)
    assert Pauli.Z.commutes_with(Pauli.Z)


def test_phase_symmetry():
    # XZ = -iY and ZX = iY
    xz = Pauli.X * Pauli.Z
    zx = Pauli.Z * Pauli.X
    assert xz.op == Pauli.Y and zx.op == Pauli.Y
    assert xz.phase == -1j
    assert zx.phase == 1j

