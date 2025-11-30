"""I need MWPM decoder and the correlated decoder here."""

# TODO
import numpy as np
import stim

class SurfaceCodeDecoder:

    def __init__(self, circuit: stim.Circuit) -> None:
        self.circuit = circuit
        # and whatever else is needed

    def decode_simple_mwpm(self, shots: np.ndarray) -> np.ndarray:
        """
        
        """
        """Decode using simple MWPM decoder."""
        pass


    def decode_correlated(self, shots: np.ndarray) -> np.ndarray:
        """Decode using correlated error decoder with hyperedges."""
        pass