"""I need MWPM decoder and the correlated decoder here."""

# TODO
from typing import Callable
import numpy as np
import stim

class SurfaceCodeDecoder:

    def __init__(self, type: str = "MWPM") -> None:
        self.decode: Callable = self.decode_simple_mwpm if type == "MWPM" else self.decode_correlated

    def decode_simple_mwpm(self, shots: np.ndarray) ->  None:
        """
        
        """
        """Decode using simple MWPM decoder."""
        pass


    def decode_correlated(self, shots: np.ndarray) -> None:
        """Decode using correlated error decoder with hyperedges."""
        pass