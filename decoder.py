"""
The idea was to create the decoder class here, which would implement the correlated decoding method
and create the graph with hyperedges, if possible using stim functionalities. 
However, since I did not get to the point of having a working stim circuit with measurement errors etc,
I leave this as a TODO for now.
"""
# TODO
from typing import Callable
import numpy as np
import stim

class SurfaceCodeDecoder:

    def __init__(self, type: str = "MWPM") -> None:
        self.decode: Callable = self.decode_simple_mwpm if type == "MWPM" else self.decode_correlated

    def decode_simple_mwpm(self, circuit:stim.Circuit, shots: np.ndarray) ->  None:
        """Decode using simple MWPM decoder.
        This should be easily done as I have in SurfaceCode class 
        using PyMatching so this might be redundant to that.
        I should just use the stim.Circuit passed here to create that
        similar to the run_simulation functions."""
        pass

    def decode_correlated(self, circuit:stim.Circuit, shots: np.ndarray) -> None:
        """Decode using correlated error decoder with hyperedges.
           Need to construct the hypergraph from the stim.Circuit first."""
        pass