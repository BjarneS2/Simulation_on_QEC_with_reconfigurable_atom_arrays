"""
How I think of a workflow - for now without initialization errors.
Simulates surface code logical error rates under different noise envs

- measurement errors
- single qubit gate errors
- idle errors
- two qubit gate errors

"""
from datetime import datetime
from itertools import product
from typing import Literal
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pymatching
import h5py
import json
import time
import stim

import SurfaceCodeDecoder  # type: ignore
# these need to be implemented maybe in a different file
# cause this is getting crowded


def now():
    # Time stamp
    return datetime.now().isoformat()

def next_dataset_name(h5grp):
    """Get the next name for hdf5 saving"""
    existing = [n for n in h5grp.keys()]
    if not existing:
        return "0001"
    nums = [int(n) for n in existing if n.isdigit()]
    nxt = (max(nums) + 1) if nums else 1
    return f"{nxt:04d}"

def save_results(path:str|Path, metadata:dict, results:dict, d:int) -> None:
    """
    So I have something that looks like:
    CodeCistance/
        - 3/
             - 0001/
             - 0002/
             ...
        - 5/ 
             - 0001/
             ...
        ...

    Where each dataset contains all params results and timestamps.

    metadata: dict with parameters + timestamp + runtime
    results: dict? with decoding results (scalars)
    """

    with h5py.File(path, "a") as f:
        root = f.require_group("CodeDistance")
        grp_d = root.require_group(str(d))
        name = next_dataset_name(grp_d)
        grp = grp_d.create_group(name)

        for k, v in results.items():
            grp.create_dataset(k, data=float(v))

        for k, v in metadata.items():
            try:
                grp.attrs[k] = v
            except Exception:
                grp.attrs[k] = json.dumps(v)

        grp.attrs["saved_at"] = now()
        f.flush() 

def run_simulation(code_distance, p_meas, p_gate, p_idle, p_cnot, n_trials,
                   mode:Literal["correlated", "simple"] = "correlated"):
    # Placeholder for the actual simulation for now
    
    if mode == "simple":
        # MWPM decoding for logical error rate
        t0 = time.time()
        results = stim.SurfaceCode(...).decode(...) # type: ignore
        sim_time = t0 - time.time()

        # do something with results
        logical_error_rate = 1e-2
    

    elif mode == "correlated":
        # Correlated error decoding explained in paper
        # MWPM with hyperedges
        t0 = time.time()
        Code = stim.SurfaceCode(...)  # type: ignore
        decoder = pymatching.Matching.something(...)  # type: ignore
        results = Code.decode(...)  # type: ignore
        sim_time = t0 - time.time()

        # do something with results
        logical_error_rate = 1e-2


    return {"logical_error_rate": float(logical_error_rate),
            "runtime": float(sim_time),}


# def run_simulation(code_distance, p_meas, p_gate, p_idle, p_cnot, n_trials,
#                    mode:Literal["correlated", "simple"]):
    
#     t0 = time.time()
    
#     # 1. Generate Clean Circuit
#     clean_circuit = noise_models.generate_clean_surface_code(
#         distance=code_distance, 
#         rounds=code_distance
#     )

#     # 2. Apply Specific Noise Models
#     noisy_circuit = noise_models.apply_custom_noise(
#         clean_circuit, 
#         p_gate=p_gate, 
#         p_meas=p_meas, 
#         p_idle=p_idle, 
#         p_cnot=p_cnot
#     )

#     # 3. Sample Shots
#     # compile_detector_sampler returns binary data: [measurements, detectors, observables]
#     # We usually want detectors and observables. 
#     sampler = noisy_circuit.compile_detector_sampler()
#     shots = sampler.sample(shots=n_trials, bit_packed=False)

#     # 4. Decode
#     decoder_engine = decoders.SurfaceCodeDecoder(noisy_circuit)
    
#     if mode == "simple":
#         logical_error_rate = decoder_engine.decode_simple_mwpm(shots)
#     elif mode == "correlated":
#         logical_error_rate = decoder_engine.decode_correlated(shots)
#     else:
#         raise ValueError(f"Unknown mode: {mode}")

#     sim_time = time.time() - t0

#     return {
#         "logical_error_rate": float(logical_error_rate),
#         "runtime": float(sim_time)
#     }

def main(code_distance, p_gate, p_meas, p_idle, p_cnot, n_trials, out_path):
    total = len(code_distance) * len(p_gate) * len(p_meas) * len(p_idle)
    print(f"Total configurations: {total}")
    print(f"Total trials per config: {n_trials}")
    
    pbar = tqdm(total=total, desc="Simulations")
    
    for d, (gate_idx, p_gate_val), p_meas_val, p_idle_val in product(
        code_distance, enumerate(p_gate), p_meas, p_idle
    ):
        if len(p_cnot) > gate_idx:
            p_cnot_val = p_cnot[gate_idx]
        else:
            p_cnot_val = p_gate_val

        metadata = {
            "code_distance": int(d),
            "p_gate": float(p_gate_val),
            "p_meas": float(p_meas_val),
            "p_idle": float(p_idle_val),
            "p_cnot": float(p_cnot_val),
            "n_trials": int(n_trials),
        }

        # Run Correlated
        res_corr = run_simulation(d, p_meas_val, p_gate_val, p_idle_val, 
                                  p_cnot_val, n_trials, "correlated")

        save_results(out_path, {**metadata, "decoder": "correlated"}, res_corr, d)
        
        res_simp = run_simulation(d, p_meas_val, p_gate_val, p_idle_val, p_cnot_val, n_trials, "simple")
        save_results(out_path, {**metadata, "decoder": "simple"}, res_simp, d)
        
        pbar.update(1)

if __name__ == "__main__":
    distances = [3, 5]
    
    p_meas_range = np.logspace(-3, -2, 2)
    p_gate_range = np.logspace(-3, -2, 2)
    p_idle_range = [0.0]
    p_cnot_range = np.logspace(-3, -2, 2) 

    n_trials = 1000

    OUT_DIR = Path("./simulation_data")
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    H5_PATH = OUT_DIR / "surface_code_results.h5"

    main(
        code_distance=distances,
        p_gate=p_gate_range,
        p_meas=p_meas_range,
        p_idle=p_idle_range,
        p_cnot=p_cnot_range,
        n_trials=n_trials,
        out_path=H5_PATH
    )

