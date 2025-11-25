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

import simple_decoder, correlated_decoder  # type: ignore
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

def main(code_distance, p_gate, p_meas, p_idle, p_cnot, n_trials):

    total = len(code_distance) * len(p_gate) * len(p_meas) * len(p_idle)
    print(f"Total simulations to run: {total}")
    pbar = tqdm(total=total, desc="Simulations completed")
    

    for d, p_gate_val, p_meas_val, p_idle_val in product(
        code_distance, p_gate, p_meas, p_idle
    ):
        if cnot_same_as_gate:
            p_cnot_val = p_gate_val
        else:
            try:
                p_cnot_val = p_cnot[np.where(p_gate == p_gate_val)[0][0]]
            except IndexError:
                print("p_cnot length mismatch with p_gate, using p_gate value for p_cnot")
                print("current run: d:", d, "p_gate:", p_gate_val, "p_meas:", p_meas_val, "p_idle:", p_idle_val)
                p_cnot_val = p_gate_val

        metadata = {
            "code_distance": int(d),
            "p_gate": float(p_gate_val),
            "p_meas": float(p_meas_val),
            "p_idle": float(p_idle_val),
            "p_cnot": float(p_cnot_val),
            "n_trials": int(n_trials),
            "started_at": now(),
        }

        results = run_simulation(
            code_distance=d,
            p_meas=p_meas_val,
            p_gate=p_gate_val,
            p_idle=p_idle_val,
            p_cnot=p_cnot_val,
            n_trials=n_trials,
            mode="correlated"
        )
        save_results(H5_PATH, metadata, results, d)
        pbar.update(0.5)

        metadata["started_at"] = now()
        results = run_simulation(
            code_distance=d,
            p_meas=p_meas_val,
            p_gate=p_gate_val,
            p_idle=p_idle_val,
            p_cnot=p_cnot_val,
            n_trials=n_trials,
            mode="simple"
        )
        save_results(H5_PATH, metadata, results, d)
        pbar.update(0.5)


if __name__ == "__main__":
    code_distance = [3, 5, 7, 9, 11, 13, 15]
    # I wanna let this run and dump/save data after each round 
    # so I can just let it run for some time and not worry about losing data

    p_meas = np.logspace(-4, -1, 4)  # measurement errors
    p_gate = np.logspace(-4, -1, 10) # single qubit gate errors
    p_idle = np.logspace(-5, -3, 3)  # less errors when no gates are applied
    p_cnot = np.logspace(-3, -1, 3)  # cnot tends to be worse than single q
    # might wanna use p_cnot same as p_gate for simplicity though

    cnot_same_as_gate = True
    if cnot_same_as_gate:
        p_cnot = p_gate

    n_trials = 10000  #10k seems valid

    OUT_DIR = Path("./simulation_data")
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    H5_PATH = OUT_DIR / "surface_code_correlated.h5"


    main(
        code_distance=code_distance,
        p_gate=p_gate,
        p_meas=p_meas,
        p_idle=p_idle,
        p_cnot=p_cnot,
        n_trials=n_trials,
    )

