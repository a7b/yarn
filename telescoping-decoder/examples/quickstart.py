"""Quickstart: decode sampled syndromes through the telescoping decoder.

Uses the bundled example artifacts: a [[150, 30, 10]] code, 10 rounds of
syndrome measurement at a physical error rate of 0.001. The GPU stages (S1,
S2) run when a CUDA GPU is visible and are skipped otherwise, so this script
works on any machine. The package installs the S4 `gurobipy` dependency, but
S4 still needs a full Gurobi license; point it at one with GRB_LICENSE_FILE
(or drop a gurobi.lic at the repo root).

    python examples/quickstart.py
    GRB_LICENSE_FILE=/path/to/gurobi.lic python examples/quickstart.py
"""
import os
from pathlib import Path

import numpy as np
import stim

from telescoping_decoder import TelescopeConfig, TelescopingDecoder

N_SHOTS = 1000

STEM = (Path(__file__).resolve().parent.parent / "tests" / "data"
        / "paper_nonab_150_30_10_xyz_p0.001_r10_xinit_L1_hookfree")


def free_vram_bytes():
    """Free VRAM on device 0, or None when there is no usable GPU."""
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() == 0:
            return None
        return int(cupy.cuda.runtime.memGetInfo()[0])
    except Exception:
        return None


def gurobi_license():
    """The S4 license to hand the IP workers, or None to let Gurobi look.

    Checked in order: GRB_LICENSE_FILE, then a gurobi.lic sitting at the repo
    root. Setting cfg.s4.license_file makes the choice explicit instead of
    relying on each worker inheriting the environment.
    """
    env_license = os.environ.get("GRB_LICENSE_FILE")
    if env_license:
        return env_license
    root_license = Path(__file__).resolve().parent.parent / "gurobi.lic"
    return str(root_license) if root_license.is_file() else None


def main() -> None:
    # 1. Build the decoder from prepared artifacts. (Alternatively:
    #    TelescopingDecoder.from_stim_circuit(circuit, init_basis="X") or
    #    .from_dem(dem, is_x_detector=mask, init_basis="X").)
    free_vram = free_vram_bytes()
    cfg = TelescopeConfig()
    cfg.s1.enabled = cfg.s2.enabled = free_vram is not None
    if free_vram is not None and free_vram < 6 * 2**30:
        # Reduce the batches to fit a smaller card. S2's random schedule is
        # batch-dependent, so keep these settings fixed when reproducing a run.
        cfg.s2.batch_size = cfg.s2.shots_per_batch = 256
    print("GPU stages: " + ("off (no CUDA device found)" if free_vram is None
                            else f"on ({free_vram // 2**20} MiB free)"))

    # S4 stays on "auto" (active iff gurobipy imports), but the license is
    # named here rather than left to Gurobi's own lookup. The restricted
    # license bundled with pip install gurobipy is too small for this model,
    # so without a full one S4 shots come back labeled IP_no_license.
    cfg.s4.license_file = gurobi_license()
    print("S4 license: " + (cfg.s4.license_file
                            or "not found (set GRB_LICENSE_FILE for S4)"))

    dec = TelescopingDecoder.from_npz(
        gari_npz=f"{STEM}_gari_matrices.npz",
        matrices_npz=f"{STEM}_matrices.npz",
        config=cfg,
    )

    # 2. Sample syndromes from the DEM. A hardware integration supplies the
    #    same (B, n_det) uint8 array in Stim detector order.
    dem = stim.DetectorErrorModel.from_file(f"{STEM}.dem")
    det, obs, _ = dem.compile_sampler(seed=1).sample(
        shots=N_SHOTS, bit_packed=False)
    syndromes = det.astype(np.uint8)
    true_obs = obs.astype(np.uint8)     # optional — enables LE scoring

    # 3. Decode. Without true_obs you still get per-shot predicted
    #    observables (result.obs_pred) and the accepting stage.
    with dec:
        result = dec.decode(syndromes, true_obs=true_obs)

    print(result.summary())
    print()
    print("per-stage diagnostics:")
    for key, val in result.diagnostics.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
