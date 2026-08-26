"""Build and run the decoder directly from a Stim circuit.

This portable example uses the checked-in distance-3 rotated surface-code
memory experiment and runs S3 only, so it needs neither a CUDA GPU nor a
Gurobi license. Replace ``toy_xz_surface_code_memory.stim`` with your own
compatible noisy X/Z memory circuit, then enable the GPU or IP stages as
appropriate.

    python examples/from_stim_circuit.py
"""
from pathlib import Path

import stim

from telescoping_decoder import TelescopeConfig, TelescopingDecoder

N_SHOTS = 100
CIRCUIT_PATH = Path(__file__).with_name("toy_xz_surface_code_memory.stim")


def main() -> None:
    circuit = stim.Circuit.from_file(CIRCUIT_PATH)

    # Keep the example portable. On a CUDA machine with the [gpu] extra,
    # leave S1/S2 enabled to run the complete GPU -> CPU telescope.
    cfg = TelescopeConfig()
    cfg.s1.enabled = False
    cfg.s2.enabled = False
    cfg.s3.n_procs = 1
    cfg.s4.enabled = False

    detectors, actual_observables = circuit.compile_detector_sampler(
        seed=1,
    ).sample(
        shots=N_SHOTS,
        separate_observables=True,
    )

    with TelescopingDecoder.from_stim_circuit(
        circuit,
        init_basis="X",
        config=cfg,
    ) as decoder:
        result = decoder.decode(
            detectors,
            true_obs=actual_observables,
        )

    print(result.summary())


if __name__ == "__main__":
    main()
