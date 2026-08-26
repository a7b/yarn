from pathlib import Path

import numpy as np
import pytest
import stim

from telescoping_decoder import (TelescopeConfig, TelescopingDecoder,
                                  circuit_to_dem, circuit_xz_detector_mask)


EXAMPLE_CIRCUIT = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "toy_xz_surface_code_memory.stim"
)


def test_example_circuit_constructs_full_decoding_system(tmp_path):
    circuit = stim.Circuit.from_file(EXAMPLE_CIRCUIT)

    generated = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        before_round_data_depolarization=0.001,
    )
    assert circuit == generated
    assert circuit.num_qubits == 26
    assert circuit.num_detectors == 24
    assert circuit.num_observables == 1

    np.testing.assert_array_equal(
        circuit_xz_detector_mask(circuit),
        np.array(
            [True] * 4
            + [True, False, True, False, False, True, False, True] * 2
            + [True] * 4,
        ),
    )

    # This small graph deliberately has too little shared structure for GARI
    # to reduce nnz; construction should still succeed and explain the case.
    with pytest.warns(UserWarning, match="expected only for toy DEMs"):
        decoder = TelescopingDecoder.from_stim_circuit(
            circuit,
            init_basis="X",
            verify=True,
            verify_samples=64,
            workdir=tmp_path,
        )
    assert decoder.system.H.shape == (24, 53)
    assert decoder.system.L.shape == (1, 53)
    assert decoder.system.has_original
    assert decoder.system.has_gari
    assert decoder.system.can_init_dets

    detectors, observables = circuit.compile_detector_sampler(seed=1).sample(
        shots=8,
        separate_observables=True,
    )
    assert detectors.shape == (8, decoder.system.n_detectors)
    assert observables.shape == (8, decoder.system.n_obs)


def test_mask_rejects_mixed_basis_detector():
    circuit = stim.Circuit("""
        RX 0
        R 1
        MX 0
        M 1
        DETECTOR rec[-2] rec[-1]
    """)

    with pytest.raises(ValueError, match="mixes measurement bases"):
        circuit_xz_detector_mask(circuit)


def test_mask_flattens_repeat_blocks_in_detector_order():
    circuit = stim.Circuit("""
        REPEAT 2 {
            RX 0
            MX 0
            DETECTOR rec[-1]
        }
    """)

    np.testing.assert_array_equal(
        circuit_xz_detector_mask(circuit),
        np.array([True, True]),
    )


def test_mask_rejects_detector_using_unsupported_measurement_gate():
    circuit = stim.Circuit("""
        MPP X0
        DETECTOR rec[-1]
    """)

    with pytest.raises(ValueError, match=r"unsupported gate.*MPP"):
        circuit_xz_detector_mask(circuit)


def test_unsupported_measurement_keeps_later_record_offsets_correct():
    circuit = stim.Circuit("""
        MPP X0
        M 1
        DETECTOR rec[-1]
    """)

    np.testing.assert_array_equal(
        circuit_xz_detector_mask(circuit),
        np.array([False]),
    )


def test_original_only_fallback_accepts_non_xz_measurement_circuit():
    circuit = stim.Circuit("""
        RX 0
        Z_ERROR(0.1) 0
        MPP X0
        DETECTOR rec[-1]
    """)
    cfg = TelescopeConfig(use_gari=False)

    decoder = TelescopingDecoder.from_dem(circuit_to_dem(circuit), config=cfg)

    assert decoder.system.has_original
    assert not decoder.system.has_gari
    assert not decoder.system.can_init_dets
