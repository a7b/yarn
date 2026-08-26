import numpy as np
import pytest
import scipy.sparse as sp
import stim

from telescoping_decoder.system import DecodingSystem, _save_matrices_npz


def _toy_gari_system(workdir):
    dem = stim.DetectorErrorModel(
        "error(0.1) D0 L0\n"
        "error(0.2) D1\n"
        "error(0.3) D0 D1 L0"
    )
    return DecodingSystem.from_dem(
        dem,
        is_x_detector=np.array([True, False]),
        init_basis="X",
        verify=False,
        workdir=workdir,
    )


def test_artifact_identity_includes_observables_and_detector_types(tmp_path):
    H = sp.csr_matrix([[1, 1], [0, 1]], dtype=np.uint8)
    p = np.array([0.1, 0.2])
    L_a = sp.csr_matrix([[1, 0]], dtype=np.uint8)
    L_b = sp.csr_matrix([[0, 1]], dtype=np.uint8)

    a = DecodingSystem.from_matrices(
        H, L_a, p, is_x_detector=[True, False], init_basis="X",
        workdir=tmp_path,
    )
    different_l = DecodingSystem.from_matrices(
        H, L_b, p, is_x_detector=[True, False], init_basis="X",
        workdir=tmp_path,
    )
    different_mask = DecodingSystem.from_matrices(
        H, L_a, p, is_x_detector=[False, True], init_basis="X",
        workdir=tmp_path,
    )

    paths = {
        a.npz_path_for("original"),
        different_l.npz_path_for("original"),
        different_mask.npz_path_for("original"),
    }
    assert len(paths) == 3
    assert len({a.source_fingerprint, different_l.source_fingerprint,
                different_mask.source_fingerprint}) == 3


def test_matching_original_and_gari_artifacts_load(tmp_path):
    system = _toy_gari_system(tmp_path)
    loaded = DecodingSystem.from_npz(
        matrices_npz=system.npz_path_for("original"),
        gari_npz=system.npz_path_for("gari"),
    )
    assert loaded.source_fingerprint == system.source_fingerprint


def test_mismatched_original_and_gari_artifacts_are_rejected(tmp_path):
    system = _toy_gari_system(tmp_path / "a")
    wrong_L = sp.csr_matrix([[0, 1, 0]], dtype=np.uint8)
    other = DecodingSystem.from_matrices(
        system.H, wrong_L, system.priors,
        is_x_detector=system.is_x_detector,
        init_basis=system.init_basis,
        workdir=tmp_path / "b",
    )

    with pytest.raises(ValueError, match="source_fingerprint mismatch"):
        DecodingSystem.from_npz(
            matrices_npz=other.npz_path_for("original"),
            gari_npz=system.npz_path_for("gari"),
        )


def test_legacy_original_is_verified_against_gari_before_pairing(tmp_path):
    system = _toy_gari_system(tmp_path / "source")
    legacy_original = tmp_path / "legacy_original.npz"
    _save_matrices_npz(
        legacy_original, system.H, system.L, system.priors,
    )

    loaded = DecodingSystem.from_npz(
        matrices_npz=legacy_original,
        gari_npz=system.npz_path_for("gari"),
    )
    assert loaded.source_fingerprint == system.source_fingerprint

    wrong_L = sp.csr_matrix([[0, 1, 0]], dtype=np.uint8)
    legacy_wrong = tmp_path / "legacy_wrong.npz"
    _save_matrices_npz(
        legacy_wrong, system.H, wrong_L, system.priors,
    )
    with pytest.raises(ValueError, match="source_fingerprint mismatch"):
        DecodingSystem.from_npz(
            matrices_npz=legacy_wrong,
            gari_npz=system.npz_path_for("gari"),
        )
