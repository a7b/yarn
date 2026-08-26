import numpy as np
import pytest
import stim

from telescoping_decoder.dem_utils import dem_to_sparse_matrices
from telescoping_decoder.gari import gari_transform


def test_rejects_equal_detector_support_with_different_observables():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.2) D0
    """)

    with pytest.raises(
        ValueError,
        match="identical detector support.*different logical-observable",
    ):
        dem_to_sparse_matrices(dem)


def test_merges_equal_detector_and_observable_support():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.2) D0 L0
    """)

    check_matrix, obs_matrix, priors = dem_to_sparse_matrices(dem)

    np.testing.assert_array_equal(check_matrix.toarray(), [[1]])
    np.testing.assert_array_equal(obs_matrix.toarray(), [[1]])
    np.testing.assert_allclose(priors, [0.1 * 0.8 + 0.2 * 0.9])


def test_gari_rejects_equal_detector_support_with_different_observables():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.2) D0
    """)

    with pytest.raises(
        ValueError,
        match="identical detector support.*different logical-observable",
    ):
        gari_transform(
            dem,
            np.array([True]),
            init_basis="X",
            verify=False,
        )
