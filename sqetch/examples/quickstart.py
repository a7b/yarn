"""Quickstart: estimate the Z-distance of the [[7,1,3]] Steane code.

The Steane code is the smallest non-trivial CSS code.  Here we use it
as a sanity demonstration: run a few hundred GPU trials and recover
d_Z = 3.
"""

import numpy as np

from sqetch import estimate_distance


# Steane code: H_X = H_Z (the [7,4,3] Hamming code's parity-check matrix).
H_X = np.array(
    [
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1],
    ],
    dtype=np.uint8,
)
H_Z = H_X.copy()

# One X-logical and one Z-logical (the all-ones vector lies in both
# ker(H_X) and ker(H_Z) and is not in the rowspan of either, so it is
# a valid representative of the single logical qubit).
L_X = np.ones((1, 7), dtype=np.uint8)
L_Z = np.ones((1, 7), dtype=np.uint8)


def main() -> None:
    result = estimate_distance(
        H_X,
        L_X,
        num_trials=500,
        d_target=3,
        k_sub=3,
        seed=0,
    )
    print(f"Steane d_Z bound: {result.best_weight}")
    print(f"trials run:       {result.trials_run}")
    print(f"early-stop found: {result.found}")
    print(f"raw iter/s:       {result.raw_iter_per_sec:.0f}")


if __name__ == "__main__":
    main()
