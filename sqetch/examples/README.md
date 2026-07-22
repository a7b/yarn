# sqetch examples

Two runnable scripts demonstrating `sqetch.estimate_distance`:

* `quickstart.py` -- the [[7, 1, 3]] Steane code (hardcoded). Recovers
  `d_Z = 3` in a few hundred GPU trials.
* `bb_code.py` -- the [[144, 12, 12]] bivariate-bicycle code. Reads
  `Hx.npy` and `Hz.npy` from a path of your choice (set via the
  `SQETCH_BB_PATH` environment variable) and runs `estimate_distance`
  with `k_sub = 16`.

Both examples require the `gpu` extra (or a manually-installed
PyTorch with CUDA).  See the main `README.md` for platform notes.
