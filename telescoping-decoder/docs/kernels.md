# Kernels

This page lists the compute kernels, their build paths, and the stage wrappers
that call them. `docs/s1_s2_decoders.html` provides a longer implementation
guide to the CUDA code.

## GPU: S1 and S2

The CUDA source is not in `.cu` files. It is CUDA-C in string literals inside
the kernel modules and compiled by `cupy.RawModule` when a decoder object is
constructed. Each module contains two source strings: `_NONTEMPLATED_KERNELS_SRC`
(compiled as-is) and `_TEMPLATED_KERNELS_SRC` where constants such as the block width
and warp count can be specified using `.format()` before
compilation.

| Module | Class | Stage |
|---|---|---|
| `kernels/s1_layered_bp.py` | `S1LayeredBP` | S1, every shot |
| `kernels/s2_relay_bp_gari.py` | `S2RelayBPGari` | S2 on the GARI system |
| `kernels/s2_relay_bp.py` | `S2RelayBP` | S2 on the original / init-dets systems |

The three decoder modules contain separate copies of some of the same BP logic. When
fixing shared BP behavior, check whether the same change is required in all
three modules.

GARI requires custom convergence, candidate scoring, and row ordering.
`conv_rows` identifies the detector checks used for acceptance. `weight_llr`
scores only the GARI variables that represent the decoded answer, excluding
auxiliary variables. `layers` processes the `U` and `V` consistency rows before
the detector rows so information reaches the answer variables before the
syndrome checks are updated. With full-row convergence, channel weights, and
contiguous layers, its acceptance rules match the plain-system variant.

`Stage1` in `s1.py` and `Stage2` in `s2.py` adapt the public decoder API to
the GPU kernels. They construct the selected kernel, split shots into
GPU-sized batches, and convert Stim-ordered syndromes into the row layout
required by the original, init-detector, or GARI system.

## CPU: S3

### How the S3 C kernels are built

S3 runs its inner belief-propagation loops in compiled C. The repository ships
the C source and compiles it into Linux shared libraries (`.so` files):

| C source | Compiled library | C function | Used by |
|---|---|---|---|
| `checkserial_bp.c` | `checkserial_bp.so` | `checkserial_bp_decode_fast` | S3-A and S3-C |
| `relay_mem_bp.c` | `relay_mem_bp.so` | `relay_mem_bp_decode` | S3-B |

`relay_mem_bp.c` implements check-serial relay BP with a per-iteration memory
term. Python loads both libraries with its standard `ctypes` module. NumPy
arrays are passed directly to the C functions as pointers.

The first time a library is needed, `_c/build.py::ensure_lib(name)` compiles
it with GCC:

```bash
gcc -O3 -ffast-math -shared -fPIC -o <name>.so <name>.c -lm
```

The flags mean:

- `-O3`: enable aggressive compiler optimizations.
- `-ffast-math`: allow faster floating-point operations that can differ
  slightly from strict IEEE arithmetic.
- `-shared`: produce a loadable `.so` library.
- `-fPIC`: generate position-independent code suitable for a shared library.
- `-o <name>.so`: select the output filename.
- `-lm`: link the C math library.

#### Build cache

Compiled libraries are stored under:

```text
~/.cache/telescoping_decoder/
```

The filename includes a hash of the C source, compiler, and compiler flags.
The library is reused while those inputs remain unchanged. Editing the source
or changing the flags produces a different hash and triggers a new build.

Compilation first writes to a temporary file. The completed library is moved
to its cache path with an atomic rename.

To compile both libraries before the first decode, run:

```bash
python -m telescoping_decoder.build
```

This is useful when building a container image or checking that a machine has
a working C compiler.

#### CPU-specific optimization

The build does not use `-march=native` by default. That option lets GCC use
instructions specific to the current CPU, but the resulting library might not
run on another CPU. Together with `-ffast-math`, it can also produce small
machine-dependent differences in floating-point results.

Enable it explicitly with:

```bash
export TELESCOPING_DECODER_MARCH_NATIVE=1
```

#### Selecting a prebuilt library

A deployment can bypass compilation and the cache by providing library paths:

```bash
export CHECKSERIAL_BP_SO=/path/to/checkserial_bp.so
export RELAY_MEM_BP_SO=/path/to/relay_mem_bp.so
```

These variables are useful when a deployment must use a specific binary or
does not include GCC.

#### How Python calls the libraries

`ctypes` needs each C function's name, argument order, and argument types.
This interface is called the application binary interface, or ABI.
`s3.py::_load_lib` and `s3.py::_load_relay_lib` define that internal
interface. Application code should use `TelescopingDecoder`; the low-level
`ctypes` functions are implementation details.

## Calling a GPU kernel directly

```python
from telescoping_decoder.kernels import S1LayeredBP

dec = S1LayeredBP(
    H, channel_llr, obs_matrix=L,
)                                      # H and L: SciPy CSR
out = dec.run(syndromes)            # (B, m) uint8, decoder row order
out["converged"]                    # (B,) bool — full-row convergence
out["correction"]                   # (B, n) uint8
out["obs_pred"]                     # (B, n_obs) uint8, computed on the GPU
out["best_residual"]                # (B,) int32 unsat-row counts
```

`S2RelayBPGari` / `S2RelayBP` take the same shape of input and return those
keys plus `obs_pred` and their relay diagnostics. `obs_pred` reuses the
host-side logical coset already collected for quorum voting and is zero for
deferred shots. Their phase parameters come from `S2RelayPhaseGari` /
`S2RelayPhase` (defaults: `DEFAULT_S2_PHASE_GARI`, `DEFAULT_S2_PHASE`).

Calling a kernel directly bypasses the stage wrappers, including
padding, row permutation, and init-detector gathers, so `H` and `syndromes`
must already be in that kernel's own row order. S1 leaves observable
extraction disabled unless `obs_matrix` is supplied; the S1 stage supplies it
and receives `obs_pred` without re-uploading the returned correction. S2
performs extraction while collecting coset-quorum votes and returns the
accepted coset as `obs_pred`.

Importing `telescoping_decoder.kernels` imports CuPy. The top-level package
therefore loads these modules only when a GPU stage is constructed.

## Stage → kernel map

| Stage | Kernel | Acceptance |
|---|---|---|
| S1 | `S1LayeredBP` (GPU) | all rows on init-dets/original; relevant detector rows on GARI |
| S2 | `S2RelayBPGari` / `S2RelayBP` (GPU) | coset quorum over converged legs |
| S3-A | `checkserial_bp` (CPU) | first converged variant in the ensemble |
| S3-B | `relay_mem_bp` (CPU) | first coset to reach quorum |
| S3-C | `checkserial_bp` (CPU) | first converged result from a sequential BP parameter sweep; no quorum |
| S4 | Gurobi, no kernel | exact solve on the original DEM |

Syndromes stay in detector space (`n_det` bits, stim order) throughout;
GARI zero-padding, row permutation, and init-detector column gathers all
happen at each stage's decode boundary, not in the caller.
