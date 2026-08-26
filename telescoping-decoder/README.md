# telescoping-decoder

A staged decoder for circuit-level quantum error correction. Each stage
receives only the shots deferred by the preceding stage. Fast approximate
decoders handle most shots; slower CPU and exact methods handle the remainder.
The GARI path targets correlated XYZ detector models from compatible X/Z
memory experiments; the original-system path supports more general DEMs.

| Stage | What it is | Hardware |
|---|---|---|
| **S1** | fixed-iteration layered BP, every shot | CUDA GPU |
| **S2** | relay-BP with coset-quorum acceptance | CUDA GPU |
| **S3-A** | ensemble of standalone BP variants (first converged wins) | CPU pool |
| **S3-B** | relay-mem BP with first-to-quorum coset voting | CPU pool |
| **S3-C** | sequential sweep of 12 BP configurations; first converged attempt wins | CPU pool |
| **S4** | exact IP certification (Gurobi), sub-DEM → full-DEM ladder | CPU pool, full Gurobi license |

If you are interested in the raw CUDA and C kernels, they can be found in
[telescoping_decoder/kernels/](telescoping_decoder/kernels/) and
[telescoping_decoder/_c/](telescoping_decoder/_c/), respectively. See
[docs/kernels.md](docs/kernels.md) for details on how to build and use them and
[docs/s1_s2_decoders.html](docs/s1_s2_decoders.html) for a detailed guide on
how we wrote the S1 and S2 CUDA kernels for our decoder.

The rest of the package provides the scaffolding needed to run our CUDA and C
kernels in the telescoping decoder. You may want to modify this infrastructure
to suit your own needs, such as running it on a cluster.

## Requirements

- Python >= 3.10 and `gcc`. The S3 kernels are compiled on first use.
- For S1/S2: an NVIDIA GPU with a CUDA 12-capable driver, `cupy-cuda12x`
  and CUDA 12 libraries. The `[gpu]` extra installs CuPy together with CUDA
  runtime wheels; a compatible system toolkit can provide the runtime
  libraries instead. NVRTC compiles the kernels when a GPU decoder is
  constructed. See [Choosing batch sizes](#choosing-batch-sizes) for
  approximate memory requirements.
- For S4: a full Gurobi license. The core package installs `gurobipy`, but
  its bundled license is size-limited and cannot solve these models.
- CPU-only machines run S3/S4 with `config.s1.enabled = config.s2.enabled = False`.

S3 runs its performance-critical BP loops in compiled C. On first use, the
package builds two cached Linux shared libraries (`.so` files) and loads them
directly from Python with `ctypes`. This requires GCC. Run
`python -m telescoping_decoder.build` to compile them in advance. See
[How the S3 C kernels are built](docs/kernels.md#how-the-s3-c-kernels-are-built)
for the compiler flags, cache behavior, and prebuilt-library overrides.

## Install

```bash
pip install /path/to/telescoping-decoder                 # S3 + S4 Python dependency
pip install '/path/to/telescoping-decoder[gpu]'          # all stages + CUDA runtime wheels
pip install -e '/path/to/telescoping-decoder[dev]'       # core + pytest, editable
pip install -e '/path/to/telescoping-decoder[gpu,dev]'   # all stages + pytest, editable
```

The core installation includes S3 and the `gurobipy` Python dependency used by
S4. A full Gurobi license is still required to solve this package's S4 models;
this project does not distribute that external commercial or academic license.
Use `[gpu]` for S1/S2. It installs CuPy and the CUDA runtime components it
uses, while an advanced installation may instead install plain
`cupy-cuda12x` against a compatible system CUDA 12 toolkit. The device check
in [Verify your machine](#verify-your-machine) confirms whether CuPy can open
the GPU; `nvidia-smi` alone is not sufficient.

## Verify your machine

```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'], \
  cupy.cuda.runtime.memGetInfo()[0] // 2**20, 'MiB free')"   # CuPy can open GPU 0
python -m telescoping_decoder.build                          # compile the S3 C kernels
pytest tests/                                                # GPU tests skip without a GPU
python examples/quickstart.py                                # end-to-end on bundled data
```

`quickstart.py` prints the per-stage table and the logical error rate for
1000 shots of the bundled [[150,30,10]] artifacts; it uses the GPU stages
when a GPU is present and falls back to CPU-only when not. It sets
`s4.license_file` explicitly from `$GRB_LICENSE_FILE`, or from a `gurobi.lic`
at the repo root, and prints which one it picked — see
[S4 and Gurobi](#s4-ip-and-gurobi).

## Quickstart

Using the decoder has two separate steps:

1. **Construct the decoder once** from a description of the noise model.
2. **Pass batches of shots to `decode()`** as detector-outcome arrays.

A circuit, DEM, or prepared NPZ file describes the decoding problem; none of
them contains the shots to decode. There is one public shot-input interface:
`decode(syndromes, ...)`.

### 1. Construct the decoder

A few options:

| What you have | Constructor | Notes |
|---|---|---|
| Stim circuit | `from_stim_circuit()` | Recommended when the circuit is available; detector types are inferred from its measurement bases |
| Stim detector error model (DEM) | `from_dem()` | Requires an X/Z detector mask (or a canonical layout) when GARI is enabled |
| Prepared decoder artifacts | `from_npz()` | Skips DEM parsing and the GARI transform on repeated runs |
| Check matrix, observable matrix, and priors | `from_matrices()` | Advanced interface; currently builds the original and optional `init_dets` systems, not GARI |

For example, construct directly from the circuit used by the experiment:

```python
import stim
from telescoping_decoder import TelescopingDecoder

circuit = stim.Circuit.from_file("examples/toy_xz_surface_code_memory.stim")
dec = TelescopingDecoder.from_stim_circuit(circuit, init_basis="X")
```

If prepared artifacts already exist, use this instead:

```python
from telescoping_decoder import TelescopingDecoder

dec = TelescopingDecoder.from_npz(
    gari_npz="stem_gari_matrices.npz",
    matrices_npz="stem_matrices.npz",
)
```

To construct directly from check matrices, use one column per independently
modeled error mechanism. `H` maps errors to detector outcomes, `L` maps the
same errors to observable flips, and `priors[j]` is the probability of error
mechanism `j`:

```python
import numpy as np
from scipy.sparse import csr_matrix
from telescoping_decoder import TelescopingDecoder

# H has shape (number of detectors, number of error mechanisms).
H = csr_matrix(np.array([
    [1, 1, 0],
    [0, 1, 1],
], dtype=np.uint8))

# L has shape (number of observables, number of error mechanisms).
# Its columns refer to exactly the same mechanisms, in the same order as H.
L = csr_matrix(np.array([
    [1, 0, 1],
], dtype=np.uint8))

priors = np.array([0.01, 0.02, 0.01], dtype=np.float64)

# Optional: classifying detector rows enables the init_dets system.
is_x_detector = np.array([True, False])
dec = TelescopingDecoder.from_matrices(
    H,
    L,
    priors,
    is_x_detector=is_x_detector,
    init_basis="X",
)
```

For an error vector `e`, the corresponding syndrome is `H @ e % 2` and its
observable flips are `L @ e % 2`. Dense arrays are also accepted. Omit
`is_x_detector` and `init_basis` if only the original system is needed.
`from_matrices()` currently does not construct GARI. The GARI implementation
is wired through the DEM constructor even though the transformation itself
uses `H`, `L`, priors, the X/Z detector mask, and the initialization basis.

### 2. Supply shots

Both simulated and experimental shots use the same `decode()` method. For a
Stim simulation, the sampler can also return the actual observable flips, so
the decoder can report a logical error rate:

```python
detectors, actual_observables = circuit.compile_detector_sampler(seed=1).sample(
    shots=1_000,
    separate_observables=True,
)

with dec:
    result = dec.decode(detectors, true_obs=actual_observables)

print(result.obs_pred)  # predicted observable flips for every shot
print(result.stage)     # stage that accepted each shot
print(result.summary()) # stage counts and logical error rate
```

For experimental data instead, pass the detector outcomes produced by the
hardware pipeline. Ground-truth observable flips are normally unavailable,
so omit `true_obs`:

```python
hardware_syndromes = read_detector_outcomes()

with dec:
    result = dec.decode(hardware_syndromes)

corrections = result.obs_pred
```

The decoder does not sample Stim or communicate with hardware itself. The
caller is responsible for producing the detector-outcome array in Stim
detector order.

## Using your own Stim circuit

The shortest complete circuit-to-result workflow is:

```python
import stim
from telescoping_decoder import TelescopeConfig, TelescopingDecoder

circuit = stim.Circuit.from_file("examples/toy_xz_surface_code_memory.stim")

# These settings make the example portable to a CPU-only machine. Adjust
# them after installing the [gpu] extra on a machine with a CUDA GPU.
cfg = TelescopeConfig()
cfg.s1.enabled = False
cfg.s2.enabled = False
cfg.s3.n_procs = 1
cfg.s4.enabled = False

detectors, actual_observables = circuit.compile_detector_sampler(seed=1).sample(
    shots=100,
    separate_observables=True,
)

with TelescopingDecoder.from_stim_circuit(
    circuit,
    init_basis="X",
    config=cfg,
) as decoder:
    result = decoder.decode(detectors, true_obs=actual_observables)

print(result.summary())
```

The checked-in `examples/toy_xz_surface_code_memory.stim` is a small
distance-3, three-round rotated surface-code X-memory experiment generated by
Stim. Replace it with the noisy circuit that produces your experiment's
detector outcomes.
`init_basis` describes the memory experiment, not an internal BP setting:

- `init_basis="X"` means logical |+> preparation and X-basis readout. The
  tracked logical-X observables are flipped by Z components and detected by
  X-type detectors.
- `init_basis="Z"` means logical |0> preparation and Z-basis readout. The
  tracked logical-Z observables are flipped by X components and detected by
  Z-type detectors.

With the default `use_gari=True`, the circuit must satisfy all of the
following:

- It contains noisy error mechanisms plus Stim `DETECTOR` and
  `OBSERVABLE_INCLUDE` annotations. Detector order defines the column order
  expected later by `decode()`.
- Each detector's measurement-record targets must resolve to one CSS family.
  Direct `MX`/`MRX` records are X-type and ordinary `M`/`MR` records are
  Z-type. The inference also recognizes Stim's Hadamard-wrapped M/MR ancillas
  as X-stabilizer measurements. Mixed-basis detectors are not supported. MPP,
  Y-basis, heralded, and other measurement-producing gates may exist in the
  circuit, but a GARI detector cannot reference their results.
- The resulting correlated XYZ DEM has GARI's pure-component structure:
  every mixed `eY` error column has matching pure `eZ` and `eX` columns.
  Detector-silent error columns and observables on the wrong pure side are
  rejected.

These requirements are checked while constructing the decoder. An
incompatible circuit raises a descriptive error; the constructor never
silently changes the decoding problem.

For a circuit outside the GARI assumptions, construct the original-system
decoder from its DEM instead:

```python
from telescoping_decoder import (
    TelescopeConfig,
    TelescopingDecoder,
    circuit_to_dem,
)

cfg = TelescopeConfig(use_gari=False)
dem = circuit_to_dem(circuit)
with TelescopingDecoder.from_dem(dem, config=cfg) as decoder:
    result = decoder.decode(detectors)
```

This path does not require an X/Z detector mask. With no mask or initialization
basis, `system="auto"` selects the original system for S1, S2, and S3. S1 and
S2 still require the `[gpu]` extra and a CUDA GPU; disable them as in the
portable example when running CPU-only.

Three runnable end-to-end scripts ship in `examples/`:

- `examples/quickstart.py` — the shortest complete path, 1000 shots.
- `examples/from_stim_circuit.py` — build directly from the checked-in Stim
  circuit, sample it, and decode on a CPU-only machine.
- `examples/memory_experiment.py` — a memory experiment: sample N shots,
  decode, report the stage table and the LER with a 95% confidence
  interval (`python examples/memory_experiment.py --shots 20000`).

## Choosing batch sizes

`s1.shots_per_batch` and `s2.batch_size` control memory use and throughput.
S1 is deterministic across batch layouts. S2 is repeatable for the same
input order and batching configuration, but its randomized gamma schedule
is seeded from each batch's position. Changing the S2 batch size or batch
composition can therefore change an S2 result. Approximate storage per shot
is `~(nnz + 6·n_cols) × 4 B` for S2 and about one third of that for S1. For
the bundled [[150,30,10]] artifacts:

| System | H (rows × cols, nnz) | S1 | S2 |
|---|---|---|---|
| `init_dets` | 660 × 9 600, 60 840 | 0.36 MB/shot | 0.46 MB/shot |
| `gari` | 19 575 × 94 620, 269 250 | 2.3 MB/shot | 3.3 MB/shot |
| `original` | 1 200 × 76 245, 769 790 | 4.0 MB/shot | 4.8 MB/shot |

The defaults (`s1.shots_per_batch=2048`, `s2.batch_size=2048`) target a GPU
with ample memory. The following values are starting points, not hard memory
requirements. S1 normally uses `init_dets`; S2 normally uses `gari`.

| VRAM | `s1.shots_per_batch` | `s2.batch_size` |
|---|---|---|
| 4 GB | 2048 | 256 |
| 8 GB | 4096 | 1024 |
| 16 GB | 8192 | 2048 |
| 24 GB+ | 8192 | 4096 |

```python
from telescoping_decoder import TelescopeConfig

cfg = TelescopeConfig()
cfg.s2.batch_size = 256          # starting point for a 4 GB GPU
cfg.s2.shots_per_batch = 256     # the outer driver slice; keep >= batch_size
```

For `cupy.cuda.memory.OutOfMemoryError`, reduce `s2.batch_size` first because
S2 is normally the larger consumer. Other processes and cached allocations
also affect the available memory.

## Shot input

Each call to `dec.decode(syndromes, true_obs=None, shot_ids=None)` accepts a
two-dimensional array containing any number of shots. The array may be the
caller's entire dataset: S1 and S2 automatically divide it into internal GPU
batches using their configured batch sizes, and S3 and S4 distribute the
shots that reach them across their worker pools. Manually splitting the data
across multiple `decode()` calls is useful for progress reporting or limiting
host-memory use, but is not required for GPU batching. Because S2's randomized
schedule depends on batch position, changing call boundaries can change S2
results; keep them fixed if you care about reproducibility.

| Argument | Shape | Meaning |
|---|---|---|
| `syndromes` | `(B, n_det)` | Required `uint8` or Boolean detector outcomes. Each row is one shot, each column is one detector, and `1` means that detector fired. Columns must be in Stim detector order. |
| `true_obs` | `(B, n_obs)` | Optional actual observable flips, normally available only in simulation. They are used to score predictions and never influence decoding. Passing them enables `result.le` and `result.ler`. |
| `shot_ids` | `(B,)` | Optional unique `uint64` identifiers used for reproducible S3 random seeds. Most users can omit them. |

Here `B` is the number of shots in this call, `n_det` is the number of
detectors in the model, and `n_obs` is the number of logical observables. For
example, 1,000 shots from a model with 1,200 detectors produce a syndrome
array with shape `(1000, 1200)`.

All GARI rows, padding, and detector subsets are internal. Do not transform
or pad the syndrome array before calling `decode()`.

By default, `shot_ids` is `arange(B)`. Supply stable IDs when the same data
may be divided into different calls and you want S3 results to remain tied to
the original shots. S2 instead derives its randomized schedule from batch
position, so changing S2 batching can still change its result.

## Prepared decoder artifacts

The NPZ files passed to `from_npz()` contain the decoder model, not shot
data. During preparation and stage construction, the package materializes
the system artifacts it needs in the decoder work directory so worker
processes can load them efficiently. Loading existing artifacts with
`from_npz()` avoids repeating DEM parsing and the GARI transform.

```python
dec = TelescopingDecoder.from_npz(
    matrices_npz="model_matrices.npz",       # original system
    gari_npz="model_gari_matrices.npz",     # transformed GARI system
)
```

Pass both files for the complete decoder. With only `matrices_npz`, the
original system is available but GARI and `init_dets` are not. With only
`gari_npz`, GARI is available but the original system and S4 are not. The
files describe the model; shots are still supplied separately to `decode()`.

### Original-system NPZ format

`matrices_npz` is a NumPy archive containing binary CSR matrices `H` and `L`,
one probability per shared matrix column, and two degree bounds. All keys in
this table are required unless marked optional:

| Key | Dtype and shape | Meaning |
|---|---|---|
| `h_data` | `uint8`, `(nnz_H,)` | Nonzero values of binary `H` |
| `h_indices` | `int32`, `(nnz_H,)` | CSR column indices of `H` |
| `h_indptr` | `int32`, `(n_det + 1,)` | CSR row pointers of `H` |
| `h_shape` | `int32`, `(2,)` | `[n_det, n_mechanisms]` |
| `l_data` | `uint8`, `(nnz_L,)` | Nonzero values of binary `L` |
| `l_indices` | `int32`, `(nnz_L,)` | CSR column indices of `L` |
| `l_indptr` | `int32`, `(n_obs + 1,)` | CSR row pointers of `L` |
| `l_shape` | `int32`, `(2,)` | `[n_obs, n_mechanisms]` |
| `probs` | `float64`, `(n_mechanisms,)` | Independent probability of each error mechanism |
| `dc_pad` | scalar `int32` | Maximum number of nonzeros in any row of `H` |
| `dv_pad` | scalar `int32` | Maximum number of nonzeros in any column of `H` |
| `source_fingerprint` | scalar string, optional | Model identity used to prevent pairing unrelated original and GARI files |

Rows of `H` are detectors in the same order as the syndrome columns passed
to `decode()`. Rows of `L` are logical observables. Column `j` of `H`, column
`j` of `L`, and `probs[j]` must all describe the same independent error
mechanism. In other words, for an error vector `e`:

```text
syndrome        = H @ e mod 2
observable_flip = L @ e mod 2
```

This helper writes a valid original-system archive from dense or sparse
matrices:

```python
import numpy as np
from scipy.sparse import csr_matrix


def save_original_npz(path, H, L, priors):
    H = csr_matrix(H)
    L = csr_matrix(L)
    priors = np.asarray(priors, dtype=np.float64)

    if H.shape[1] != L.shape[1] or priors.shape != (H.shape[1],):
        raise ValueError("H, L, and priors must share the mechanism axis")
    if np.any(H.data != 1) or np.any(L.data != 1):
        raise ValueError("H and L must be binary")
    H = H.astype(np.uint8)
    L = L.astype(np.uint8)

    np.savez(
        path,
        h_data=H.data.astype(np.uint8),
        h_indices=H.indices.astype(np.int32),
        h_indptr=H.indptr.astype(np.int32),
        h_shape=np.asarray(H.shape, dtype=np.int32),
        l_data=L.data.astype(np.uint8),
        l_indices=L.indices.astype(np.int32),
        l_indptr=L.indptr.astype(np.int32),
        l_shape=np.asarray(L.shape, dtype=np.int32),
        probs=priors,
        dc_pad=np.int32(np.diff(H.indptr).max(initial=0)),
        dv_pad=np.int32(np.diff(H.tocsc().indptr).max(initial=0)),
    )
```

For example, `save_original_npz("model_matrices.npz", H, L, priors)` writes
a file accepted as `matrices_npz`. A hand-written original archive does not
need `source_fingerprint`. When it is paired with a GARI archive, the loader
reconstructs and checks the identity from the matrices and GARI metadata.

### GARI NPZ format

`gari_npz` contains the same required CSR keys (`h_*`, `l_*`, `probs`,
`dc_pad`, and `dv_pad`), but they describe the transformed GARI matrices
`Hbar` and `Lbar`. It also requires every key below:

| Key | Dtype and shape | Meaning |
|---|---|---|
| `gari_n_detectors` | scalar `int64` | Width of the unpadded input syndrome |
| `gari_is_x_detector` | `bool`, `(gari_n_detectors,)` | Detector family mask; `True` means X-type |
| `gari_col_block_bounds` | `int64`, `(6,)` | Half-open boundaries for `eZ`, `eX`, `eY`, `ebarZ`, `ebarX`, in that order |
| `gari_row_block_bounds` | `int64`, `(4,)` | Half-open boundaries for detector, `U`, and `V` rows, in that order |
| `gari_relevant_rows` | `int64`, `(n_relevant,)` | Original detector rows used by the GARI acceptance test |
| `gari_relevant_priors` | `float64`, `(n_answer,)` | Combined priors for the answer block |
| `gari_layers` | `int64`, `(3, 2)` | Half-open row ranges for the `U`, `V`, and detector layers |
| `gari_u_map` | `int64`, `(n_eY,)` | Map from each `eY` column to its relative `eZ` partner |
| `gari_v_map` | `int64`, `(n_eY,)` | Map from each `eY` column to its relative `eX` partner |
| `gari_init_basis` | scalar string | `"X"` or `"Z"` |
| `gari_answer_block` | scalar string | `"ebarZ"` for X initialization or `"ebarX"` for Z initialization |
| `source_fingerprint` | scalar string, optional | Identity shared with the corresponding original-system archive |

The transformed matrix row order must be `[detectors, U, V]`, and its column
order must be `[eZ, eX, eY, ebarZ, ebarX]`. The block-bound arrays give the
six column boundaries and four row boundaries, including the initial zero
and final matrix dimension.

More explicitly, let `nZ`, `nX`, and `nY` be the sizes of the `eZ`, `eX`, and
`eY` blocks. Then:

```text
Hbar shape = (n_det + nZ + nX, 2*nZ + 2*nX + nY)
Lbar shape = (n_obs,             2*nZ + 2*nX + nY)

gari_col_block_bounds = [
    0,
    nZ,
    nZ + nX,
    nZ + nX + nY,
    2*nZ + nX + nY,
    2*nZ + 2*nX + nY,
]

gari_row_block_bounds = [
    0,
    n_det,
    n_det + nZ,
    n_det + nZ + nX,
]

gari_layers = [
    [n_det,       n_det + nZ],
    [n_det + nZ,  n_det + nZ + nX],
    [0,            n_det],
]
```

The first `nZ + nX + nY` entries of `probs` are the physical `eZ`, `eX`, and
`eY` mechanism probabilities. The remaining `nZ + nX` auxiliary `ebar`
entries are `0.5`. For X initialization, `n_answer = nZ`, the answer block is
`ebarZ`, and the relevant rows are X-type detector rows. For Z initialization,
`n_answer = nX`, the answer block is `ebarX`, and the relevant rows are Z-type
detector rows.

Constructing this metadata by hand is error-prone. When starting from a DEM,
use the package's public writer so the transform and metadata stay
consistent:

```python
from telescoping_decoder import gari_transform, save_gari_npz

gari = gari_transform(
    dem,
    is_x_detector,
    init_basis="X",
)
save_gari_npz("model_gari_matrices.npz", gari)
```

When both NPZ files contain fingerprints, `from_npz()` requires them to
match. Legacy or hand-written files may omit the field; the loader then
reconstructs the identity before pairing the files.

## Results

`decode()` returns a `DecodeResult` (struct of arrays, one entry per shot):

| Field | Meaning |
|---|---|
| `obs_pred` | `(B, n_obs)` predicted observable flips; zeros for NC shots |
| `stage` | `Stage` code of the stage that accepted the shot (`Stage.NC` = none) |
| `label` | winning variant / status string, e.g. `S3B_p1.0`, `IP_full_OPTIMAL` |
| `converged` | `stage != Stage.NC` |
| `le` | `(B,)` logical error flags — only when `true_obs` was passed |
| `ler`, `counts_by_stage()`, `summary()` | aggregate views |
| `diagnostics` | per-stage dict: system used, shots in/accepted, wall time |

An NC shot was deferred by every enabled stage. It keeps the zero prediction
and is scored against that prediction when `true_obs` is supplied. Report
`diagnostics["n_nc"]` with the LER.

## Decode systems

Every BP stage independently selects the type of DEM it decodes via
`config.sN.system`:

- `"auto"` (default) — check which DEM is available and use the first one the
  decoder can build. **S1: `init_dets` → `gari` → `original`.** S2/S3:
  `gari` → `original`.
- `"gari"` — the GARI-transformed DEM (graph augmentation and rewiring
  for inference, arXiv:2510.14060) with relevant-half acceptance. Needs the
  X/Z detector-type mask (`from_stim_circuit` derives it; `from_dem` takes
  `is_x_detector=` or `canonical_layout=(n_x, n_z, rounds)`).
- `"original"` — the correlated XYZ DEM as-is, full-row acceptance.
- `"init_dets"` — the init-basis-detectors-only system derived from the
  original matrices (needs the mask + `init_basis`).

An explicitly named system is strict: if the decoder cannot build it you get
an error, never a silent fallback. Only `"auto"` walks the chain.

For the bundled benchmark, the init-detector system is smaller than GARI
(660×9600 versus 19575×94620) and accepted more shots in S1 (96.5% versus
84.3% at p=0.001). S4 always solves the
**original** DEM; `config.s4.init_dets_only=True` restricts it to the init
family (one exact solve, ~16x faster, validated LER-equivalent).

The shipped configuration uses init-dets for S1 and GARI for S2/S3. The
repository benchmarks also cover all-GARI, init-dets throughout, and the
original system throughout. Other mixed configurations have not been
benchmarked here.

## Configuration

```python
from telescoping_decoder import TelescopeConfig

cfg = TelescopeConfig()          # shipped defaults
cfg.s2.batch_size = 256          # reduce S2 memory use
cfg.s2.quorum = 2                # benchmarked value for system="original"
cfg.s3.n_procs = 32
cfg.s4.license_file = "/path/to/gurobi.lic"
dec = TelescopingDecoder.from_npz(..., config=cfg)
```

The shipped values, including the S3 variant tables, are those used for the
bundled benchmark. Per-system caveats are documented on the configuration
fields; for example, the benchmark used `s2.quorum=3` on GARI and 2 on the
original system.

**S1 knobs come as a set.** `config.S1_PRESETS` holds one row per decode
system, and `S1Config` ships the `init_dets` row (`k=32`, `n_iters=10`,
`hybrid_sp_iters=5`). Rows are not interchangeable: `hybrid_sp_iters=5` on
the GARI graph lands the min-sum tail on an oscillating phase that accepts
19.3% of shots, where `>=6` accepts ~62%. If S1 resolves to a system other
than `init_dets` while the knobs are untouched defaults, the decoder swaps
in that system's row and warns; setting any of them yourself disables the
swap.

## S4 (IP) and Gurobi

The core installation includes `gurobipy`; no separate IP install extra is
needed. With `s4.enabled="auto"` (default), S4 activates when that dependency
is importable. Set `s4.enabled=False` when you deliberately do not want to run
S4. Set `s4.license_file` (becomes `GRB_LICENSE_FILE` in each worker) or rely
on Gurobi's standard license lookup. The pool starts with `forkserver`, so an
exported `GRB_LICENSE_FILE` reaches the workers too.

The models here are far larger than the 2000-variable cap of the restricted
license that ships with `pip install gurobipy`, so S4 needs a full license.
Without a usable license, shots are labeled `IP_no_license` and
`diagnostics["s4"]["status_hist"]` naming the cause (`LICENSE_TOO_SMALL`,
`LICENSE_EXPIRED`, `LICENSE_MISSING`) and a warning pointing at
`s4.license_file`. `IP_giveup_uncertified` instead means that the solver ran
out of its configured budget without an accepted incumbent. Both statuses
leave the shot deferred.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `cudaErrorNoDevice` though `nvidia-smi` sees the GPU | verify the CuPy/CUDA installation; install the `[gpu]` extra or a compatible CUDA 12 toolkit |
| `cupy.cuda.memory.OutOfMemoryError` | batch too big for free VRAM — see [Choosing batch sizes](#choosing-batch-sizes) |
| `RuntimeError: S1/S2 need a CUDA GPU with cupy` | no GPU/cupy; set `config.s1.enabled = config.s2.enabled = False` |
| Worker bootstrap fails / `__main__` errors from the S3 pool | `forkserver`/`spawn` need a real main module: run from a file, not piped stdin (or set `cfg.s3.mp_start_method="fork"`, before any CUDA work) |
| `warning: s1.system resolved to 'gari' ...` | S1 couldn't build `init_dets` (no X/Z mask + init basis); pass them, or accept the GARI preset |
| Shots labeled `IP_no_license` | Gurobi license missing/expired/size-limited — see [S4 and Gurobi](#s4-ip-and-gurobi) |
| `s1.system='init_dets' needs ... the X/Z detector mask` | construct via `from_stim_circuit`, or pass `is_x_detector=`/`canonical_layout=` |
| Results differ from an earlier run | check `base_seed`, `shot_ids`, stage settings, input order, and S2 batch settings; pool size does not affect S3 results |

## Documentation

- **`docs/s1_s2_decoders.html`** — an implementation guide to the S1 and S2 GPU decoders. It explains the underlying BP algorithms and how we wrote the CUDA kernels that implement them.
- **`docs/kernels.md`** — where each kernel lives, how the GPU kernels are
  JIT-compiled and the CPU kernels built and cached, and which stage calls
  which.

## Tests

```bash
pip install -e '.[dev]'
pytest tests/          # GPU tests skip automatically without a CUDA device
```

The suite and examples read the bundled [[150,30,10]] artifacts from
`tests/data/` (~16 MB).
`scripts/run_gpu_tests_modal.py` runs the whole suite on a cloud H100 if you
have no local GPU (`modal run scripts/run_gpu_tests_modal.py`); the package
itself has no Modal dependency.
