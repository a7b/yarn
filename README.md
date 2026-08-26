<h1 align="center">yarn</h1>

<p align="center">
  A toolkit for qLDPC processor discovery.
</p>

This repository accompanies the paper [*High-rate qLDPC processors*](https://arxiv.org/abs/2607.28795).

<p align="center">
  <img src="assets/mitten_layouts.svg" width="100%"
       alt="Chip layouts of the [[150,30,10]], [[500,100,16]] and [[975,195,24]] mitten codes compiled using HAL showing only the Manhattan distance 2 couplers">
</p>

<p align="center">
  <em>Three of the eight mitten codes, laid out on chip — data qubits and checks
  tile the plane in |G| three-by-three clusters.</em>
</p>

## Repository structure

```
yarn/
├── sqetch/                    # GPU distance estimator (pip-installable package)
├── code_search/               # LP CSS code search toolkit
├── processor_codes/           # the finalized code suite: check matrices + gadgets
├── scq_hardware_layouts_HAL/  # superconducting chip layouts for the mitten codes
├── SE_cycle_movies/           # atom-array syndrome-extraction animations
└── telescoping-decoder/       # staged circuit-level decoder with GPU and CPU stages
```

`sqetch` and `code_search` are the discovery tooling; `processor_codes` records the codes
that came out of it; `scq_hardware_layouts_HAL` and `SE_cycle_movies` show how those codes
map onto the two hardware modalities we considered — superconducting chips and neutral-atom
arrays. Each directory has its own README with file conventions.

The `telescoping-decoder` package implements the staged decoding pipeline used
to handle circuit-level quantum error correction, combining fast GPU decoding,
CPU fallback stages, and a final exact integer-programming stage. A detailed explanation of 
how we implemented the CUDA kernels for the GPU stages can be found [here.](telescoping-decoder/docs/s1_s2_decoders.html)

## Tooling

- **[`sqetch/`](sqetch/)** — GPU random information-set decoder for
  estimating the minimum distance of CSS quantum codes (pip-installable
  Python package).
- **[`code_search/`](code_search/)** — YAML-driven search toolkit for LP CSS
  codes over finite group algebras F₂[G]: GF(2) and group-ring primitives,
  distance estimators (CPU BP+OSD, GPU sqetch), paired and canonical logical
  bases, and the sample → filter → pair → report pipeline. Start at
  [`code_search/README.md`](code_search/README.md).
- **[`telescoping-decoder/`](telescoping-decoder/)** — staged circuit-level
  decoder with GPU layered BP and relay-BP stages, CPU BP fallbacks, and exact
  integer-programming decoding. Installation, configuration, and examples are documented
  in [`telescoping-decoder/README.md`](telescoping-decoder/README.md).

## Codes and hardware realizations

- **[`processor_codes/`](processor_codes/)** — the finalized rate-1/5 code
  suite: `mitten` (eight codes, [[150,30,10]] through [[975,195,24]], each
  with logical-measurement gadgets and the full-extractor stabilizer
  specification), `structured_mitten` (six codes), and `abelian_poly_LP`
  (one code). File conventions in
  [`processor_codes/README.md`](processor_codes/README.md).
- **[`scq_hardware_layouts_HAL/`](scq_hardware_layouts_HAL/)** —
  superconducting hardware layouts for the eight `mitten` codes, produced with
  HAL: for each code the lowest-hardware-complexity layout we found, kept as
  the full HAL output (per-tier coupler routes, benchmark metrics, settings,
  rendered tier images) alongside a `placements/*.npz` giving every
  Tanner-graph node's position in both the compact and routed frames. Metric
  table and array conventions in
  [`scq_hardware_layouts_HAL/README.md`](scq_hardware_layouts_HAL/README.md).
- **[`SE_cycle_movies/`](SE_cycle_movies/)** — animations of full
  syndrome-extraction cycles for the mitten and structured-mitten codes on
  atom-array layouts (2-AOD, and pipelined 4-AOD).
  
One full 2-AOD syndrome-extraction cycle with 2 pairs of AODs for the [[150,30,10]] mitten code
(C₅×S₃):

https://github.com/user-attachments/assets/e844187e-8fb5-42e9-af6d-c1ea727fedb3

## AI Acknowledgment and Usage

We used Claude to help develop the software in this repository. The AI was used to assist with code generation, documentation, and other programming tasks. Here we document all the ways we used AI in ways that were instrumental to the results in the paper: 
- We had written our GPU distance estimator in JAX in November 2025 but it was not fast enough to let us directly brute force through the search spaces we considered in the paper. We asked Claude if we could improve our distance estimator using sketching and Claude wrote the CUDA kernel that implements the $\textsf{sQetch}$ algorithm.
- We asked Claude if we could lower the HAL hardware complexity by optimizing the placement of the $3 \times 3$ modules on the first tier so for example the non-nearest-neighbor coupler lengths could be reduced. Claude wrote the scripts that performed this optimization. 
- For the CPU C kernels that implement belief propagation (BP) and relay BP we simply ran Claude in a loop where it wrote a kernel, we benchmarked it on a test dataset, and we kept running this in a loop until it was fast enough. 
- For the GPU CUDA kernels we asked Claude to try out all the ideas we had for how to layout BP and relay BP on the GPU. Claude implemented these ideas and wrote the actual CUDA code. Once we had a good implementation, we also ran the same automated Claude loop we ran for the CPU kernels although the throughput increase from this was much smaller compared to the throughput gains made in the first stage where we were talking with Claude and asked Claude to implement our ideas. We will provide a detailed breakdown of how our implementations improved over time in our next update to this repo.

## Citation

When using tools from this repository for research, please cite:

```bibtex
@misc{bhardwaj2026highrateqldpcprocessors,
  title         = {High-rate qLDPC processors},
  author        = {Aditya Bhardwaj and Muzhou Ma and Nadine Meister and Robbie King and Dolev Bluvstein and John Preskill and Madelyn Cain and Qian Xu and Hsin-Yuan Huang},
  year          = {2026},
  eprint        = {2607.28795},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  url           = {https://arxiv.org/abs/2607.28795}
}
```
