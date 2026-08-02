# yarn

Supporting assets and concrete instances for high-rate qLDPC processor discovery pipeline.

Code base for paper: upcoming

## Contents

- **[`sqetch/`](sqetch/)** — GPU random information-set decoder for
  estimating the minimum distance of CSS quantum codes (pip-installable
  Python package).
- **[`code_search/`](code_search/)** — YAML-driven search toolkit for LP CSS
  codes over finite group algebras F₂[G]: GF(2) and group-ring primitives,
  distance estimators (CPU BP+OSD, GPU sqetch), paired and canonical logical
  bases, and the sample → filter → pair → report pipeline. Start at
  [`code_search/README.md`](code_search/README.md).
- **[`processor_codes/`](processor_codes/)** — the finalized rate-1/5 code
  suite: `mitten` (eight codes, [[150,30,10]] through [[975,195,24]], each
  with logical-measurement gadgets and the full-extractor stabilizer
  specification), `structured_mitten` (six codes), and `abelian_poly_LP`
  (one code). File conventions in
  [`processor_codes/README.md`](processor_codes/README.md).
- **[`SE_cycle_movies/`](SE_cycle_movies/)** — animations of full
  syndrome-extraction cycles for the mitten and structured-mitten codes on
  atom-array layouts (2-AOD, and pipelined 4-AOD).

## Coming soon

- **Telescoping decoder**.
