# search/sampling/

How matrix entries are parameterized. One file per parameterization.

## Planned files

- `monomial.py` — each entry is a single group element (weight 1) or empty.
- `polynomial.py` — each entry is a sum of group elements (weight ≥ 2).
- `weight_matrix.py` — weight pattern as the primary search variable; ring elements placed afterwards.

## Knobs (shared across parameterizations)

- `include_identity` — whether the group identity `e` is allowed as an entry term.
- `min_element_order` — reject elements of order below this.
- `avoid_same_coset` — (non-abelian only) require entry terms in pairwise-distinct cosets of `[G,G]`. Improves abelianization bound.
- `max_tries` — rejection sampling budget per entry.

## Notes

- For abelian G, `avoid_same_coset=True` rejects every weight-≥2 entry (`[G,G] = {e}`). Always `False` for abelian groups.
- Weight-2 entries `{g₁, g₂}` are bounded by `d(A_bin) ≤ ord(g₁⁻¹g₂)` — see `filters/distance_bounds.py`. Sampling should be aware of this.
