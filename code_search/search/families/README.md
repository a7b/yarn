# search/families/

Fixed structural priors that constrain the sampling space. A family
describes a construction rule and exposes:
- a sampler (or a constraint on the sampling space)
- family-specific filters
- expected `[n, k, d]` ranges

## Files

- `base.py` — the abstract `Family` interface.

No concrete family ships currently; new families (structured multi-row
shapes, bivariate polynomial, abelian specializations, ...) plug in through
the same `Family` interface.
