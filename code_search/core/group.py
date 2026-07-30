"""Group setup, ring arithmetic over F2[G], and binary representations.

All Python-side indices are 0-based; GAP uses 1-based internally.

Ring elements are stored as sorted tuples of **distinct** 0-based group-element
indices — i.e. just the support, since coefficients are 0/1 in F2[G].

All public functions accept any iterable of ints on input and produce output in
canonical form. Repeated entries on input are reduced mod 2 (even count → drop,
odd count → keep) before / during the operation, so callers don't need to
pre-canonicalize; the helper `canonicalize` is exposed for that purpose.

GroupData auto-detects direct-product structure: if the group was constructed
via GAP's `DirectProduct(...)` (so `HasDirectProductInfo(G)` is true with at
least two factors), two extra Python tables are built:
  - decompose_table[g_idx]   = (i_1, …, i_k)  — factor indices
  - compose_table[(i_1, …, i_k)] = g_idx       — inverse mapping
Plus a Kronecker-order permutation `kron_perm` that maps GAP's element order to
the natural Kronecker order over factor indices. Built tables are verified at
the binary level via `_verify_kron_factorization` (a sample of L_G[g] is
compared to P · ⊗_i L_factor_i[g_i] · Pᵀ).

To use the decomposition feature, construct the group via `DirectProduct(...)`.
For groups without `DirectProductInfo`, `decompose`/`compose` raise.
"""

import itertools
from math import prod as _prod
from typing import Iterable, Optional, Sequence

import numpy as np
from gappy import gap


# ─────────────────────────────────────────────────────────────────
# 0. Python ↔ GAP indexing helpers
#
# All Python-side indices are 0-based. GAP is 1-based. Every Python ↔ GAP
# boundary should go through these helpers — never write `i + 1` or `j - 1`
# inline.
# ─────────────────────────────────────────────────────────────────


def to_gap(i: int) -> int:
    """Convert a 0-based Python index to a 1-based GAP position."""
    return i + 1


def from_gap(j: int) -> int:
    """Convert a 1-based GAP position to a 0-based Python index."""
    return j - 1


# ─────────────────────────────────────────────────────────────────
# 1. Group setup
# ─────────────────────────────────────────────────────────────────


_instance_counter = itertools.count()


class GroupData:
    """Precomputed group data: elements, multiplication table, inverse table.

    Auto-detects direct-product decomposition if the group has
    `DirectProductInfo` (i.e. was constructed via `DirectProduct(...)`) with
    at least two factors.

    All Python-side indices are 0-based.
    GAP calls use 1-based indices internally (converted via to_gap/from_gap).
    """

    def __init__(self, gap_expr: str, *, _cleanup: bool = True):
        # Per-instance GAP variable names so nested factor construction doesn't
        # clobber the parent's `_G`, `_elems`, etc.
        inst_id = next(_instance_counter)
        self._inst_id = inst_id
        self._G_var = f"_G_{inst_id}"
        self._elems_var = f"_elems_{inst_id}"
        self._comm_var = f"_comm_{inst_id}"
        self._comm_elems_var = f"_comm_elems_{inst_id}"
        self._gap_temp_vars: list = []   # extra vars to clean up

        # Initialize cleanup-relevant attrs BEFORE any GAP eval can raise, so
        # _cleanup_gap_state can run safely on a half-constructed instance.
        self.factors = None
        self.decompose_table = None
        self.compose_table = None
        self.kron_perm = None

        try:
            self._build(gap_expr)
        except Exception:
            # Any failure during construction leaks GAP state otherwise:
            # main vars (_G, _elems, _comm, _comm_elems), temp vars
            # (_dpfactor_*, _proj_*, _g_tmp_*), and any partially-built
            # factor GroupData. _cleanup_gap_state handles all of these.
            self._cleanup_gap_state()
            raise

        if _cleanup:
            self._cleanup_gap_state()

    def _build(self, gap_expr: str):
        """Heavy construction body. Split out from __init__ so the caller can
        wrap it in try/except for cleanup-on-error discipline. Operates entirely
        on self; raises propagate up to __init__."""
        gap.eval(f"{self._G_var} := {gap_expr};")
        gap.eval(f"{self._elems_var} := Elements({self._G_var});")

        self.n = int(gap.eval(f"Size({self._G_var})"))
        self.gap_expr = gap_expr
        self.is_abelian = bool(gap.eval(f"IsAbelian({self._G_var})"))
        self.structure = str(gap.eval(f"StructureDescription({self._G_var})"))

        # Human-readable element strings (0-based).
        self.elem_strs = [
            str(gap.eval(f"{self._elems_var}[{to_gap(i)}]"))
            for i in range(self.n)
        ]

        # mult[i][j] = 0-based index of g_i * g_j
        self.mult = [
            [
                from_gap(int(gap.eval(
                    f"Position({self._elems_var}, "
                    f"{self._elems_var}[{to_gap(i)}]*{self._elems_var}[{to_gap(j)}])"
                )))
                for j in range(self.n)
            ]
            for i in range(self.n)
        ]

        # inv[i] = 0-based index of g_i^{-1}
        self.inv = [
            from_gap(int(gap.eval(
                f"Position({self._elems_var}, "
                f"Inverse({self._elems_var}[{to_gap(i)}]))"
            )))
            for i in range(self.n)
        ]

        # identity element index (0-based); satisfies mult[e][h] == h for all h.
        # Enforced convention: GAP's Elements(G) must put identity at GAP
        # position 1 (= Python index 0). All standard GAP groups satisfy this.
        self.identity = from_gap(int(gap.eval(
            f"Position({self._elems_var}, Identity({self._G_var}))"
        )))
        if self.identity != 0:
            raise RuntimeError(
                f"GAP's Elements({gap_expr}) does not put identity at index 0 "
                f"(got identity at index {self.identity}). The package "
                f"convention requires identity at index 0. File an issue if "
                f"you hit this with a standard group."
            )

        # Commutator subgroup [G,G] = DerivedSubgroup(G).
        gap.eval(f"{self._comm_var} := DerivedSubgroup({self._G_var});")
        gap.eval(f"{self._comm_elems_var} := Elements({self._comm_var});")
        comm_size = int(gap.eval(f"Size({self._comm_var})"))
        self.commutator = frozenset(
            from_gap(int(gap.eval(
                f"Position({self._elems_var}, "
                f"{self._comm_elems_var}[{to_gap(i)}])"
            )))
            for i in range(comm_size)
        )
        self.commutator_order = comm_size
        self.abelianization_order = self.n // comm_size

        # coset_id[g] = canonical representative of the left coset g·[G,G].
        coset_id = [-1] * self.n
        for g in range(self.n):
            if coset_id[g] == -1:
                coset = [self.mult[g][c] for c in self.commutator]
                rep = min(coset)
                for h in coset:
                    coset_id[h] = rep
        self.coset_id = coset_id

        # Direct product setup (auto-detect via DirectProductInfo).
        factor_gap_exprs = self._auto_detect_factors()
        if factor_gap_exprs:
            self._setup_direct_product(factor_gap_exprs)
        # else: factors/decompose_table/compose_table/kron_perm stay None.

    # ─── cleanup ───────────────────────────────────────────────────

    def _cleanup_gap_state(self):
        """Unbind this instance's GAP variables. Idempotent; recursively
        cleans up factors first."""
        if self.factors is not None:
            for f in self.factors:
                f._cleanup_gap_state()
        for var in (self._G_var, self._elems_var,
                    self._comm_var, self._comm_elems_var):
            gap.eval(f"Unbind({var});")
        for v in self._gap_temp_vars:
            gap.eval(f"Unbind({v});")
        self._gap_temp_vars = []

    # ─── direct product ────────────────────────────────────────────

    def _auto_detect_factors(self) -> Optional[list]:
        """Auto-detect direct-product factors via DirectProductInfo.

        Returns a list of GAP variable names binding factor subgroups (each
        recorded in self._gap_temp_vars for cleanup), or None if no usable
        decomposition exists.
        """
        if not bool(gap.eval(f"HasDirectProductInfo({self._G_var})")):
            return None
        num_factors = int(gap.eval(
            f"Length(DirectProductInfo({self._G_var}).groups)"
        ))
        if num_factors < 2:
            return None
        factor_exprs = []
        for fi in range(num_factors):
            var = f"_dpfactor_{self._inst_id}_{fi}"
            gap.eval(
                f"{var} := DirectProductInfo({self._G_var}).groups[{to_gap(fi)}];"
            )
            factor_exprs.append(var)
            self._gap_temp_vars.append(var)
        return factor_exprs

    def _setup_direct_product(self, factor_gap_exprs: list):
        """Construct factor GroupData and the decompose/compose tables.

        Verifies binary-level consistency: L_G[g] = P · ⊗_i L_factor_i[g_i] · Pᵀ
        for a sample of elements, where P is the permutation matrix associated
        with `kron_perm` (gap order → kron order).

        Sets:
            self.factors        — list of factor GroupData (left alive; the
                                  outer __init__ cleanup handles them).
            self.decompose_table — list of length n, factor-index tuples.
            self.compose_table   — dict mapping tuple → g_idx.
            self.kron_perm       — list of length n, gap_idx → kron_idx.

        Any raise propagates to __init__'s try/except, which calls
        _cleanup_gap_state to release this instance's GAP state plus any
        partially-built factors recorded in self.factors.
        """
        # Build factors incrementally so a mid-list failure leaves the
        # successfully-built ones reachable via self.factors for cleanup.
        # `_cleanup=False` keeps each factor's GAP state alive for the
        # projection-image lookups below.
        self.factors = []
        for expr in factor_gap_exprs:
            self.factors.append(GroupData(expr, _cleanup=False))

        if _prod(f.n for f in self.factors) != self.n:
            sizes = [f.n for f in self.factors]
            raise ValueError(
                f"Factor sizes {sizes} do not multiply to |G| = {self.n}. "
                f"Factors do not match the parent group."
            )

        k = len(self.factors)
        proj_vars = []
        for i in range(k):
            pv = f"_proj_{self._inst_id}_{i}"
            gap.eval(f"{pv} := Projection({self._G_var}, {to_gap(i)});")
            proj_vars.append(pv)
            self._gap_temp_vars.append(pv)

        tmp_g = f"_g_tmp_{self._inst_id}"
        self._gap_temp_vars.append(tmp_g)

        # Cache the GAP `fail` sentinel so we can detect a missing projection
        # image without crashing on int(fail) — see GapBoolean handling.
        gap_fail = gap.eval("fail")

        self.decompose_table = []
        self.compose_table = {}
        for g_idx in range(self.n):
            gap.eval(f"{tmp_g} := {self._elems_var}[{to_gap(g_idx)}];")
            indices = []
            for i in range(k):
                pos_result = gap.eval(
                    f"Position({self.factors[i]._elems_var}, "
                    f"Image({proj_vars[i]}, {tmp_g}))"
                )
                if pos_result == gap_fail:
                    raise RuntimeError(
                        f"Element {g_idx} ({self.elem_strs[g_idx]}): "
                        f"projection to factor {i} not found in factor's "
                        f"Elements list. Factor {i} does not match the "
                        f"original DirectProduct factor."
                    )
                indices.append(from_gap(int(pos_result)))
            indices = tuple(indices)
            self.decompose_table.append(indices)
            self.compose_table[indices] = g_idx

        if len(self.compose_table) != self.n:
            raise RuntimeError(
                f"Decomposition is not a bijection: "
                f"{len(self.compose_table)} distinct tuples for {self.n} "
                f"elements. Inconsistent factor data."
            )

        # Kronecker-order permutation: kron_idx is lex over factor indices.
        # kron_perm[gap_idx] = kron_idx.
        self.kron_perm = []
        for g_idx in range(self.n):
            indices = self.decompose_table[g_idx]
            kron_idx = 0
            product = 1
            for i in range(k - 1, -1, -1):
                kron_idx += indices[i] * product
                product *= self.factors[i].n
            self.kron_perm.append(kron_idx)

        # Binary-level verification (uses left_rep, which only reads
        # self.mult / self.factors[i].mult — no GAP).
        self._verify_kron_factorization()

    def _verify_kron_factorization(self, n_samples: int = 5):
        """Verify L_G[g] = P · (⊗_i L_factor_i[g_i]) · Pᵀ for a sample of g.

        P is the permutation matrix corresponding to `kron_perm`:
            P[gap_idx, kron_perm[gap_idx]] = 1.
        Raises RuntimeError if any sampled element fails.
        """
        n = self.n
        P = np.zeros((n, n), dtype=np.uint8)
        for gap_idx, kron_idx in enumerate(self.kron_perm):
            P[gap_idx, kron_idx] = 1

        rng = np.random.default_rng(0)
        sample = rng.choice(n, size=min(n_samples, n), replace=False)
        for g_idx_arr in sample:
            g_idx = int(g_idx_arr)
            L_G = left_rep((g_idx,), self)
            factor_indices = self.decompose_table[g_idx]
            L_factors_list = [
                left_rep((factor_indices[i],), self.factors[i])
                for i in range(len(self.factors))
            ]
            L_kron = L_factors_list[0]
            for L_f in L_factors_list[1:]:
                L_kron = np.kron(L_kron, L_f)
            expected = (
                (P.astype(np.int64)
                 @ L_kron.astype(np.int64)
                 @ P.T.astype(np.int64)) % 2
            ).astype(np.uint8)
            if not np.array_equal(L_G, expected):
                raise RuntimeError(
                    f"Binary-level verification of direct-product "
                    f"decomposition failed for element {g_idx} "
                    f"({self.elem_strs[g_idx]}): L_G[g] != "
                    f"P · (⊗ L_factor_i[g_i]) · Pᵀ."
                )

    def decompose(self, g_index: int) -> tuple:
        """Return the factor-index tuple for a 0-based element index in G.

        Raises RuntimeError if this GroupData has no direct-product structure.
        """
        if self.decompose_table is None:
            raise RuntimeError(
                f"{self!r} has no direct-product decomposition. Pass "
                f"factor_gap_exprs at construction, or use a group with "
                f"DirectProductInfo (constructed via DirectProduct(...))."
            )
        return self.decompose_table[g_index]

    def compose(self, indices: Sequence[int]) -> int:
        """Return the 0-based element index in G from a tuple of factor indices.

        Raises RuntimeError if no direct-product structure, or KeyError if the
        indices don't correspond to any element (e.g. out-of-range).
        """
        if self.compose_table is None:
            raise RuntimeError(
                f"{self!r} has no direct-product decomposition."
            )
        return self.compose_table[tuple(indices)]

    # ─── misc ─────────────────────────────────────────────────────

    def __repr__(self):
        extra = ""
        if self.factors is not None:
            extra = f", factors={[f.gap_expr for f in self.factors]!r}"
        return (f"GroupData({self.gap_expr!r}, n={self.n}, "
                f"structure={self.structure!r}, "
                f"|[G,G]|={self.commutator_order}{extra})")

    def print_elements(self):
        print(f"Elements of {self.structure} (n={self.n}):")
        for i, s in enumerate(self.elem_strs):
            inv_str = self.elem_strs[self.inv[i]]
            print(f"  g[{i:2d}] = {s:<20s}  inv = {inv_str}")


# ─────────────────────────────────────────────────────────────────
# 2. Ring element operations — dagger and predicates
# ─────────────────────────────────────────────────────────────────


def canonicalize(x: Iterable[int]) -> tuple:
    """Reduce an F2[G] element to canonical form.

    Returns a sorted tuple of distinct indices, with even-count entries dropped
    (mod-2 cancellation). Idempotent on already-canonical input.
    """
    seen: set = set()
    for g in x:
        if g in seen:
            seen.discard(g)
        else:
            seen.add(g)
    return tuple(sorted(seen))


def dagger(x: Iterable[int], gd: GroupData) -> tuple:
    """x† = canonical sorted tuple of g⁻¹ for each g in x  (invert each element).

    Duplicates in x cancel mod 2 (so dagger of a non-canonical x is reduced).
    """
    return canonicalize(gd.inv[g] for g in x)


def matrix_dagger(M: list, gd: GroupData) -> list:
    """Dagger of a ring matrix: transpose rows/cols AND dagger each entry.

    If M is ma×na, returns na×ma. M†[ja][ia] = M[ia][ja]†.
    """
    ma = len(M)
    na = len(M[0])
    return [[dagger(M[ia][ja], gd) for ia in range(ma)] for ja in range(na)]


def element_order(g: int, gd: GroupData) -> int:
    """Order of group element g (0-based index): smallest k ≥ 1 with g^k = identity.

    Computed from the multiplication table — no GAP call needed.
    """
    if g == gd.identity:
        return 1
    k = 1
    current = g
    while current != gd.identity:
        current = gd.mult[current][g]
        k += 1
    return k


def is_self_dagger(x: Iterable[int], gd: GroupData) -> bool:
    """Return True iff x = x† under mod-2 / canonical semantics.

    Both x and x† are canonicalized before comparison, so duplicates in x
    cancel mod 2 as expected. Self-dagger elements satisfy L[x] = L[x]^T
    and R[x] = R[x]^T, i.e. the rep matrices are symmetric.
    """
    return dagger(x, gd) == canonicalize(x)


# ─────────────────────────────────────────────────────────────────
# 3. Ring arithmetic — F2[G] addition, multiplication, permanent
# ─────────────────────────────────────────────────────────────────


def ring_add(x: Iterable[int], y: Iterable[int]) -> tuple:
    """Add two ring elements in F2[G]: mod-2 sum of supports.

    Each index that appears with odd total multiplicity across x and y
    survives; even-multiplicity indices cancel. Returns a canonical sorted
    tuple. Robust to duplicates in either operand.
    """
    result: set = set()
    for g in itertools.chain(x, y):
        if g in result:
            result.discard(g)
        else:
            result.add(g)
    return tuple(sorted(result))


def ring_mul(x: Iterable[int], y: Iterable[int], gd: GroupData) -> tuple:
    """Multiply two ring elements in F2[G].

    (Σ α_g g) · (Σ β_h h)  =  Σ_{g,h} α_g β_h (g·h)   (mod 2)

    Works for any G (abelian or non-abelian). Commutative iff G is abelian.
    Returns a sorted tuple of 0-based group-element indices.
    """
    result: set = set()
    for g in x:
        for h in y:
            k = gd.mult[g][h]
            if k in result:
                result.discard(k)
            else:
                result.add(k)
    return tuple(sorted(result))


def ring_permanent(M: list, gd: GroupData) -> tuple:
    """Permanent of a J×J matrix of ring elements in F2[G].

    perm(M) = Σ_{σ ∈ S_J}  M[0][σ(0)] · M[1][σ(1)] · … · M[J-1][σ(J-1)]

    where · is ring_mul and Σ is ring_add (mod-2 coefficient sum).

    Args:
        M: J×J list of lists of iterables of int (ring elements).
        gd: GroupData.

    Returns a sorted tuple of group-element indices.
    """
    J = len(M)
    result: tuple = ()       # zero element of F2[G]
    for sigma in itertools.permutations(range(J)):
        term = M[0][sigma[0]]
        for i in range(1, J):
            term = ring_mul(term, M[i][sigma[i]], gd)
        result = ring_add(result, term)
    return result


# ─────────────────────────────────────────────────────────────────
# 4. Binary representation matrices  (n×n binary numpy arrays)
# ─────────────────────────────────────────────────────────────────


def left_rep(x: Iterable[int], gd: GroupData) -> np.ndarray:
    """L[x]: left-regular representation of x ∈ F2[G].

    L[g]_{h,k} = 1  iff  g·h = k   (g acts on the left).
    Equivalently: column h gets a 1 in row (g·h).
    """
    n = gd.n
    M = np.zeros((n, n), dtype=np.uint8)
    for g in x:
        for h in range(n):
            k = gd.mult[g][h]   # k = g·h
            M[k, h] ^= 1
    return M


def right_rep(x: Iterable[int], gd: GroupData) -> np.ndarray:
    """R[x]: right-regular representation of x ∈ F2[G].

    Convention:
        R[g]·e_h = e_{h·g⁻¹}
        R[g]_{h,k} = 1  iff  h·g⁻¹ = k
    Equivalently: column h gets a 1 in row (h·g⁻¹).

    Key consequences (hold under this convention):
        - R[g₁]·R[g₂] = R[g₁·g₂]  (homomorphism — same as L)
        - L[x] · R[y] = R[y] · L[x]  (L and R commute)
        - R[g]ᵀ = R[g⁻¹], so R[x]ᵀ = R[x†]
    Hx · Hzᵀ = 0 holds automatically with this convention.
    """
    n = gd.n
    M = np.zeros((n, n), dtype=np.uint8)
    for g in x:
        ginv = gd.inv[g]
        for h in range(n):
            k = gd.mult[h][ginv]   # k = h·g⁻¹
            M[k, h] ^= 1
    return M
