"""Fresh-eye contract tests for ``search/filters/``.

All expectations here are derived INDEPENDENTLY of the implementation:

- hand-built binary matrices with known girth / rank structure,
- a brute-force reference girth (per-edge removal + BFS) written below,
- a brute-force GF(2) rank and minimum-kernel-weight written below,
- group facts (element orders, commutator subgroup) recomputed in this
  file directly from the multiplication table.

Nothing in this file copies an output of the code under test as its
"expected" value.
"""

import itertools
from collections import deque

import numpy as np
import pytest

from search.filters.classical._shared.any_block_col_full_rank import (
    any_block_col_full_rank,
)
from search.filters.classical._shared.girth_tanner import girth_tanner
from search.filters.classical.abelian.base_girth_bound import base_girth_bound
from search.filters.config import (
    ClassicalFilterConfig,
    QuantumPairingFilterConfig,
    apply_classical_filters,
    apply_quantum_pairing_filters,
)
from search.filters.quantum_pairing._shared.full_extractor_bridge import (
    full_extractor_size_d_xz_bridge,
)


# ─────────────────────────────────────────────────────────────────
# Independent reference implementations (used as ground truth)
# ─────────────────────────────────────────────────────────────────


def _ref_tanner_girth(H):
    """Exact Tanner girth: for each edge, remove it and BFS between its
    endpoints; girth = min(dist + 1). None if the graph is a forest."""
    H = np.asarray(H) % 2
    r, c = H.shape
    N = r + c
    edges = []
    adj = [set() for _ in range(N)]
    for i in range(r):
        for j in np.where(H[i])[0]:
            u, v = i, r + int(j)
            edges.append((u, v))
            adj[u].add(v)
            adj[v].add(u)
    best = None
    for (u, v) in edges:
        dist = {u: 0}
        q = deque([u])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if {x, y} == {u, v}:
                    continue  # the removed edge
                if y not in dist:
                    dist[y] = dist[x] + 1
                    q.append(y)
        if v in dist:
            cyc = dist[v] + 1
            if best is None or cyc < best:
                best = cyc
    return best


def _ref_gf2_rank(M):
    """Plain Gaussian elimination over GF(2)."""
    M = np.array(M, dtype=np.uint8) % 2
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = next((rr for rr in range(r, rows) if M[rr, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for rr in range(rows):
            if rr != r and M[rr, c]:
                M[rr] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def _ref_min_kernel_weight(M):
    """Brute-force minimum weight of a nonzero kernel vector (<= 14 cols)."""
    M = np.asarray(M) % 2
    ncols = M.shape[1]
    assert ncols <= 14, "brute force reference limited to small matrices"
    best = None
    for bits in range(1, 2 ** ncols):
        v = np.array([(bits >> i) & 1 for i in range(ncols)], dtype=np.uint8)
        if not ((M @ v) % 2).any():
            w = int(v.sum())
            if best is None or w < best:
                best = w
    return best


def _ref_element_order(g, gd):
    """Order of g from the multiplication table only."""
    if g == gd.identity:
        return 1
    k, cur = 1, g
    while cur != gd.identity:
        cur = gd.mult[cur][g]
        k += 1
    return k


def _ref_commutator_subgroup(gd):
    """[G, G] generated from all commutators, via the mult table only."""
    comms = set()
    for a in range(gd.n):
        for b in range(gd.n):
            ab, ba = gd.mult[a][b], gd.mult[b][a]
            comms.add(gd.mult[ab][gd.inv[ba]])
    H = {gd.identity}
    frontier = list(comms)
    while frontier:
        x = frontier.pop()
        if x not in H:
            H.add(x)
        for h in list(H):
            for y in (gd.mult[x][h], gd.mult[h][x]):
                if y not in H:
                    H.add(y)
                    frontier.append(y)
    return H


def _ref_gf2_inverse(M):
    """Inverse of a square GF(2) matrix (Gauss-Jordan); None if singular."""
    M = np.array(M, dtype=np.uint8) % 2
    n = M.shape[0]
    A = np.concatenate([M, np.eye(n, dtype=np.uint8)], axis=1)
    row = 0
    for col in range(n):
        piv = next((rr for rr in range(row, n) if A[rr, col]), None)
        if piv is None:
            return None
        A[[row, piv]] = A[[piv, row]]
        for rr in range(n):
            if rr != row and A[rr, col]:
                A[rr] ^= A[row]
        row += 1
    return A[:, n:]


# ─────────────────────────────────────────────────────────────────
# Group fixtures (module-scoped to amortize GAP startup)
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def gd_c6():
    from core.group import GroupData
    return GroupData("CyclicGroup(6)")


@pytest.fixture(scope="module")
def gd_s3():
    from core.group import GroupData
    return GroupData("SymmetricGroup(3)")


# ─────────────────────────────────────────────────────────────────
# girth_tanner — hand-built matrices + differential vs reference
# ─────────────────────────────────────────────────────────────────


class TestGirthTannerFresh:
    pytestmark = pytest.mark.fast

    def test_triangle_incidence_has_girth_6(self):
        # Vertex-edge incidence matrix of a triangle: each pair of rows
        # shares exactly ONE column, so there is no 4-cycle; going around
        # the triangle gives a 6-cycle.
        H = np.array([[1, 1, 0],
                      [1, 0, 1],
                      [0, 1, 1]], dtype=np.uint8)
        assert girth_tanner(H) == 6

    def test_ring_of_four_checks_has_girth_8(self):
        # Check i touches vars {i, i+1 mod 4}: the Tanner graph is one
        # 8-cycle C0-v0-... and nothing shorter (no two checks share 2 vars).
        H = np.array([[1, 1, 0, 0],
                      [0, 1, 1, 0],
                      [0, 0, 1, 1],
                      [1, 0, 0, 1]], dtype=np.uint8)
        assert girth_tanner(H) == 8

    def test_shortest_cycle_wins_in_mixed_graph(self):
        # A 4-cycle (rows 0,1 share cols 0,1) embedded next to a longer
        # 6-cycle structure: girth must report 4, not 6.
        H = np.array([[1, 1, 1, 0, 0],
                      [1, 1, 0, 1, 0],
                      [0, 0, 1, 1, 1]], dtype=np.uint8)
        assert girth_tanner(H) == 4

    def test_weight_one_columns_never_make_cycles(self):
        # Every column has a single 1: the Tanner graph is a star/forest.
        H = np.array([[1, 1, 0],
                      [0, 0, 1]], dtype=np.uint8)
        assert girth_tanner(H) is None

    def test_differential_vs_bruteforce_reference(self):
        # 150 random small parity-check matrices; the fast BFS girth must
        # agree exactly with the per-edge-removal reference on every one.
        rng = np.random.default_rng(20260722)
        for _ in range(150):
            r = int(rng.integers(2, 6))
            c = int(rng.integers(2, 8))
            H = (rng.random((r, c)) < 0.4).astype(np.uint8)
            assert girth_tanner(H) == _ref_tanner_girth(H), H.tolist()

    def test_girth_always_even_or_none(self):
        # Bipartite graphs cannot have odd cycles.
        rng = np.random.default_rng(7)
        for _ in range(40):
            H = (rng.random((4, 5)) < 0.5).astype(np.uint8)
            g = girth_tanner(H)
            assert g is None or (g % 2 == 0 and g >= 4)


# ─────────────────────────────────────────────────────────────────
# base_girth_bound — boundary and precedence cases (pure integers)
# ─────────────────────────────────────────────────────────────────


class TestBaseGirthBoundFresh:
    pytestmark = pytest.mark.fast

    def test_entry_3_wins_over_row_rule(self):
        # Triggers both "any >= 3" (6) and "two >= 2 in a row" (8):
        # earliest rule wins -> 6.
        assert base_girth_bound([[3, 2], [2, 2]]) == 6

    def test_row_rule_wins_over_2x2_rule(self):
        # Two 2s in row 0 (-> 8) and also a 2x2 all-nonzero block with a
        # >= 2 entry (-> 10): the 8 must win.
        assert base_girth_bound([[2, 2], [1, 1]]) == 8

    def test_col_rule_two_weight2_same_column(self):
        assert base_girth_bound([[2, 1], [2, 1]]) == 8

    def test_diagonal_2s_hit_the_2x2_rule(self):
        # No row/col has two >= 2 entries, but the (all-nonzero) 2x2 block
        # contains entries >= 2 -> 10.
        assert base_girth_bound([[2, 1], [1, 2]]) == 10

    def test_all_ones_2x2_gives_no_bound(self):
        # 2x2 all-nonzero but NO entry >= 2, and shared support has only
        # 2 columns -> none of the rules fire.
        assert base_girth_bound([[1, 1], [1, 1]]) == float("inf")

    def test_antidiagonal_2s_give_no_bound(self):
        # The two weight-2 entries share no row, no column, and the rows
        # have no common support -> inf.
        assert base_girth_bound([[0, 2], [2, 0]]) == float("inf")

    def test_all_ones_2x3_gives_12(self):
        assert base_girth_bound([[1, 1, 1], [1, 1, 1]]) == 12

    def test_all_ones_3x2_gives_12(self):
        assert base_girth_bound([[1, 1], [1, 1], [1, 1]]) == 12

    def test_single_weight2_entry_no_bound(self):
        assert base_girth_bound([[2]]) == float("inf")

    def test_ndarray_and_list_agree(self):
        W = [[2, 1], [1, 2]]
        assert base_girth_bound(W) == base_girth_bound(np.array(W))


class TestBaseGirthBoundSoundness:
    """The mathematical claim behind the filter: for abelian G the actual
    Tanner girth of the lift never exceeds the weight-only bound."""

    pytestmark = pytest.mark.gap

    @pytest.mark.parametrize("W", [
        [[3]],                      # claims girth <= 6
        [[2, 2]],                   # claims girth <= 8 (row rule)
        [[2], [2]],                 # claims girth <= 8 (col rule)
        [[2, 1], [1, 2]],           # claims girth <= 10 (2x2 rule)
        [[1, 1, 1], [1, 1, 1]],     # claims girth <= 12 (2x3 rule)
    ])
    def test_bound_dominates_actual_girth(self, gd_c6, W):
        from core.classical_code import build_A_bin
        from search.sampling._shared.random_ring_matrix import (
            random_ring_matrix,
        )
        bound = base_girth_bound(W)
        assert bound < float("inf")
        rng = np.random.default_rng(hash(str(W)) % (2 ** 32))
        for _ in range(4):
            M = random_ring_matrix(gd_c6, np.array(W), rng=rng,
                                   canonicalize=False)
            assert M is not None
            g = _ref_tanner_girth(build_A_bin(M, gd_c6))
            # A finite bound promises a cycle of at most that length exists.
            assert g is not None and g <= bound, (M, g, bound)


# ─────────────────────────────────────────────────────────────────
# entry_order_bound — theorem-level verification
# ─────────────────────────────────────────────────────────────────


class TestEntryOrderBoundFresh:
    pytestmark = pytest.mark.gap

    def test_single_entry_bound_is_exact_kernel_minimum(self, gd_c6, gd_s3):
        # For a 1x1 ring matrix with one weight-2 entry {g1, g2}, the
        # kernel of L[g1] + L[g2] has minimum weight EXACTLY ord(g1^-1 g2).
        # entry_order_bound must therefore equal the brute-forced minimum
        # kernel weight of the lift.
        from core.classical_code import build_A_bin
        from search.filters.classical._shared.entry_order_bound import (
            entry_order_bound,
        )
        for gd in (gd_c6, gd_s3):
            for pair in itertools.combinations(range(gd.n), 2):
                bound = entry_order_bound([[pair]], gd)
                exact = _ref_min_kernel_weight(build_A_bin([[pair]], gd))
                assert bound == exact, (gd.structure, pair, bound, exact)

    def test_minimum_over_multiple_weight2_entries(self, gd_c6):
        from search.filters.classical._shared.entry_order_bound import (
            entry_order_bound,
        )
        # (0,1): ratio has order 6 in C6; (0,3): ratio has order 2 (my
        # own order computation below keeps this independent).
        A = [[(0, 1), (0, 3)]]
        expected = min(
            _ref_element_order(gd_c6.mult[gd_c6.inv[g1]][g2], gd_c6)
            for (g1, g2) in A[0]
        )
        assert expected == 2  # sanity of the hand-picked example
        assert entry_order_bound(A, gd_c6) == 2

    def test_non_weight2_entries_are_ignored(self, gd_c6):
        from search.filters.classical._shared.entry_order_bound import (
            entry_order_bound,
        )
        # Only the weight-2 entry (0,3) may contribute; weight 1 and
        # weight 3 entries must not.
        assert entry_order_bound([[(0,), (1, 2, 3), (0, 3)]], gd_c6) == 2
        assert entry_order_bound([[(0,), (1, 2, 3)]], gd_c6) is None


# ─────────────────────────────────────────────────────────────────
# abelianization_bound — formula + upper-bound property
# ─────────────────────────────────────────────────────────────────


class TestAbelianizationBoundFresh:
    pytestmark = pytest.mark.gap

    def test_formula_against_independent_commutator_subgroup(self, gd_s3):
        from search.filters.classical.non_abelian.abelianization_bound import (
            abelianization_bound,
        )
        comm = _ref_commutator_subgroup(gd_s3)
        assert len(comm) == 3  # [S3, S3] = A3, computed from the mult table
        assert abelianization_bound([[(0,), (1,)]], gd_s3) == 2 * len(comm)
        assert abelianization_bound([[(0, 1), (2, 3, 5)]], gd_s3) == 5 * len(comm)

    def test_bound_dominates_true_distance(self, gd_s3):
        # d(A_bin) <= bound must hold for real codes: brute-force the
        # exact kernel minimum weight for a few small S3 sides.
        from core.classical_code import build_A_bin
        from search.filters.classical.non_abelian.abelianization_bound import (
            abelianization_bound,
        )
        for A in ([[(0,), (1,)]], [[(0, 1), (2,)]], [[(1, 2), (3, 4)]]):
            d_true = _ref_min_kernel_weight(build_A_bin(A, gd_s3))
            assert d_true is not None
            assert d_true <= abelianization_bound(A, gd_s3), A


# ─────────────────────────────────────────────────────────────────
# any_block_col_full_rank — adversarial hand-built binary inputs
# ─────────────────────────────────────────────────────────────────


class TestAnyBlockColFullRankFresh:
    pytestmark = pytest.mark.fast

    def test_full_row_rank_but_no_invertible_block(self):
        # Docstring claims "strictly stronger than full row rank".
        # n = 2, ma = 1, na = 2: both 2x2 blocks singular (rank 1) but
        # the full 2x4 matrix has rank 2.
        A_bin = np.array([[1, 0, 1, 1],
                          [1, 0, 0, 0]], dtype=np.uint8)
        assert _ref_gf2_rank(A_bin) == 2                    # full row rank
        assert _ref_gf2_rank(A_bin[:, 0:2]) == 1            # block 0 singular
        assert _ref_gf2_rank(A_bin[:, 2:4]) == 1            # block 1 singular
        assert any_block_col_full_rank(A_bin, n=2, ma=1) is False

    def test_only_middle_block_invertible(self):
        # Position independence: the invertible block sits at index 1 of 3.
        sing = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        eye = np.eye(2, dtype=np.uint8)
        A_bin = np.hstack([sing, eye, sing])
        assert any_block_col_full_rank(A_bin, n=2, ma=1) is True

    def test_rank_deficient_by_one_everywhere(self):
        # Each block has rank n-1: none passes.
        b = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        A_bin = np.hstack([b, b, b])
        assert any_block_col_full_rank(A_bin, n=2, ma=1) is False

    def test_ma2_specific_pair_needed(self):
        # ma = 2, n = 2 (4 rows). Three block-cols built so that ONLY the
        # subset {0, 2} stacks to an invertible 4x4.
        col0 = np.vstack([np.eye(2, dtype=np.uint8),
                          np.zeros((2, 2), dtype=np.uint8)])
        col1 = np.vstack([np.eye(2, dtype=np.uint8),
                          np.zeros((2, 2), dtype=np.uint8)])   # duplicate of col0
        col2 = np.vstack([np.zeros((2, 2), dtype=np.uint8),
                          np.eye(2, dtype=np.uint8)])
        A_bin = np.hstack([col0, col1, col2])
        assert _ref_gf2_rank(np.hstack([col0, col1])) == 2   # {0,1} fails
        assert _ref_gf2_rank(np.hstack([col0, col2])) == 4   # {0,2} works
        assert any_block_col_full_rank(A_bin, n=2, ma=2) is True

    def test_ma2_no_pair_works(self):
        # All three block-cols identical with joint rank 3 < 4.
        col = np.vstack([np.eye(2, dtype=np.uint8),
                         np.array([[1, 0], [0, 0]], dtype=np.uint8)])
        A_bin = np.hstack([col, col, col])
        assert any_block_col_full_rank(A_bin, n=2, ma=2) is False

    def test_fewer_blocks_than_ma_is_false(self):
        # na = 1 block but ma = 2: no size-2 subset exists at all.
        A_bin = np.vstack([np.eye(2, dtype=np.uint8),
                           np.eye(2, dtype=np.uint8)])
        assert any_block_col_full_rank(A_bin, n=2, ma=2) is False


# ─────────────────────────────────────────────────────────────────
# canonical_logical_weight — independent linear-algebra expectation
# ─────────────────────────────────────────────────────────────────


class TestCanonicalLogicalWeightFresh:
    pytestmark = pytest.mark.gap

    def test_two_permutation_blocks_give_weight_2(self, gd_c6, gd_s3):
        # [[(e,), (g,)]]: unit vector in the free block plus the forced
        # solve through a permutation block -> weight exactly 2.
        from core.classical_code import build_A_bin
        from search.filters.classical._shared.canonical_logical_weight import (
            canonical_logical_weight,
        )
        for gd in (gd_c6, gd_s3):
            M_bin = build_A_bin([[(0,), (2,)]], gd)
            assert canonical_logical_weight(M_bin, gd.n) == 2

    def test_matches_independent_solve_through_unit_anchor(self, gd_s3):
        # A = [[a0, a1]] with a1 a unit: the structured kernel codewords
        # are e_i (free block) + L[a1]^{-1} L[a0] e_i (anchor block).
        # Expected weight = max_i (1 + wt(L[a1]^{-1} L[a0] e_i)),
        # computed here with our own GF(2) inverse.
        from core.classical_code import build_A_bin
        from core.group import left_rep
        from search.filters.classical._shared.canonical_logical_weight import (
            canonical_logical_weight,
        )
        n = gd_s3.n
        a1 = next(
            c for c in itertools.combinations(range(n), 5)
            if _ref_gf2_rank(left_rep(c, gd_s3)) == n
        )
        a0 = (0, 1)
        M_bin = build_A_bin([[a0, a1]], gd_s3)
        L0 = left_rep(a0, gd_s3)
        L1inv = _ref_gf2_inverse(left_rep(a1, gd_s3))
        assert L1inv is not None
        expected = 0
        for i in range(n):
            e = np.zeros(n, dtype=np.uint8)
            e[i] = 1
            y = (L1inv @ ((L0 @ e) % 2)) % 2
            expected = max(expected, 1 + int(y.sum()))
        assert canonical_logical_weight(M_bin, n) == expected

    def test_none_when_no_block_invertible(self, gd_c6):
        # Both entries (0,3): I + L[g^3] is singular (even weight).
        from core.classical_code import build_A_bin
        from search.filters.classical._shared.canonical_logical_weight import (
            canonical_logical_weight,
        )
        M_bin = build_A_bin([[(0, 3), (0, 3)]], gd_c6)
        assert canonical_logical_weight(M_bin, gd_c6.n) is None

    def test_weight_is_realized_by_an_actual_kernel_codeword(self, gd_c6):
        # The returned value must be the weight of a genuine element of
        # ker(M_bin), and at least the true minimum distance.
        from core.classical_code import build_A_bin
        from core.group import left_rep
        from search.filters.classical._shared.canonical_logical_weight import (
            canonical_logical_weight,
        )
        # Anchor: first weight-3 element of C6 whose lift is invertible,
        # located with OUR OWN rank routine (e.g. 1 + x + x^3 works; the
        # naive 1 + x + x^2 is divisible by x^2 + x + 1 and does not).
        a1 = next(
            c for c in itertools.combinations(range(gd_c6.n), 3)
            if _ref_gf2_rank(left_rep(c, gd_c6)) == gd_c6.n
        )
        M_bin = build_A_bin([[(0, 1), a1]], gd_c6)
        w = canonical_logical_weight(M_bin, gd_c6.n)
        assert w is not None
        kernel_weights = set()
        ncols = M_bin.shape[1]
        for bits in range(1, 2 ** ncols):
            v = np.array([(bits >> i) & 1 for i in range(ncols)],
                         dtype=np.uint8)
            if not ((M_bin @ v) % 2).any():
                kernel_weights.add(int(v.sum()))
        assert w in kernel_weights
        assert w >= min(kernel_weights)


# ─────────────────────────────────────────────────────────────────
# apply_classical_filters — thresholds exactly AT the boundary
# ─────────────────────────────────────────────────────────────────


class TestClassicalDispatcherBoundaries:
    pytestmark = pytest.mark.gap

    def _abin(self, A, gd):
        from core.classical_code import build_A_bin
        return build_A_bin(A, gd)

    def test_girth_threshold_boundary(self, gd_c6):
        # C6 weight-3 entry {0,1,2}: adjacent columns overlap in two rows
        # -> a 4-cycle exists; our reference confirms girth 4.
        A = [[(0, 1, 2)]]
        A_bin = self._abin(A, gd_c6)
        g = _ref_tanner_girth(A_bin)
        assert g == 4
        ok_at = apply_classical_filters(
            A, A_bin, gd_c6, ClassicalFilterConfig(min_girth_tanner_A_bin=g))
        ok_above = apply_classical_filters(
            A, A_bin, gd_c6,
            ClassicalFilterConfig(min_girth_tanner_A_bin=g + 1))
        assert ok_at is True      # girth == threshold passes
        assert ok_above is False  # girth < threshold rejects

    def test_forest_passes_any_girth_threshold(self, gd_c6):
        # Single permutation block: the Tanner graph is a perfect matching.
        A = [[(0,)]]
        A_bin = self._abin(A, gd_c6)
        assert _ref_tanner_girth(A_bin) is None
        cfg = ClassicalFilterConfig(min_girth_tanner_A_bin=100)
        assert apply_classical_filters(A, A_bin, gd_c6, cfg) is True

    def test_base_girth_bound_boundary(self, gd_c6):
        A = [[(0, 1, 2), (0,)]]          # W = [[3, 1]] -> bound 6
        A_bin = self._abin(A, gd_c6)
        assert apply_classical_filters(
            A, A_bin, gd_c6,
            ClassicalFilterConfig(min_base_girth_bound=6)) is True
        assert apply_classical_filters(
            A, A_bin, gd_c6,
            ClassicalFilterConfig(min_base_girth_bound=7)) is False

    def test_entry_order_bound_boundary(self, gd_c6):
        # (0,3): ratio g^3 has order 2 (independent computation above).
        A = [[(0, 3), (0,)]]
        A_bin = self._abin(A, gd_c6)
        assert apply_classical_filters(
            A, A_bin, gd_c6,
            ClassicalFilterConfig(min_entry_order_bound=2)) is True
        assert apply_classical_filters(
            A, A_bin, gd_c6,
            ClassicalFilterConfig(min_entry_order_bound=3)) is False

    def test_entry_order_bound_none_passes_any_threshold(self, gd_c6):
        A = [[(0,), (1, 2, 3)]]          # no weight-2 entry
        A_bin = self._abin(A, gd_c6)
        cfg = ClassicalFilterConfig(min_entry_order_bound=999)
        assert apply_classical_filters(A, A_bin, gd_c6, cfg) is True

    def test_abelianization_bound_boundary(self, gd_s3):
        # total 5 entries x |[S3,S3]| = 3 -> bound 15.
        A = [[(0, 1), (2, 3, 5)]]
        A_bin = self._abin(A, gd_s3)
        assert apply_classical_filters(
            A, A_bin, gd_s3,
            ClassicalFilterConfig(min_abelianization_bound=15)) is True
        assert apply_classical_filters(
            A, A_bin, gd_s3,
            ClassicalFilterConfig(min_abelianization_bound=16)) is False

    def test_ring_and_weight_distance_bound_boundary(self, gd_c6):
        # J=1: both bounds equal |a0| + |a1| = 2 for two weight-1 entries.
        A = [[(0,), (1,)]]
        A_bin = self._abin(A, gd_c6)
        for field in ("min_ring_distance_bound", "min_weight_distance_bound"):
            assert apply_classical_filters(
                A, A_bin, gd_c6,
                ClassicalFilterConfig(**{field: 2})) is True, field
            assert apply_classical_filters(
                A, A_bin, gd_c6,
                ClassicalFilterConfig(**{field: 3})) is False, field

    def test_max_canonical_logical_weight_boundary_and_noop(self, gd_s3):
        from core.group import left_rep
        n = gd_s3.n
        a1 = next(
            c for c in itertools.combinations(range(n), 5)
            if _ref_gf2_rank(left_rep(c, gd_s3)) == n
        )
        A = [[(0, 1), a1]]
        A_bin = self._abin(A, gd_s3)
        # Recompute the canonical weight independently (same as above).
        L0 = left_rep((0, 1), gd_s3)
        L1inv = _ref_gf2_inverse(left_rep(a1, gd_s3))
        w = max(
            1 + int(((L1inv @ ((L0 @ e) % 2)) % 2).sum())
            for e in np.eye(n, dtype=np.uint8)
        )
        assert w < A_bin.shape[1]  # the threshold below is genuinely active
        assert apply_classical_filters(
            A, A_bin, gd_s3,
            ClassicalFilterConfig(max_canonical_logical_weight=w)) is True
        assert apply_classical_filters(
            A, A_bin, gd_s3,
            ClassicalFilterConfig(max_canonical_logical_weight=w - 1)) is False
        # Threshold == code length is the documented no-op.
        assert apply_classical_filters(
            A, A_bin, gd_s3,
            ClassicalFilterConfig(
                max_canonical_logical_weight=A_bin.shape[1])) is True

    def test_max_canonical_logical_weight_passes_unstructured_side(self, gd_c6):
        # No invertible block-col -> no canonical basis -> the filter
        # defers (documented) even with an aggressive threshold.
        A = [[(0, 3), (0, 3)]]
        A_bin = self._abin(A, gd_c6)
        cfg = ClassicalFilterConfig(max_canonical_logical_weight=1)
        assert apply_classical_filters(A, A_bin, gd_c6, cfg) is True

    def test_require_full_rank_rejects_unitless_weights(self, gd_s3):
        # S3 has no units at weight 2 (augmentation + brute fact), so any
        # 1x2 matrix of weight-2 entries must fail the full-rank gate.
        A = [[(0, 1), (2, 3)]]
        A_bin = self._abin(A, gd_s3)
        cfg = ClassicalFilterConfig(require_any_block_col_full_rank=True)
        assert apply_classical_filters(A, A_bin, gd_s3, cfg) is False


# ─────────────────────────────────────────────────────────────────
# apply_quantum_pairing_filters — boundary passes + bridge plumbing
# ─────────────────────────────────────────────────────────────────


class TestPairingDispatcherBoundaries:
    pytestmark = pytest.mark.fast

    def _meta(self, **kw):
        base = {"group_tag": "G", "shape": (1, 2), "dist": 7, "girth": 8,
                "weight_matrix": np.array([[1, 2]])}
        base.update(kw)
        return base

    def test_distance_exactly_at_threshold_passes(self):
        cfg = QuantumPairingFilterConfig(min_classical_distance=7)
        assert apply_quantum_pairing_filters(
            self._meta(), self._meta(), cfg) is True
        cfg2 = QuantumPairingFilterConfig(min_classical_distance=8)
        assert apply_quantum_pairing_filters(
            self._meta(), self._meta(), cfg2) is False

    def test_girth_exactly_at_threshold_passes(self):
        cfg = QuantumPairingFilterConfig(min_classical_girth=8)
        assert apply_quantum_pairing_filters(
            self._meta(), self._meta(), cfg) is True
        # One side a forest, other side exactly at threshold -> pass.
        assert apply_quantum_pairing_filters(
            self._meta(girth=None), self._meta(), cfg) is True

    def test_check_weight_cap_exactly_at_derived_weight(self):
        # W_A = [[1,2]], W_B = [[1,1]]:
        #   Hx = max rowsum(W_A) + max colsum(W_B) = 3 + 1 = 4
        #   Hz = max colsum(W_A) + max rowsum(W_B) = 2 + 2 = 4
        mA = self._meta(weight_matrix=np.array([[1, 2]]))
        mB = self._meta(weight_matrix=np.array([[1, 1]]))
        at = QuantumPairingFilterConfig(max_Hx_check_weight=4,
                                        max_Hz_check_weight=4)
        assert apply_quantum_pairing_filters(mA, mB, at) is True
        below = QuantumPairingFilterConfig(max_Hx_check_weight=3)
        assert apply_quantum_pairing_filters(mA, mB, below) is False

    def test_bridge_filter_needs_gd(self):
        cfg = QuantumPairingFilterConfig(min_full_extractor_bridge_d=2)
        meta = self._meta(matrix=[[[0], [1]]])
        with pytest.raises(ValueError, match="GroupData"):
            apply_quantum_pairing_filters(meta, meta, cfg)

    def test_bridge_filter_missing_matrix_key_reported(self):
        cfg = QuantumPairingFilterConfig(min_full_extractor_bridge_d=2)
        with pytest.raises(ValueError, match="matrix"):
            apply_quantum_pairing_filters(self._meta(), self._meta(), cfg)


class TestFullExtractorBridgePredicate:
    """Hand-built graphs with independently known longest simple paths."""

    pytestmark = pytest.mark.fast

    def _star_and_path(self):
        # X side: Lx supported on {0,1,2,3}; Hz rings make a STAR centered
        # at 0 (edges 0-1, 0-2, 0-3): component size 4, longest simple
        # path 3. Z side: Lz on {4,5,6,7}; Hx rings make a PATH 4-5-6-7:
        # longest simple path 4. No pivots (disjoint supports).
        n = 8
        Lx = np.zeros((1, n), dtype=np.uint8)
        Lx[0, [0, 1, 2, 3]] = 1
        Lz = np.zeros((1, n), dtype=np.uint8)
        Lz[0, [4, 5, 6, 7]] = 1
        Hz = np.zeros((3, n), dtype=np.uint8)
        Hz[0, [0, 1]] = 1
        Hz[1, [0, 2]] = 1
        Hz[2, [0, 3]] = 1
        Hx = np.zeros((3, n), dtype=np.uint8)
        Hx[0, [4, 5]] = 1
        Hx[1, [5, 6]] = 1
        Hx[2, [6, 7]] = 1
        return Hx, Hz, Lx, Lz

    def test_star_limits_path_not_component(self):
        Hx, Hz, Lx, Lz = self._star_and_path()
        # d = 3: star's longest simple path (leaf-center-leaf) suffices.
        assert full_extractor_size_d_xz_bridge(Hx, Hz, Lx, Lz, 3) is True
        # d = 4: the star COMPONENT has 4 vertices but no simple path of
        # 4 — an implementation that only checks component size would
        # wrongly pass here.
        assert full_extractor_size_d_xz_bridge(Hx, Hz, Lx, Lz, 4) is False

    def test_trivial_d1_passes(self):
        Hx, Hz, Lx, Lz = self._star_and_path()
        assert full_extractor_size_d_xz_bridge(Hx, Hz, Lx, Lz, 1) is True

    def test_pivot_deletion_fragments_the_ring(self):
        # Lx on {0,1,2}, Lz on {2,3,4}: pivot = {2}. A single Hz check
        # rings 0-1-2, but only the edge (0,1) survives pivot deletion;
        # similarly Hx rings 2-3-4 leaving edge (3,4).
        n = 5
        Lx = np.zeros((1, n), dtype=np.uint8)
        Lx[0, [0, 1, 2]] = 1
        Lz = np.zeros((1, n), dtype=np.uint8)
        Lz[0, [2, 3, 4]] = 1
        Hz = np.zeros((1, n), dtype=np.uint8)
        Hz[0, [0, 1, 2]] = 1
        Hx = np.zeros((1, n), dtype=np.uint8)
        Hx[0, [2, 3, 4]] = 1
        assert full_extractor_size_d_xz_bridge(Hx, Hz, Lx, Lz, 2) is True
        assert full_extractor_size_d_xz_bridge(Hx, Hz, Lx, Lz, 3) is False

    def test_d_below_1_raises(self):
        Hx, Hz, Lx, Lz = self._star_and_path()
        with pytest.raises(ValueError, match="d >= 1"):
            full_extractor_size_d_xz_bridge(Hx, Hz, Lx, Lz, 0)

    def test_column_count_mismatch_raises(self):
        Hx, Hz, Lx, Lz = self._star_and_path()
        with pytest.raises(ValueError, match="column count"):
            full_extractor_size_d_xz_bridge(
                Hx, Hz, Lx, np.zeros((1, 7), dtype=np.uint8), 2)


class TestFullExtractorBridgeFromRings:
    pytestmark = pytest.mark.gap

    def test_unstructured_pair_returns_false_not_raise(self, gd_c6):
        # Both blocks even-weight -> singular -> no full-rank block-col on
        # either side -> the full extractor is unbuildable: documented as
        # a reject (False), never an exception.
        from search.filters.quantum_pairing._shared.full_extractor_bridge import (
            full_extractor_bridge_from_rings,
        )
        A = [[(0, 3), (0, 3)]]
        assert full_extractor_bridge_from_rings(A, A, gd_c6, 2) is False

    def test_bad_d_raises(self, gd_c6):
        from search.filters.quantum_pairing._shared.full_extractor_bridge import (
            full_extractor_bridge_from_rings,
        )
        with pytest.raises(ValueError, match="d >= 1"):
            full_extractor_bridge_from_rings([[(0,), (1,)]],
                                             [[(0,), (1,)]], gd_c6, 0)

    def test_bridge_subgroup_order_matches_hand_derivation(self, gd_c6, gd_s3):
        from search.filters.quantum_pairing._shared.full_extractor_bridge import (
            bridge_subgroup_order,
        )
        # C6, anchor {e, g}: difference set {g, g^-1} generates all of C6.
        assert bridge_subgroup_order((0, 1), gd_c6) == 6
        # C6, anchor {e, g^3}: g^3 has order 2 -> subgroup of size 2.
        assert bridge_subgroup_order((0, 3), gd_c6) == 2
        # weight-1 anchor: empty difference set -> trivial subgroup.
        assert bridge_subgroup_order((0,), gd_c6) == 1
        # S3, anchor = full group -> difference set generates S3.
        assert bridge_subgroup_order(tuple(range(6)), gd_s3) == 6
