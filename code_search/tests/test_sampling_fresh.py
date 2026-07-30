"""Fresh-eye contract tests for ``search/sampling/``.

Every expectation is derived independently:

- group facts (element orders, commutator cosets) recomputed here from
  the multiplication table,
- GF(2) ranks recomputed with a hand-rolled Gaussian elimination,
- pool contents cross-checked against a from-scratch exhaustive
  enumeration written in this file,
- unit-existence facts derived from the augmentation map (even-weight
  ring elements can never be units).
"""

import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest

from search.sampling._shared.full_rank_block_pool import (
    build_full_rank_block_pool,
    build_full_rank_block_pool_brute,
    count_weight_elements,
    enumerate_weight_elements,
    sample_A_from_pool,
)
from search.sampling._shared.random_ring_element import random_ring_element
from search.sampling._shared.random_ring_matrix import random_ring_matrix
from search.sampling._shared.weight_matrix import (
    all_weight_patterns,
    random_weight_patterns,
)


# ─────────────────────────────────────────────────────────────────
# Independent reference helpers
# ─────────────────────────────────────────────────────────────────


def _ref_gf2_rank(M):
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


def _ref_element_order(g, gd):
    if g == gd.identity:
        return 1
    k, cur = 1, g
    while cur != gd.identity:
        cur = gd.mult[cur][g]
        k += 1
    return k


def _ref_commutator_subgroup(gd):
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


def _ref_left_coset(g, H, gd):
    """Left coset g·H as a frozenset (from the mult table)."""
    return frozenset(gd.mult[g][h] for h in H)


# ─────────────────────────────────────────────────────────────────
# Fixtures
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
# random_ring_element — constraint conjunction across many draws
# ─────────────────────────────────────────────────────────────────


class TestRandomRingElementFresh:
    pytestmark = pytest.mark.gap

    def test_all_constraints_hold_simultaneously(self, gd_s3):
        # 40 seeded draws with BOTH include_identity=False AND
        # min_element_order=2: every documented constraint must hold on
        # every draw (weight, sortedness, distinctness, range, identity
        # exclusion, per-element order via our own order computation).
        rng = np.random.default_rng(101)
        for _ in range(40):
            x = random_ring_element(
                gd_s3, 2, rng=rng,
                include_identity=False, min_element_order=2,
            )
            assert x is not None
            assert len(x) == 2
            assert list(x) == sorted(set(x))
            assert all(0 <= g < gd_s3.n for g in x)
            assert gd_s3.identity not in x
            assert all(_ref_element_order(g, gd_s3) >= 2 for g in x)

    def test_weight_equals_n_returns_whole_group(self, gd_c6):
        rng = np.random.default_rng(0)
        x = random_ring_element(gd_c6, gd_c6.n, rng=rng)
        assert x == tuple(range(gd_c6.n))

    def test_weight_n_without_identity_is_infeasible(self, gd_c6):
        rng = np.random.default_rng(0)
        x = random_ring_element(gd_c6, gd_c6.n, rng=rng,
                                include_identity=False)
        assert x is None

    def test_min_order_above_group_exponent_is_infeasible(self, gd_s3):
        # No element of S3 has order >= 7.
        rng = np.random.default_rng(0)
        assert random_ring_element(gd_s3, 1, rng=rng,
                                   min_element_order=7) is None

    def test_avoid_same_coset_against_independent_cosets(self, gd_s3):
        # Verify the coset-distinctness promise with OUR OWN commutator
        # subgroup and coset computation (never touching gd.coset_id).
        H = _ref_commutator_subgroup(gd_s3)
        rng = np.random.default_rng(11)
        for _ in range(30):
            x = random_ring_element(gd_s3, 2, rng=rng,
                                    avoid_same_coset=True)
            assert x is not None
            cosets = {_ref_left_coset(g, H, gd_s3) for g in x}
            assert len(cosets) == 2, (x, cosets)

    def test_results_vary_across_seeds(self, gd_s3):
        draws = {
            random_ring_element(gd_s3, 3, rng=np.random.default_rng(seed))
            for seed in range(10)
        }
        assert len(draws) >= 2


# ─────────────────────────────────────────────────────────────────
# random_ring_matrix — determinism + canonical-form contract
# ─────────────────────────────────────────────────────────────────


class TestRandomRingMatrixFresh:
    pytestmark = pytest.mark.gap

    def test_matrix_level_seed_determinism(self, gd_s3):
        W = np.array([[2, 3]])
        m1 = random_ring_matrix(gd_s3, W, rng=np.random.default_rng(42))
        m2 = random_ring_matrix(gd_s3, W, rng=np.random.default_rng(42))
        assert m1 == m2

    def test_matrix_level_seed_divergence(self, gd_s3):
        W = np.array([[2, 3]])
        results = {
            str(random_ring_matrix(gd_s3, W,
                                   rng=np.random.default_rng(seed)))
            for seed in range(6)
        }
        assert len(results) >= 2

    def test_canonicalize_moves_unit_anchor_to_last_block(self, gd_s3):
        # Every weight-5 element of S3 is a unit (all 6 of them — the
        # augmentation is 1 and brute rank confirms), so sampling
        # W = [[5, 2]] with canonicalize=True must ALWAYS deliver the
        # weight-5 unit in the LAST block-col: sampled weights become
        # [2, 5] and the last binary block is invertible (checked with
        # our own GF(2) rank).
        from core.classical_code import build_A_bin
        rng = np.random.default_rng(5)
        n = gd_s3.n
        for _ in range(15):
            M = random_ring_matrix(gd_s3, np.array([[5, 2]]), rng=rng,
                                   canonicalize=True)
            assert M is not None
            assert [len(e) for e in M[0]] == [2, 5]
            A_bin = build_A_bin(M, gd_s3)
            assert _ref_gf2_rank(A_bin[:, n:2 * n]) == n

    def test_canonicalize_false_keeps_weights_in_place(self, gd_s3):
        rng = np.random.default_rng(5)
        M = random_ring_matrix(gd_s3, np.array([[5, 2]]), rng=rng,
                               canonicalize=False)
        assert M is not None
        assert [len(e) for e in M[0]] == [5, 2]

    def test_entries_respect_matrix_level_knobs(self, gd_s3):
        rng = np.random.default_rng(9)
        for _ in range(10):
            M = random_ring_matrix(
                gd_s3, np.array([[2, 2]]), rng=rng,
                include_identity=False, canonicalize=False,
            )
            assert M is not None
            for entry in M[0]:
                assert gd_s3.identity not in entry


# ─────────────────────────────────────────────────────────────────
# full-rank pools — differential vs from-scratch enumeration
# ─────────────────────────────────────────────────────────────────


class TestFullRankPoolFresh:
    pytestmark = pytest.mark.gap

    def test_brute_pool_matches_independent_enumeration(self, gd_c6):
        # Recompute the ENTIRE pool from scratch: every weight-3 support
        # containing the identity whose left lift has full rank per our
        # own Gaussian elimination.
        from core.group import left_rep
        n = gd_c6.n
        expected = [
            (0,) + rest
            for rest in itertools.combinations(range(1, n), 2)
            if _ref_gf2_rank(left_rep((0,) + rest, gd_c6)) == n
        ]
        pool = build_full_rank_block_pool_brute(gd_c6, 3, force_identity=True)
        assert sorted(pool) == sorted(expected)
        assert len(expected) > 0  # C6 does have weight-3 units

    def test_even_weights_never_yield_units(self, gd_c6, gd_s3):
        # Augmentation argument: for even |x| the rows of L[x] sum to 0
        # over GF(2), so L[x] is singular. The pool must come back empty.
        for gd in (gd_c6, gd_s3):
            for w in (2, 4):
                assert build_full_rank_block_pool_brute(
                    gd, w, force_identity=False) == []

    def test_s3_units_only_at_weights_1_and_5(self, gd_s3):
        # Documented in the pool builder's docstring: S3 has units only
        # at weights 1 and 5 (weight 3 gives zero despite being odd).
        assert build_full_rank_block_pool_brute(
            gd_s3, 3, force_identity=False) == []
        w1 = build_full_rank_block_pool_brute(gd_s3, 1, force_identity=False)
        assert len(w1) == 6      # every L[g] is a permutation matrix
        w5 = build_full_rank_block_pool_brute(gd_s3, 5, force_identity=False)
        assert len(w5) == 6      # complement of each single element
        # With force_identity: exactly the C(5,4)=5 supports containing e.
        w5_id = build_full_rank_block_pool_brute(gd_s3, 5, force_identity=True)
        assert len(w5_id) == 5
        assert all(0 in x for x in w5_id)

    def test_brute_max_pool_size_is_prefix_of_enumeration(self, gd_c6):
        full = build_full_rank_block_pool_brute(gd_c6, 3, force_identity=True)
        head = build_full_rank_block_pool_brute(gd_c6, 3, force_identity=True,
                                                max_pool_size=2)
        assert head == full[:2]

    def test_random_pool_is_subset_of_brute_pool(self, gd_c6):
        brute = set(build_full_rank_block_pool_brute(
            gd_c6, 3, force_identity=False))
        rand = build_full_rank_block_pool(
            gd_c6, 3, max_pool_size=50, max_tries=3000, seed=1)
        assert len(rand) > 0
        assert set(rand) <= brute
        assert len(set(rand)) == len(rand)   # no duplicates in the pool

    def test_enumerate_counts_and_shape(self, gd_c6):
        n = gd_c6.n
        for w in (1, 2, 3, 6):
            els = list(enumerate_weight_elements(gd_c6, w))
            assert len(els) == math.comb(n, w)
            assert len(set(els)) == len(els)
            assert all(tuple(sorted(e)) == e and len(e) == w for e in els)
        forced = list(enumerate_weight_elements(gd_c6, 3, force_identity=True))
        assert len(forced) == math.comb(n - 1, 2)
        assert all(0 in e for e in forced)

    def test_count_weight_elements_pure_math(self):
        assert count_weight_elements(6, 3) == 20
        assert count_weight_elements(6, 3, force_identity=True) == 10
        assert count_weight_elements(8, 1, force_identity=True) == 1


class TestSampleAFromPoolFresh:
    pytestmark = pytest.mark.gap

    def test_anchor_always_comes_from_the_pool(self, gd_c6):
        pool = build_full_rank_block_pool_brute(gd_c6, 3, force_identity=True)
        assert pool
        rng = np.random.default_rng(2)
        for _ in range(10):
            M = sample_A_from_pool(gd_c6, pool, (1, 3), 2, rng)
            assert M is not None
            assert len(M) == 1 and len(M[0]) == 3
            assert M[0][-1] in set(pool)          # anchor membership
            assert all(len(e) == 2 for e in M[0][:-1])   # free-col weights

    def test_seed_determinism_and_divergence(self, gd_c6):
        pool = build_full_rank_block_pool_brute(gd_c6, 3, force_identity=True)
        a = sample_A_from_pool(gd_c6, pool, (1, 2), 2,
                               np.random.default_rng(7))
        b = sample_A_from_pool(gd_c6, pool, (1, 2), 2,
                               np.random.default_rng(7))
        assert a == b
        draws = {
            str(sample_A_from_pool(gd_c6, pool, (1, 2), 2,
                                   np.random.default_rng(s)))
            for s in range(8)
        }
        assert len(draws) >= 2

    def test_infeasible_free_col_knobs_return_none(self, gd_c6):
        # weight = n with include_identity=False leaves only n-1 eligible
        # elements: the free-column sampler must fail and the whole call
        # must return None (not raise).
        pool = build_full_rank_block_pool_brute(gd_c6, 1, force_identity=True)
        M = sample_A_from_pool(
            gd_c6, pool, (1, 2), gd_c6.n, np.random.default_rng(0),
            free_col_knobs={"include_identity": False},
        )
        assert M is None


# ─────────────────────────────────────────────────────────────────
# weight-pattern streams (pure integer arithmetic — fast)
# ─────────────────────────────────────────────────────────────────


def _stub_abelian_gd():
    """weight_distance_bound only reads gd.is_abelian (and .structure in
    error paths); a stub keeps these tests GAP-free."""
    return SimpleNamespace(is_abelian=True, structure="stub-abelian")


def _ref_pairwise_min_sum(W):
    """Reference J=1 weight-distance bound: min over column pairs of the
    two entries' sum."""
    row = list(W[0])
    return min(a + b for a, b in itertools.combinations(row, 2))


class TestWeightPatternsFresh:
    pytestmark = pytest.mark.fast

    def test_missing_gd_raises_on_iteration(self):
        gen = random_weight_patterns(
            (1, 3), 2, 5, rng=np.random.default_rng(0),
            min_weight_distance_bound=3, gd=None,
        )
        with pytest.raises(ValueError, match="GroupData"):
            list(gen)

    def test_min_weight_distance_bound_filter_semantics(self):
        # Every survivor must satisfy OUR pairwise-min-sum reference.
        gd = _stub_abelian_gd()
        out = list(random_weight_patterns(
            (1, 3), 3, 30, rng=np.random.default_rng(3),
            entry_min=1, min_weight_distance_bound=4, gd=gd,
        ))
        assert out, "expected at least one surviving pattern"
        for W in out:
            assert _ref_pairwise_min_sum(W) >= 4, W.tolist()

    def test_all_patterns_exact_contents_vs_reference(self):
        # Exhaustive stream == our own brute-force filter of the space.
        gd = _stub_abelian_gd()
        got = [tuple(W[0]) for W in all_weight_patterns(
            (1, 2), 2, entry_min=0, max_row_weight=3,
            min_weight_distance_bound=2, gd=gd,
        )]
        expected = []
        for a in range(3):
            for b in range(3):
                if a + b <= 3 and (a + b) >= 2:  # row cap AND pairwise sum
                    expected.append((a, b))
        assert got == expected   # contents AND lexicographic order

    def test_all_patterns_lexicographic_order(self):
        got = [tuple(W[0]) for W in all_weight_patterns((1, 2), 1)]
        assert got == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_random_patterns_conjunction_of_filters(self):
        # max_col_weight AND base-girth threshold enforced together; every
        # yield checked against independent predicates.
        out = list(random_weight_patterns(
            (1, 3), 3, 40, rng=np.random.default_rng(8),
            entry_min=0, max_col_weight=2, min_base_girth_bound=8,
        ))
        assert out
        for W in out:
            row = list(W[0])
            assert max(row) <= 2                       # col cap (1 row)
            assert all(x < 3 for x in row)             # girth>=8 kills >=3
            # girth >= 8 also kills "two entries >= 2 in the row" only at
            # threshold > 8; at threshold 8 two 2s are allowed.

    def test_min_base_girth_9_forbids_two_weight2_in_row(self):
        out = list(random_weight_patterns(
            (1, 3), 3, 40, rng=np.random.default_rng(8),
            min_base_girth_bound=9,
        ))
        for W in out:
            row = list(W[0])
            assert all(x < 3 for x in row)
            assert sum(1 for x in row if x >= 2) <= 1, row

    def test_rejecting_filter_yields_nothing(self):
        # entry_min=3 forces every entry >= 3 -> base girth bound 6 < 99
        # -> everything rejected; must terminate and yield nothing.
        out = list(random_weight_patterns(
            (1, 2), 4, 10, rng=np.random.default_rng(0),
            entry_min=3, min_base_girth_bound=99,
        ))
        assert out == []

    def test_max_tries_caps_raw_draws(self):
        out = list(random_weight_patterns(
            (1, 2), 5, 100, rng=np.random.default_rng(0), max_tries=3,
        ))
        assert len(out) <= 3

    def test_dedupe_false_allows_repeats(self):
        # entry_min == entry_max pins the pattern; without dedupe we get
        # num_samples copies, with dedupe just one.
        dup = list(random_weight_patterns(
            (1, 2), 1, 5, rng=np.random.default_rng(0),
            entry_min=1, dedupe=False,
        ))
        assert len(dup) == 5
        assert all(tuple(W[0]) == (1, 1) for W in dup)
        uniq = list(random_weight_patterns(
            (1, 2), 1, 5, rng=np.random.default_rng(0), entry_min=1,
        ))
        assert len(uniq) == 1

