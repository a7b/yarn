"""Fresh-eye contract tests for the canonical brute-force campaign (fast tier).

Black-box tests written against search/canonical/README.md and the module
docstrings. All expectations are derived independently in
``tests/_fresh_helpers.py`` (own GF(2) toolbox, own regular representations,
own Cayley tables); the group is a GAP-free dihedral shim and the heavy
sqetch/bposd estimators are stubbed, so nothing here needs GAP or a GPU.

Covers: build_canonical_code / canonical_logical_basis contracts (shapes,
CSS invariants, anchor conditions, NotSingleOrbit), save_passed_code (dir
layout, npy round-trip, JSON schema, the k_sub hard rule), ScreenParams
knobs in run_group (anchor/free/scan caps, early stop, ranking, per-side
pools, A-vs-B lift orientation), the pair→confirm→certify→save plumbing,
and the sample_search budgets / verdict state machine.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

import _fresh_helpers as fh
from search.canonical import run_group as rg
from search.canonical import sample_search as ss
from search.canonical.build import (
    NotSingleOrbit,
    a_bin_for,
    b_bin_for,
    build_canonical_code,
    canonical_logical_basis,
)
from search.canonical.groups import GroupSpec, enumerate_band
from search.canonical.run_group import (
    ScreenParams,
    _passes,
    _run_record,
    certify_quantum,
    iter_side_survivors,
    pair_and_verify,
    run_group,
    run_group_streaming,
    screen_classical_side,
)
from search.canonical.sample_search import (
    SweepParams,
    pair_sampled,
    run_group_sweep,
    run_group_sweep_streaming,
    sample_classical_side,
)
from search.canonical.save import save_passed_code
from search.sampling._shared.full_rank_block_pool import (
    build_full_rank_block_pool_brute,
    count_weight_elements,
)

pytestmark = pytest.mark.fast


# ─────────────────────────────────────────────────────────────────
# Fixtures (GAP-free)
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def d8():
    g = fh.dihedral_shim(4)          # dihedral group of order 8, non-abelian
    fh.check_group_axioms(g)         # the fixture itself is verified
    assert not g.is_abelian
    return g


@pytest.fixture(scope="module")
def d8_pool(d8):
    pool = build_full_rank_block_pool_brute(d8, 3, force_identity=True)
    assert pool, "D8 must have weight-3 identity units"
    return pool


# Known-single-orbit D8 reference entries (weight-3, unit anchors).
A0, A1 = (0, 1, 3), (0, 1, 2)
B0, B1 = (0, 1, 4), (0, 1, 2)


@pytest.fixture(scope="module")
def ref_code(d8):
    return build_canonical_code(A0, A1, B0, B1, d8)


@pytest.fixture(scope="module")
def ref_basis(ref_code, d8):
    return canonical_logical_basis(ref_code, d8)


# Estimator stubs -----------------------------------------------------------


def _cl_const(d):
    def est(M, **kw):
        return d
    return est


def _q_const(dx, dz):
    def est(Hx, Hz, **kw):
        return (dx, dz)
    return est


class _Counting:
    """Wrap an estimator; count calls and capture positional matrices."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.mats = []

    def __call__(self, M, **kw):
        self.calls += 1
        self.mats.append(np.array(M, copy=True))
        return self.inner(M, **kw)


# ─────────────────────────────────────────────────────────────────
# Pure contracts (no group needed)
# ─────────────────────────────────────────────────────────────────


def test_group_spec_gap_expr_and_tag():
    g = GroupSpec(84, 11)
    assert g.gap_expr == "SmallGroup(84,11)"
    assert g.tag == "Sg84_11"
    # frozen + hashable (used as dict keys / in sets by drivers)
    with pytest.raises(Exception):
        g.order = 85
    assert len({GroupSpec(84, 11), GroupSpec(84, 11), GroupSpec(84, 12)}) == 2


def test_enumerate_band_unknown_band_raises_before_gap():
    # The band-name check must fire before any GAP work.
    with pytest.raises(ValueError, match="unknown band"):
        enumerate_band("d99")


def test_not_single_orbit_is_a_value_error():
    assert issubclass(NotSingleOrbit, ValueError)


def test_pass_criterion_truth_table():
    # PASS = nothing-found (None) OR >= target (strict-< early stop).
    assert _passes(None, 16)
    assert _passes(16, 16)
    assert _passes(17, 16)
    assert not _passes(15, 16)
    assert not _passes(1, 16)


def test_run_record_sqetch_requires_k_sub():
    # The hard rule: a sqetch run without k_sub must not be constructible.
    with pytest.raises(TypeError):
        _run_record("classical_A", "sqetch", num_trials=100, k_sub=None, d=4)
    rec = _run_record("quantum_screen", "sqetch", num_trials=100, k_sub=7,
                      batch_size=10, dx=3, dz=4)
    assert rec["k_sub"] == 7 and rec["num_trials"] == 100
    assert rec["dx"] == 3 and rec["dz"] == 4 and "d" not in rec
    # bposd runs never carry k_sub
    recb = _run_record("bposd_confirm", "bposd", num_trials=50, osd_order=4,
                       dx=None, dz=5)
    assert "k_sub" not in recb and recb["osd_order"] == 4
    assert "dx" not in recb and recb["dz"] == 5   # None distances are omitted


# ─────────────────────────────────────────────────────────────────
# build.py: lifts, shapes, CSS invariants, anchors
# ─────────────────────────────────────────────────────────────────


def test_a_b_bin_use_documented_left_right_reps(d8):
    """a_bin_for must be [L[a0]|L[a1]] and b_bin_for [R[b0]|R[b1]], with L/R
    exactly the README conventions — checked against an independent
    implementation on a NON-abelian group (catches any L/R swap)."""
    n = d8.n
    MA = a_bin_for(A0, A1, d8)
    MB = b_bin_for(B0, B1, d8)
    assert MA.shape == (n, 2 * n) and MB.shape == (n, 2 * n)
    assert np.array_equal(MA, fh.my_A_bin([A0, A1], d8))
    assert np.array_equal(MB, fh.my_B_bin([B0, B1], d8))
    # L and R genuinely differ on the supports used above (B0 = {e, r, f} is
    # not conjugation-invariant), so the checks are sharp against an L/R swap.
    assert not np.array_equal(fh.my_left((1,), d8), fh.my_right((1,), d8))
    assert not np.array_equal(fh.my_left(B0, d8), fh.my_right(B0, d8))


def test_a_bin_for_canonicalizes_mod2(d8):
    # duplicate entries cancel mod 2: (1,1,2) ≡ (2,)
    assert np.array_equal(a_bin_for((1, 1, 2), A1, d8),
                          a_bin_for((2,), A1, d8))
    assert np.array_equal(b_bin_for((3, 2, 3), B1, d8),
                          b_bin_for((2,), B1, d8))


def test_build_shapes_and_parameters(ref_code, d8):
    n = d8.n
    assert ref_code["n_phys"] == 5 * n                     # shape (1,2) → 5|G|
    assert ref_code["Hx"].shape == (2 * n, 5 * n)
    assert ref_code["Hz"].shape == (2 * n, 5 * n)
    assert ref_code["A_bin"].shape == (n, 2 * n)
    assert ref_code["B_bin"].shape == (n, 2 * n)
    # k from my own rank computation
    k_mine = 5 * n - fh.rank2(ref_code["Hx"]) - fh.rank2(ref_code["Hz"])
    assert ref_code["k"] == k_mine == n                    # single orbit


def test_css_orthogonality_independent(ref_code):
    Hx = ref_code["Hx"].astype(np.int64)
    Hz = ref_code["Hz"].astype(np.int64)
    assert np.all((Hx @ Hz.T) % 2 == 0)
    assert ref_code["is_css"] is True


def test_check_weights_match_definition(ref_code):
    # every row of Hx has weight |a0|+|a1| ⊕-free = 6 plus one B†-block col
    # weight 3 → 9; Hz symmetric. Verify against a direct row-sum.
    assert ref_code["max_Hx_check_weight"] == int(ref_code["Hx"].sum(axis=1).max())
    assert ref_code["max_Hz_check_weight"] == int(ref_code["Hz"].sum(axis=1).max())


def test_anchor_conditions_and_block_ranks(ref_code, d8):
    n = d8.n
    # identity in the support of both anchors, anchors at the LAST block-col
    assert ref_code["A_canonical"][0][1] == A1 and 0 in A1
    assert ref_code["B_canonical"][0][1] == B1 and 0 in B1
    # unit anchors: rank L[a1] = rank R[b1] = |G| (independent rank)
    assert fh.rank2(ref_code["A_bin"][:, n:]) == n
    assert fh.rank2(ref_code["B_bin"][:, n:]) == n
    assert ref_code["has_full_rank_a"] and ref_code["has_full_rank_b"]
    # canonicalization is idempotent when a1 is already the unit: no permute
    assert ref_code["qcode"]["perm_a"] == [0, 1]
    assert ref_code["qcode"]["perm_b"] == [0, 1]


def test_rank_L_equals_rank_R_pool_symmetry(d8):
    # README: one pool serves both sides because rank L[x] == rank R[x].
    for x in [(0, 1, 2), (0, 1, 3), (1, 2, 5), (0, 3, 4, 6), (2,)]:
        assert fh.rank2(fh.my_left(x, d8)) == fh.rank2(fh.my_right(x, d8))


def test_non_unit_anchor_gets_permuted_to_last(d8):
    """If a1 is not a unit but a0 is, the canonical form must move the unit
    into the LAST block-col (canonical-by-construction contract)."""
    a0_unit, a1_nonunit = (0, 1, 2), (1, 2)       # even weight → not a unit
    assert fh.rank2(fh.my_left(a1_nonunit, d8)) < d8.n
    code = build_canonical_code(a0_unit, a1_nonunit, B0, B1, d8)
    assert code["has_full_rank_a"] is True
    assert code["qcode"]["perm_a"] == [1, 0]
    assert code["A_canonical"][0] == [a1_nonunit, a0_unit]
    n = d8.n
    assert fh.rank2(code["A_bin"][:, n:]) == n     # unit ended up last


def test_logical_basis_full_contract(ref_code, ref_basis, d8):
    n = d8.n
    Lx, Lz = ref_basis["Lx"], ref_basis["Lz"]
    Hx, Hz = ref_code["Hx"], ref_code["Hz"]
    assert Lx.shape == (n, 5 * n) and Lz.shape == (n, 5 * n)
    assert ref_basis["k"] == n
    # Lx rows in ker(Hz); Lz rows in ker(Hx)
    assert np.all((Hz.astype(np.int64) @ Lx.T.astype(np.int64)) % 2 == 0)
    assert np.all((Hx.astype(np.int64) @ Lz.T.astype(np.int64)) % 2 == 0)
    # Lx · Lzᵀ = I (independent computation, plus the reported flag)
    P = (Lx.astype(np.int64) @ Lz.T.astype(np.int64)) % 2
    assert np.array_equal(P, np.eye(n, dtype=np.int64))
    assert ref_basis["Lx_dot_LzT_is_I"] is True
    # rows are independent AND independent of the stabilizers (true logicals)
    assert fh.rank2(Lx) == n
    assert fh.rank2(np.vstack([Hx, Lx])) == fh.rank2(Hx) + n
    assert fh.rank2(np.vstack([Hz, Lz])) == fh.rank2(Hz) + n
    # row-weight summaries match a direct recount; single orbit → uniform
    for L, summ in ((Lx, ref_basis["Lx_weight"]), (Lz, ref_basis["Lz_weight"])):
        w = sorted(set(int(v) for v in L.sum(axis=1)))
        assert summ["values"] == w
        assert summ["min"] == w[0] and summ["max"] == w[-1]
        assert summ["uniform"] == (w[0] if len(w) == 1 else None)
        assert summ["uniform"] is not None


def test_not_single_orbit_raised_for_k_not_multiple(d8):
    # k = 9 and k = 10 (not multiples of 8) — both must raise
    for entries in [((0, 1), (0, 4), (0, 1, 2), (0, 1, 2)),
                    ((0, 1), (0, 1), (0, 1, 2), (0, 1, 2))]:
        code = build_canonical_code(*entries, d8)
        assert code["k"] != d8.n
        with pytest.raises(NotSingleOrbit):
            canonical_logical_basis(code, d8)


def test_not_single_orbit_raised_for_k_2n_no_anchor(d8):
    # k = 16 = 2|G| (a strict multiple — multi-orbit) must also raise
    code = build_canonical_code((0, 1), (0, 1), (0, 1), (0, 1), d8)
    assert code["k"] == 2 * d8.n
    with pytest.raises(NotSingleOrbit):
        canonical_logical_basis(code, d8)


def test_single_orbit_boundary_does_not_raise(d8):
    # k == |G| exactly → the basis must build (no spurious raise).
    basis = canonical_logical_basis(build_canonical_code(A0, A1, B0, B1, d8), d8)
    assert basis["k"] == d8.n


# ─────────────────────────────────────────────────────────────────
# save.py: directory layout, npy round-trip, JSON schema
# ─────────────────────────────────────────────────────────────────


def _runs_ok():
    return [
        _run_record("classical_A", "sqetch", num_trials=1000, k_sub=8,
                    batch_size=100, d=4),
        _run_record("classical_B", "sqetch", num_trials=1000, k_sub=8,
                    batch_size=100, d=None),
        _run_record("quantum_screen", "sqetch", num_trials=200, k_sub=12,
                    batch_size=10, dx=3, dz=4),
        _run_record("bposd_confirm", "bposd", num_trials=50, osd_order=4,
                    dx=3, dz=4),
        _run_record("certify", "sqetch", num_trials=500, k_sub=12,
                    batch_size=10, dx=3, dz=4, seed=1),
    ]


def test_save_dir_layout_and_npy_roundtrip(tmp_path, ref_code, ref_basis, d8):
    d = save_passed_code(
        tmp_path, gd=d8, gap_expr="shim-d8", tag="Sg8_3", code=ref_code,
        basis=ref_basis, dx=3, dz=4, classical_dA=4, classical_dB=4,
        distance_runs=_runs_ok(), provenance={"who": "fresh"})
    assert d.parent == tmp_path
    assert re.fullmatch(r"Sg8_3__[0-9a-f]{12}", d.name)
    for fn, arr in (("Hx.npy", ref_code["Hx"]), ("Hz.npy", ref_code["Hz"]),
                    ("A_bin.npy", ref_code["A_bin"]),
                    ("B_bin.npy", ref_code["B_bin"]),
                    ("Lx.npy", ref_basis["Lx"]), ("Lz.npy", ref_basis["Lz"])):
        back = np.load(d / fn)
        assert back.shape == arr.shape
        assert np.array_equal(back, arr), fn
    # deterministic naming: a re-save of the same code goes to the SAME dir
    d2 = save_passed_code(
        tmp_path, gd=d8, gap_expr="shim-d8", tag="Sg8_3", code=ref_code,
        basis=ref_basis, dx=3, dz=4, classical_dA=4, classical_dB=4,
        distance_runs=_runs_ok())
    assert d2 == d
    # ...and a different ring gets a different hash
    other = build_canonical_code((0, 1, 4), A1, B0, B1, d8)
    ob = canonical_logical_basis(other, d8)
    d3 = save_passed_code(
        tmp_path, gd=d8, gap_expr="shim-d8", tag="Sg8_3", code=other,
        basis=ob, dx=3, dz=4, classical_dA=4, classical_dB=4,
        distance_runs=_runs_ok())
    assert d3.name != d.name


def test_save_json_schema_and_k_sub_hard_rule(tmp_path, ref_code, ref_basis, d8):
    n = d8.n
    d = save_passed_code(
        tmp_path, gd=d8, gap_expr="shim-d8", tag="Sg8_3", code=ref_code,
        basis=ref_basis, dx=3, dz=4, classical_dA=4, classical_dB=2,
        distance_runs=_runs_ok(), provenance={"who": "fresh"})
    meta = json.loads((d / "code.json").read_text())
    # top-level schema
    for key in ("code", "group", "shape", "ring", "canonical", "logical_basis",
                "distance", "check_weight", "files", "provenance", "timestamp"):
        assert key in meta, key
    assert meta["shape"] == [1, 2]
    assert meta["code"] == {"n": 5 * n, "k": n, "dx": 3, "dz": 4,
                            "label": f"[[{5*n},{n},3]]"}
    assert meta["group"]["order"] == n and meta["group"]["abelian"] is False
    assert meta["group"]["gap_expr"] == "shim-d8" and meta["group"]["tag"] == "Sg8_3"
    # ring section: anchors, identity flags, weights
    assert meta["ring"]["anchor_col"] == 1
    assert meta["ring"]["identity_in_anchor_A"] is True
    assert meta["ring"]["identity_in_anchor_B"] is True
    assert meta["ring"]["A"] == [[list(A0), list(A1)]]
    assert meta["ring"]["B"] == [[list(B0), list(B1)]]
    assert meta["ring"]["weight_A"] == [[3, 3]] and meta["ring"]["weight_B"] == [[3, 3]]
    # canonical block: block-col ranks recomputed independently
    assert meta["canonical"]["block_col_ranks_A"] == [
        fh.rank2(ref_code["A_bin"][:, :n]), fh.rank2(ref_code["A_bin"][:, n:])]
    assert meta["canonical"]["block_col_ranks_B"] == [
        fh.rank2(ref_code["B_bin"][:, :n]), fh.rank2(ref_code["B_bin"][:, n:])]
    assert meta["canonical"]["block_col_ranks_A"][-1] == n
    assert meta["canonical"]["single_orbit"] is True
    assert meta["canonical"]["is_css"] is True
    # distance block: classical + verdict + verbatim runs
    assert meta["distance"]["classical_dA"] == 4
    assert meta["distance"]["classical_dB"] == 2
    assert meta["distance"]["verdict"] == "pass"
    runs = meta["distance"]["runs"]
    assert len(runs) == 5
    assert any(r["stage"] == "certify" and r.get("seed") == 1 for r in runs)
    # THE hard rule: every sqetch run carries BOTH num_trials and k_sub
    for r in runs:
        assert "num_trials" in r
        if r["backend"] == "sqetch":
            assert isinstance(r["k_sub"], int) and r["k_sub"] >= 1
    # files map matches what's on disk
    for fn in meta["files"].values():
        assert (d / fn).exists()
    assert meta["provenance"] == {"who": "fresh"}
    assert isinstance(meta["timestamp"], int)


def test_save_label_handles_missing_distances(tmp_path, ref_code, ref_basis, d8):
    n = d8.n
    # dx=None → label uses dz
    d = save_passed_code(tmp_path / "a", gd=d8, gap_expr="g", tag="T",
                         code=ref_code, basis=ref_basis, dx=None, dz=5,
                         classical_dA=None, classical_dB=None, distance_runs=[])
    meta = json.loads((d / "code.json").read_text())
    assert meta["code"]["label"] == f"[[{5*n},{n},5]]"
    assert meta["code"]["dx"] is None and meta["code"]["dz"] == 5
    # both None → label without a distance at all
    d = save_passed_code(tmp_path / "b", gd=d8, gap_expr="g", tag="T",
                         code=ref_code, basis=ref_basis, dx=None, dz=None,
                         classical_dA=None, classical_dB=None, distance_runs=[])
    meta = json.loads((d / "code.json").read_text())
    assert meta["code"]["label"] == f"[[{5*n},{n}]]"


# ─────────────────────────────────────────────────────────────────
# run_group.py: classical screen knobs
# ─────────────────────────────────────────────────────────────────


def test_screen_full_brute_bounded_by_anchor_and_free_limits(d8, d8_pool, monkeypatch):
    est = _Counting(_cl_const(2))                      # below target → no survivor
    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", est)
    p = ScreenParams(target=16, anchor_limit=2, free_limit=3, cl_early_stop=False)
    surv = screen_classical_side(d8, d8_pool, p, "A")
    assert surv == []
    assert est.calls == 2 * 3                          # anchors × free cap exactly


def test_screen_without_limits_walks_all_free_elements(d8, d8_pool, monkeypatch):
    est = _Counting(_cl_const(2))
    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", est)
    p = ScreenParams(target=16, anchor_limit=1, cl_early_stop=False)
    screen_classical_side(d8, d8_pool, p, "A")
    # free columns brute over ALL weight-3 elements (not identity-forced)
    assert est.calls == count_weight_elements(d8.n, 3) == 56


def test_screen_scan_cap_binds_total_work(d8, d8_pool, monkeypatch):
    est = _Counting(_cl_const(2))
    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", est)
    p = ScreenParams(target=16, cl_scan_cap=7, cl_early_stop=False)
    screen_classical_side(d8, d8_pool, p, "A")
    assert est.calls == 7


def test_screen_early_stop_and_per_side_pool_caps(d8, d8_pool, monkeypatch):
    estA = _Counting(_cl_const(None))                  # everything passes
    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", estA)
    p = ScreenParams(target=16, max_A_pool=4, max_B_pool=2)
    survA = screen_classical_side(d8, d8_pool, p, "A")
    assert len(survA) == 4 and estA.calls == 4         # stops the moment A fills
    estB = _Counting(_cl_const(None))
    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", estB)
    survB = screen_classical_side(d8, d8_pool, p, "B")
    assert len(survB) == 2 and estB.calls == 2         # B uses ITS own cap


def test_screen_sides_use_left_and_right_lifts(d8, d8_pool, monkeypatch):
    """Side A must screen [L[a0]|L[a1]] matrices; side B [R[b0]|R[b1]].
    On a non-abelian group these differ, so a lift mix-up fails here."""
    n = d8.n
    p = ScreenParams(target=16, anchor_limit=2, free_limit=2, cl_early_stop=False)
    for side, my_lift, decode in (("A", fh.my_A_bin, fh.decode_A_side),
                                  ("B", fh.my_B_bin, fh.decode_B_side)):
        est = _Counting(_cl_const(2))
        monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", est)
        screen_classical_side(d8, d8_pool, p, side)
        assert est.calls == 4
        seen = []
        for M in est.mats:
            free, anchor = decode(M, n)
            assert np.array_equal(M, my_lift([free, anchor], d8))
            seen.append((free, anchor))
        # anchors from the pool prefix; free in brute enumeration order
        expect = [(f, a) for a in [tuple(x) for x in d8_pool[:2]]
                  for f in [(0, 1, 2), (0, 1, 3)]]
        assert seen == expect


def test_screen_survivor_constraints_and_k_sub(d8, d8_pool, monkeypatch):
    n = d8.n
    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", _cl_const(None))
    p = ScreenParams(target=16, max_A_pool=3, max_B_pool=3)
    surv = screen_classical_side(d8, d8_pool, p, "A")
    assert len(surv) == 3
    for s in surv:
        assert set(s) == {"free", "anchor", "d", "k_sub"}
        assert len(s["free"]) == 3                     # free weight
        assert len(s["anchor"]) == 3 and 0 in s["anchor"]   # identity in anchor
        assert fh.rank2(fh.my_left(s["anchor"], d8)) == n   # unit anchor
        assert s["d"] is None or s["d"] >= p.target
        assert s["k_sub"] == n                         # cl_k_sub=0 ⇒ ker dim = |G|


def test_screen_k_sub_resolution_and_cap(d8, d8_pool, monkeypatch):
    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", _cl_const(None))
    monkeypatch.setattr(rg, "max_k_sub_for", lambda *a, **k: 5)
    p = ScreenParams(target=16, max_A_pool=1, max_B_pool=1)
    surv = screen_classical_side(d8, d8_pool, p, "A")
    assert surv[0]["k_sub"] == 5                       # GPU ceiling caps ker dim
    p2 = ScreenParams(target=16, max_A_pool=1, max_B_pool=1, cl_k_sub=3)
    surv2 = screen_classical_side(d8, d8_pool, p2, "A")
    assert surv2[0]["k_sub"] == 3                      # explicit value honored


def test_screen_ranking_top_n_none_best(d8, d8_pool, monkeypatch):
    """Full brute keeps the top ``keep`` by distance, None (nothing found)
    ranking best of all."""
    n = d8.n
    frees = [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5), (0, 1, 6)]
    d_map = {frees[0]: 17, frees[1]: None, frees[2]: 20,
             frees[3]: 15, frees[4]: 19}               # 15 < target → dropped

    def est(M, **kw):
        free, _anchor = fh.decode_A_side(M, n)
        return d_map[free]

    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", est)
    p = ScreenParams(target=16, anchor_limit=1, free_limit=5,
                     cl_early_stop=False, max_A_pool=3, max_B_pool=3)
    surv = screen_classical_side(d8, d8_pool, p, "A")
    assert [s["d"] for s in surv] == [None, 20, 19]
    assert [s["free"] for s in surv] == [frees[1], frees[2], frees[4]]


def test_iter_side_survivors_streams_scan_order(d8, d8_pool, monkeypatch):
    est = _Counting(_cl_const(None))
    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", est)
    p = ScreenParams(target=16, anchor_limit=1, free_limit=4, cl_early_stop=False)
    got = list(iter_side_survivors(d8, d8_pool, p, "A"))
    assert est.calls == 4
    assert [s["free"] for s in got] == [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5)]
    assert all(s["anchor"] == tuple(d8_pool[0]) for s in got)
    # scan cap: generator returns after exactly cl_scan_cap estimator calls
    est2 = _Counting(_cl_const(None))
    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", est2)
    p2 = ScreenParams(target=16, cl_scan_cap=3, cl_early_stop=False)
    got2 = list(iter_side_survivors(d8, d8_pool, p2, "A"))
    assert est2.calls == 3 and len(got2) == 3


# ─────────────────────────────────────────────────────────────────
# run_group.py: pair → screen → confirm → certify → save
# ─────────────────────────────────────────────────────────────────


def _surv(free, anchor, d=16, k_sub=8):
    return {"free": free, "anchor": anchor, "d": d, "k_sub": k_sub}


def _patch_quantum(monkeypatch, *, screen=(None, None), bposd=(None, None),
                   certify=None):
    """Stub the three estimator entry points; certify keyed by seed."""
    calls = {"screen": 0, "bposd": 0, "certify": 0}

    def sqetch(Hx, Hz, **kw):
        if "seed" in kw:
            calls["certify"] += 1
            return certify[kw["seed"]] if certify else (None, None)
        calls["screen"] += 1
        return screen

    def bp(Hx, Hz, **kw):
        calls["bposd"] += 1
        return bposd

    monkeypatch.setattr(rg, "estimate_quantum_distances_sqetch", sqetch)
    monkeypatch.setattr(rg, "estimate_quantum_distances_bposd", bp)
    return calls


def test_pair_and_verify_saved_min_and_run_records(d8, tmp_path, monkeypatch):
    """Saved dx/dz must be the MIN over screen, bposd and every certify seed;
    the JSON must carry one record per stage with k_sub on every sqetch one."""
    monkeypatch.setattr(rg, "max_k_sub_for", lambda *a, **k: 10)
    calls = _patch_quantum(monkeypatch, screen=(6, 7), bposd=(5, 9),
                           certify={1: (6, 6), 2: (4, 8)})
    A = [_surv(A0, A1, d=16)]
    B = [_surv(B0, B1, d=None)]
    p = ScreenParams(target=2, certify_seeds=[1, 2])
    res = pair_and_verify(d8, A, B, p, tmp_path, gap_expr="shim", tag="T",
                          provenance={"p": 1})
    assert len(res["passed"]) == 1
    assert res["n_pairs"] == 1 and res["n_single_orbit"] == 1
    assert res["best_dx_seen"] == 6 and res["best_dz_seen"] == 7  # screen values
    assert calls == {"screen": 1, "bposd": 1, "certify": 2}
    meta = json.loads((Path(res["passed"][0]) / "code.json").read_text())
    assert meta["distance"]["dx"] == 4 and meta["distance"]["dz"] == 6
    assert meta["code"]["label"] == f"[[{5*d8.n},{d8.n},4]]"
    runs = meta["distance"]["runs"]
    stages = [r["stage"] for r in runs]
    assert stages == ["classical_A", "classical_B", "quantum_screen",
                      "bposd_confirm", "certify", "certify"]
    by_stage = {}
    for r in runs:
        by_stage.setdefault(r["stage"], []).append(r)
    assert by_stage["classical_A"][0]["k_sub"] == 8
    assert by_stage["classical_A"][0]["d"] == 16
    assert "d" not in by_stage["classical_B"][0]        # None d is omitted
    assert by_stage["quantum_screen"][0]["k_sub"] == 10  # q_k_sub capped by ceiling
    assert [r["seed"] for r in by_stage["certify"]] == [1, 2]
    assert all(r["k_sub"] == 10 for r in by_stage["certify"])
    assert by_stage["bposd_confirm"][0]["osd_order"] == p.bposd_osd_order
    for r in runs:
        if r["backend"] == "sqetch":
            assert "k_sub" in r and "num_trials" in r
    # saved dx/dz self-consistent with the recorded runs
    rec_dx = [r["dx"] for r in runs if "dx" in r]
    rec_dz = [r["dz"] for r in runs if "dz" in r]
    assert meta["distance"]["dx"] == min(rec_dx)
    assert meta["distance"]["dz"] == min(rec_dz)


def test_pair_screen_fail_fast_skips_bposd(d8, tmp_path, monkeypatch):
    calls = _patch_quantum(monkeypatch, screen=(1, 5))   # dx < target
    p = ScreenParams(target=2)
    res = pair_and_verify(d8, [_surv(A0, A1)], [_surv(B0, B1)], p, tmp_path,
                          gap_expr="shim", tag="T")
    assert res["passed"] == []
    assert calls["bposd"] == 0 and calls["certify"] == 0   # fail-fast
    assert res["n_single_orbit"] == 1
    assert res["best_dx_seen"] == 1 and res["best_dz_seen"] == 5
    assert list(tmp_path.iterdir()) == []                  # nothing saved


def test_pair_bposd_fail_skips_certify(d8, tmp_path, monkeypatch):
    calls = _patch_quantum(monkeypatch, screen=(None, None), bposd=(1, None))
    p = ScreenParams(target=2)
    res = pair_and_verify(d8, [_surv(A0, A1)], [_surv(B0, B1)], p, tmp_path,
                          gap_expr="shim", tag="T")
    assert res["passed"] == [] and calls["certify"] == 0
    assert calls["bposd"] == 1


def test_pair_certify_refutes(d8, tmp_path, monkeypatch):
    _patch_quantum(monkeypatch, screen=(None, None), bposd=(None, None),
                   certify={1: (None, None), 2: (None, 1)})
    p = ScreenParams(target=2, certify_seeds=[1, 2])
    res = pair_and_verify(d8, [_surv(A0, A1)], [_surv(B0, B1)], p, tmp_path,
                          gap_expr="shim", tag="T")
    assert res["passed"] == []                             # seed 2 refuted dz
    assert list(tmp_path.iterdir()) == []


def test_pair_max_quantum_pass_caps_grid(d8, tmp_path, monkeypatch):
    _patch_quantum(monkeypatch, screen=(None, None), bposd=(None, None),
                   certify={1: (None, None)})
    A = [_surv((0, 1, 2), A1), _surv((0, 1, 3), A1)]
    B = [_surv((0, 1, 4), B1), _surv((0, 1, 5), B1), _surv((0, 1, 6), B1)]
    p = ScreenParams(target=2, certify_seeds=[1], max_quantum_pass=2)
    res = pair_and_verify(d8, A, B, p, tmp_path, gap_expr="shim", tag="T")
    assert len(res["passed"]) == 2 and res["n_pairs"] == 2   # stops mid-grid
    # without the cap the full 2×3 grid is evaluated
    p2 = ScreenParams(target=2, certify_seeds=[1], max_quantum_pass=50)
    res2 = pair_and_verify(d8, A, B, p2, tmp_path / "full", gap_expr="shim",
                           tag="T")
    assert res2["n_pairs"] == 6 and len(res2["passed"]) == 6


def test_pair_check_weight_gate(d8, tmp_path, monkeypatch):
    calls = _patch_quantum(monkeypatch, screen=(None, None), bposd=(None, None))
    # weight-3 (1,2) rings → max check weight 9; cap 8 gates EVERYTHING out
    p = ScreenParams(target=2, max_check_weight=8, certify_enabled=False)
    res = pair_and_verify(d8, [_surv(A0, A1)], [_surv(B0, B1)], p, tmp_path,
                          gap_expr="shim", tag="T")
    assert res["n_pairs"] == 1 and res["n_single_orbit"] == 0
    assert calls["screen"] == 0 and res["passed"] == []
    # cap 9 lets the pair through
    p2 = ScreenParams(target=2, max_check_weight=9, certify_enabled=False)
    res2 = pair_and_verify(d8, [_surv(A0, A1)], [_surv(B0, B1)], p2, tmp_path,
                           gap_expr="shim", tag="T")
    assert res2["n_single_orbit"] == 1 and len(res2["passed"]) == 1


def test_pair_q_floor_below_target_keeps_best_effort(d8, tmp_path, monkeypatch):
    # quantum (3,3) with target 4: rejected by default, kept with q_floor=3
    _patch_quantum(monkeypatch, screen=(3, 3), bposd=(3, 3))
    p = ScreenParams(target=4, certify_enabled=False)
    res = pair_and_verify(d8, [_surv(A0, A1)], [_surv(B0, B1)], p,
                          tmp_path / "strict", gap_expr="shim", tag="T")
    assert res["passed"] == []
    p2 = ScreenParams(target=4, q_floor=3, certify_enabled=False)
    res2 = pair_and_verify(d8, [_surv(A0, A1)], [_surv(B0, B1)], p2,
                           tmp_path / "floor", gap_expr="shim", tag="T")
    assert len(res2["passed"]) == 1
    meta = json.loads((Path(res2["passed"][0]) / "code.json").read_text())
    assert meta["distance"]["dx"] == 3 and meta["distance"]["dz"] == 3


def test_certify_quantum_min_over_seeds_and_records(d8, ref_code, ref_basis,
                                                    monkeypatch):
    per_seed = {1: (None, 5), 2: (7, None), 3: (6, 9)}

    def sqetch(Hx, Hz, **kw):
        return per_seed[kw["seed"]]

    monkeypatch.setattr(rg, "estimate_quantum_distances_sqetch", sqetch)
    p = ScreenParams(target=2, certify_seeds=[1, 2, 3])
    cdx, cdz, runs = certify_quantum(ref_code, ref_basis, p, 0, k_sub=13)
    assert cdx == 6 and cdz == 5                       # min over seeds, None skipped
    assert [r["seed"] for r in runs] == [1, 2, 3]
    assert all(r["stage"] == "certify" and r["backend"] == "sqetch" for r in runs)
    assert all(r["k_sub"] == 13 for r in runs)
    assert runs[0]["dz"] == 5 and "dx" not in runs[0]  # None fields omitted


# ─────────────────────────────────────────────────────────────────
# run_group.py: top-level drivers + stats persistence
# ─────────────────────────────────────────────────────────────────


def test_run_group_full_funnel_with_stats(d8, tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", _cl_const(4))
    _patch_quantum(monkeypatch, screen=(None, None), bposd=(None, None),
                   certify={1: (None, None)})
    g = GroupSpec(8, 3)
    p = ScreenParams(target=2, max_A_pool=2, max_B_pool=2, anchor_limit=1,
                     free_limit=3, max_quantum_pass=2, certify_seeds=[1])
    res = run_group(g, p, tmp_path, gd=d8, provenance={"run": 1})
    assert res["group"] == "Sg8_3" and res["order"] == 8
    assert res["pool_size"] == len(fh.identity_unit_supports(d8, 3))  # = my brute
    assert res["n_A_survivors"] == 2 and res["n_B_survivors"] == 2
    assert res["A_d_classical"] == [4] and res["B_d_classical"] == [4]
    assert res["n_pairs"] == 2 and res["n_single_orbit"] == 2
    assert len(res["passed"]) == 2
    stats = json.loads((tmp_path / "_stats_Sg8_3.json").read_text())
    assert stats["n_passed"] == 2
    assert stats["passed"] == res["passed"]
    assert stats["target"] == 2


def test_run_group_zero_passes_still_saves_stats(d8, tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", _cl_const(4))
    _patch_quantum(monkeypatch, screen=(1, 1))          # everything refuted
    p = ScreenParams(target=2, max_A_pool=1, max_B_pool=1, anchor_limit=1,
                     free_limit=1)
    res = run_group(GroupSpec(8, 3), p, tmp_path, gd=d8)
    assert res["passed"] == []
    stats = json.loads((tmp_path / "_stats_Sg8_3.json").read_text())
    assert stats["n_passed"] == 0
    assert stats["best_dx_seen"] == 1 and stats["best_dz_seen"] == 1


def test_run_group_no_units_group_short_circuits(tmp_path):
    """S3 (= dihedral of order 6) has NO weight-3 units: the funnel must
    report pool_size 0, never call an estimator, and still save stats."""
    s3 = fh.dihedral_shim(3)
    fh.check_group_axioms(s3)
    assert fh.identity_unit_supports(s3, 3) == []       # independent oracle

    def boom(*a, **k):
        raise AssertionError("estimator must not be called")

    import unittest.mock as mock
    with mock.patch.object(rg, "estimate_classical_distance_sqetch", boom), \
         mock.patch.object(rg, "estimate_quantum_distances_sqetch", boom):
        res = run_group(GroupSpec(6, 1), ScreenParams(target=2), tmp_path, gd=s3)
    assert res["pool_size"] == 0 and res["passed"] == []
    assert "note" in res
    stats = json.loads((tmp_path / "_stats_Sg6_1.json").read_text())
    assert stats["n_passed"] == 0 and stats["pool_size"] == 0


def test_run_group_streaming_matches_batch_first_passer(d8, tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "estimate_classical_distance_sqetch", _cl_const(4))
    _patch_quantum(monkeypatch, screen=(None, None), bposd=(None, None),
                   certify={1: (None, None)})
    p = ScreenParams(target=2, max_A_pool=2, max_B_pool=2, anchor_limit=1,
                     free_limit=3, max_quantum_pass=1, certify_seeds=[1])
    rb = run_group(GroupSpec(8, 3), p, tmp_path / "batch", gd=d8)
    rs = run_group_streaming(GroupSpec(8, 3), p, tmp_path / "stream", gd=d8)
    assert len(rb["passed"]) == len(rs["passed"]) == 1
    assert Path(rb["passed"][0]).name == Path(rs["passed"][0]).name
    assert (tmp_path / "stream" / "_stats_Sg8_3.json").exists()
    assert rs["n_pairs"] == 1 and rs["n_single_orbit"] == 1


# ─────────────────────────────────────────────────────────────────
# sample_search.py: budgets, verdicts, PASS criterion
# ─────────────────────────────────────────────────────────────────


def test_sweep_classical_budget_exact_call_count(d8, d8_pool):
    calls = {"n": 0}

    def est(M, **kw):
        calls["n"] += 1
        return 2                                        # never a survivor

    p = SweepParams(target=16, cl_sample_budget=25, cl_barren_skip=10**9,
                    surv_cap=10**9)
    surv, status = sample_classical_side(
        d8, d8_pool, p, "A", rng=np.random.default_rng(0),
        classical_estimator=est)
    assert calls["n"] == 25                             # budget binds exactly
    assert surv == [] and status == "barren"


def test_sweep_barren_skip_fires_before_budget(d8, d8_pool):
    calls = {"n": 0}

    def est(M, **kw):
        calls["n"] += 1
        return 2

    p = SweepParams(target=16, cl_sample_budget=1000, cl_barren_skip=12)
    surv, status = sample_classical_side(
        d8, d8_pool, p, "B", rng=np.random.default_rng(0),
        classical_estimator=est)
    assert status == "barren" and surv == []
    assert calls["n"] == 12                             # stops AT the skip mark


def test_sweep_classical_k_sub_resolution(d8, d8_pool, monkeypatch):
    # cl_k_sub=0 ⇒ ker dim (=|G|); the shmem ceiling caps it when smaller
    monkeypatch.setattr(ss, "_resolve_max_k_sub", lambda n_cols: 100)
    assert ss._classical_k_sub(d8, SweepParams(cl_k_sub=0)) == d8.n
    assert ss._classical_k_sub(d8, SweepParams(cl_k_sub=3)) == 3
    monkeypatch.setattr(ss, "_resolve_max_k_sub", lambda n_cols: 5)
    assert ss._classical_k_sub(d8, SweepParams(cl_k_sub=0)) == 5
    # quantum: 0 ⇒ the ceiling itself; explicit values clamped to it
    assert ss._quantum_k_sub(d8, SweepParams(q_k_sub=0)) == 5
    assert ss._quantum_k_sub(d8, SweepParams(q_k_sub=3)) == 3
    assert ss._quantum_k_sub(d8, SweepParams(q_k_sub=10**6)) == 5


@pytest.mark.parametrize("dx,dz,should_pass", [
    (None, None, True),           # nothing found in either direction
    (2, 2, True),                 # both at target
    (2, None, True),              # mixed None / at-target
    (None, 1, False),             # dz below target refutes
    (1, 2, False),                # dx below target refutes
])
def test_sweep_pass_criterion_each_direction(d8, d8_pool, tmp_path, dx, dz,
                                             should_pass):
    A = [{"free": A0, "anchor": A1, "d": 16, "k_sub": 8}]
    B = [{"free": B0, "anchor": B1, "d": 16, "k_sub": 8}]
    p = SweepParams(target=2, max_pass=1, pair_budget=1, pair_fail=100)
    res = pair_sampled(d8, A, B, p, tmp_path, rng=np.random.default_rng(0),
                       gap_expr="shim", tag="T",
                       quantum_estimator=_q_const(dx, dz))
    assert res["n_pairs"] == 1
    if should_pass:
        assert res["verdict"] == "pass" and len(res["passed"]) == 1
        meta = json.loads((Path(res["passed"][0]) / "code.json").read_text())
        assert meta["distance"]["dx"] == dx and meta["distance"]["dz"] == dz
        # the single quantum_screen record carries the hard-rule fields
        qs = [r for r in meta["distance"]["runs"] if r["stage"] == "quantum_screen"]
        assert len(qs) == 1 and "k_sub" in qs[0] and "num_trials" in qs[0]
    else:
        assert res["verdict"] == "budget" and res["passed"] == []
        assert res["best_dx_seen"] == dx and res["best_dz_seen"] == dz


def test_sweep_gate_skip_costs_no_budget(d8, d8_pool, tmp_path, monkeypatch):
    """Non-single-orbit pairs must be skipped WITHOUT consuming the quantum
    budget and without calling the estimator."""
    calls = {"n": 0}

    def est(Hx, Hz, **kw):
        calls["n"] += 1
        return (None, None)

    # force the gate shut: pretend k != |G| for every pair
    monkeypatch.setattr(ss, "screen_basis",
                        lambda Hx, Hz: (np.zeros((1, Hx.shape[1]), np.uint8),
                                        np.zeros((1, Hx.shape[1]), np.uint8),
                                        d8.n - 1))
    A = [{"free": (0, 1, 2), "anchor": A1, "d": 16, "k_sub": 8},
         {"free": (0, 1, 3), "anchor": A1, "d": 16, "k_sub": 8}]
    B = [{"free": (0, 1, 4), "anchor": B1, "d": 16, "k_sub": 8}]
    p = SweepParams(target=2, max_pass=1, pair_budget=100, pair_fail=50,
                    max_consecutive_misses=10)
    res = pair_sampled(d8, A, B, p, tmp_path, rng=np.random.default_rng(0),
                       gap_expr="shim", tag="T", quantum_estimator=est)
    assert res["verdict"] == "exhausted"
    assert res["n_pairs"] == 0 and calls["n"] == 0
    assert res["n_gate_skip"] >= 1 and res["passed"] == []


def test_sweep_exhausted_when_pair_space_saturates(d8, tmp_path):
    # 1×1 survivors: after the single unique pair everything is a repeat
    A = [{"free": A0, "anchor": A1, "d": 16, "k_sub": 8}]
    B = [{"free": B0, "anchor": B1, "d": 16, "k_sub": 8}]
    p = SweepParams(target=2, max_pass=5, pair_budget=1000, pair_fail=100,
                    max_consecutive_misses=7)
    res = pair_sampled(d8, A, B, p, tmp_path, rng=np.random.default_rng(0),
                       gap_expr="shim", tag="T", quantum_estimator=_q_const(1, 1))
    assert res["verdict"] == "exhausted"
    assert res["n_pairs"] == 1                          # ran exactly once


def test_sweep_seed_reproducibility(d8, tmp_path):
    g = GroupSpec(8, 3)
    kw = dict(classical_estimator=_cl_const(None),
              quantum_estimator=_q_const(None, None))
    p = SweepParams(target=2, surv_cap=3, cl_sample_budget=50,
                    cl_barren_skip=50, max_pass=2, pair_budget=100,
                    pair_fail=10**9, seed=7)
    r1 = run_group_sweep(g, p, tmp_path / "r1", gd=d8, **kw)
    r2 = run_group_sweep(g, p, tmp_path / "r2", gd=d8, **kw)
    for key in ("verdict", "n_pairs", "n_A_survivors", "n_B_survivors",
                "A_d_classical", "B_d_classical", "n_gate_skip"):
        assert r1[key] == r2[key], key
    assert [Path(x).name for x in r1["passed"]] == \
           [Path(x).name for x in r2["passed"]]


def test_sweep_budget_verdict_saves_stats_on_zero_pass(d8, tmp_path):
    g = GroupSpec(8, 3)
    p = SweepParams(target=2, surv_cap=2, cl_sample_budget=20,
                    cl_barren_skip=20, max_pass=5, pair_budget=3,
                    pair_fail=10**9, seed=0)
    res = run_group_sweep(g, p, tmp_path, gd=d8,
                          classical_estimator=_cl_const(None),
                          quantum_estimator=_q_const(1, 1))
    assert res["verdict"] == "budget"
    assert res["n_pairs"] == 3 and res["passed"] == []
    stats = json.loads((tmp_path / "_sweep_Sg8_3.json").read_text())
    assert stats["n_passed"] == 0 and stats["verdict"] == "budget"


def test_sweep_streaming_budget_stop(d8, tmp_path):
    g = GroupSpec(8, 3)
    p = SweepParams(target=2, surv_cap=4, cl_sample_budget=50,
                    cl_barren_skip=50, max_pass=5, pair_budget=3,
                    pair_fail=10**9, seed=0)
    res = run_group_sweep_streaming(g, p, tmp_path, gd=d8,
                                    classical_estimator=_cl_const(None),
                                    quantum_estimator=_q_const(1, 1))
    assert res["verdict"] == "budget"
    assert res["n_pairs"] == 3 and res["passed"] == []
    assert (tmp_path / "_sweep_Sg8_3.json").exists()
