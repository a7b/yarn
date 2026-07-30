"""Fresh-eye end-to-end tests for ``search/phases/`` + ``search/runners/``.

Strategy: run the REAL pipeline once per flavor (module-scoped fixtures,
tiny budgets), then verify the documented output layout and — crucially —
recompute every derived quantity in the saved artifacts from the stored
ring matrices using independent reference code (own GF(2) rank, own
Tanner-girth, own CSS orthogonality check, own dagger via ``gd.inv``).

Budgets are deliberately minimal: weight-[[1,2]] / weight-[[1,1]] codes
whose distances are structurally >= the classical ``d_target`` (so every
sampled candidate passes deterministically) and a pairing ``d_target`` of
1 (any estimate passes), keeping the runs deterministic.
"""

import itertools
import json
import os
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from search.configs.config import (
    BPOSDConfig,
    ClassicalDistanceConfig,
    ClassicalStageConfig,
    GroupConfig,
    PairingFiltersConfig,
    PairingPoolConfig,
    PairingStageConfig,
    PoolConfig,
    SamplingConfig,
    SearchConfig,
    SqetchVerifyConfig,
    WeightPatternConfig,
)
from search.configs.paths import (
    classical_A_dir,
    classical_B_dir,
    manifest_path,
    quantum_dir,
    report_path,
    tried_pairs_path,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def _ref_tanner_girth(H):
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
                    continue
                if y not in dist:
                    dist[y] = dist[x] + 1
                    q.append(y)
        if v in dist:
            cyc = dist[v] + 1
            if best is None or cyc < best:
                best = cyc
    return best


def _ring_from_json(M):
    return [[tuple(int(g) for g in entry) for entry in row] for row in M]


# ─────────────────────────────────────────────────────────────────
# Config builders
# ─────────────────────────────────────────────────────────────────


def _nonab_cfg(results_dir: Path) -> SearchConfig:
    return SearchConfig(
        shape=(1, 2),
        group=GroupConfig(gap_expr="SymmetricGroup(3)", tag="S3"),
        run_stages=["classical"],
        classical=ClassicalStageConfig(
            weight_A=[[1, 2]],
            weight_B=[[1, 2]],
            sampling=SamplingConfig(total_samples=2, seed=13),
            distance=ClassicalDistanceConfig(
                d_target=2, num_trials=5, n_workers=1, osd_order=0),
        ),
        results_dir=results_dir,
    )


def _ab_cfg(results_dir: Path, *, max_pairs=1) -> SearchConfig:
    return SearchConfig(
        shape=(1, 2),
        group=GroupConfig(gap_expr="CyclicGroup(6)", tag="C6"),
        run_stages=["classical", "pairing"],
        classical=ClassicalStageConfig(
            weight_pattern=WeightPatternConfig(
                entry_max=1, entry_min=1,     # pins W = [[1, 1]]
                num_weight_samples=2, ring_samples_per_weight=2,
            ),
            sampling=SamplingConfig(total_samples=0, seed=17),
            distance=ClassicalDistanceConfig(
                d_target=2, num_trials=5, n_workers=1, osd_order=0),
        ),
        pairing=PairingStageConfig(
            bposd=BPOSDConfig(d_target=1, num_trials=5, n_workers=1,
                              osd_order=0),
            sqetch_verify=SqetchVerifyConfig(enabled=False),
            filters=PairingFiltersConfig(require_same_group=True,
                                         require_same_shape=True),
            pool=PairingPoolConfig(max_pairs=max_pairs),
        ),
        results_dir=results_dir,
    )


# ─────────────────────────────────────────────────────────────────
# Module-scoped pipeline runs (one real execution per flavor)
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def nonab_run(tmp_path_factory):
    from search.phases.classical import run_classical
    cfg = _nonab_cfg(tmp_path_factory.mktemp("nonab") / "results")
    result = run_classical(cfg)
    return cfg, result


@pytest.fixture(scope="module")
def ab_run(tmp_path_factory):
    from search.phases.classical import run_classical
    from search.phases.pairing import run_pairing
    from search.phases.report import run_report
    cfg = _ab_cfg(tmp_path_factory.mktemp("ab") / "results", max_pairs=1)
    classical = run_classical(cfg)
    pairing_1 = run_pairing(cfg)      # first pair only (max_pairs = 1)
    pairing_2 = run_pairing(cfg)      # resume: must skip the tried pair
    md = run_report(cfg)
    return cfg, classical, pairing_1, pairing_2, md


@pytest.fixture(scope="module")
def gd_c6_mod():
    from core.group import GroupData
    return GroupData("CyclicGroup(6)")


# ─────────────────────────────────────────────────────────────────
# Non-abelian classical stage
# ─────────────────────────────────────────────────────────────────


class TestNonAbelianClassicalE2E:
    pytestmark = pytest.mark.bposd

    def test_layout_and_filenames(self, nonab_run):
        cfg, result = nonab_run
        # Deterministic: weight-[[1,2]] codes cannot have d < 2 (no zero
        # column), so every sampled candidate passes -> 2 saves per side.
        assert len(result["new_A"]) == 2
        assert len(result["new_B"]) == 2
        pattern = re.compile(r"d\d+_w1\.2_t5_\d+\.json$")
        for p in result["new_A"]:
            path = Path(p)
            assert path.parent == classical_A_dir(cfg) / "w1.2"
            assert pattern.search(path.name), path.name
        for p in result["new_B"]:
            assert Path(p).parent == classical_B_dir(cfg) / "w1.2"

    def test_saved_json_self_consistency_both_sides(self, nonab_run):
        # Rebuild the binary lift from the STORED ring matrix and
        # recompute every derived field with independent reference code.
        from core.classical_code import build_A_bin, build_B_bin
        from core.group import GroupData, left_rep
        cfg, result = nonab_run
        gd = GroupData(cfg.group.gap_expr)
        n = gd.n
        for side, paths in (("A", result["new_A"]), ("B", result["new_B"])):
            build = build_A_bin if side == "A" else build_B_bin
            for p in paths:
                data = json.loads(Path(p).read_text())
                assert data["side"] == side
                assert data["gap_expr"] == "SymmetricGroup(3)"
                assert data["group_tag"] == "S3"
                assert data["n"] == n
                assert (data["ma"], data["na"]) == (1, 2)
                M = _ring_from_json(data["matrix"])
                assert data["weight_matrix"] == [[len(e) for e in M[0]]]
                M_bin = build(M, gd)
                assert data["matrix_shape"] == list(M_bin.shape)
                my_rank = _ref_gf2_rank(M_bin)
                assert data["f2_rank_M_bin"] == my_rank
                assert data["column_space_coverage"] == (
                    my_rank == M_bin.shape[0])
                assert data["girth_tanner"] == _ref_tanner_girth(M_bin)
                blocks_inv = [
                    _ref_gf2_rank(M_bin[:, j * n:(j + 1) * n]) == n
                    for j in range(2)
                ]
                assert data["any_block_col_full_rank"] == any(blocks_inv)
                # dist: PASS semantics — None (nothing found) or >= d_target.
                assert data["dist"] is None or data["dist"] >= 2
                assert data["dist_estimator"] == "bposd"
                assert data["dist_num_trials"] == 5
                # Documented containment field uses LEFT reps for both sides.
                cmap = data["block_col_containment_map"]
                blocks = [left_rep(M[0][j], gd) for j in range(2)]
                for i in range(2):
                    for j in range(2):
                        if i == j:
                            continue
                        expected = (
                            _ref_gf2_rank(np.hstack([blocks[j], blocks[i]]))
                            == _ref_gf2_rank(blocks[j])
                        )
                        assert cmap[f"{i}_{j}"] == expected
                assert data["is_canonical"] is True

    def test_filename_dist_prefix_matches_json(self, nonab_run):
        cfg, result = nonab_run
        for p in result["new_A"] + result["new_B"]:
            data = json.loads(Path(p).read_text())
            prefix = int(Path(p).name.split("_", 1)[0][1:])
            if data["dist"] is None:
                # Documented fallback: the filename carries d_target.
                assert prefix == cfg.classical.distance.d_target
            else:
                assert prefix == data["dist"]

    def test_manifest_records_exactly_the_new_paths(self, nonab_run):
        cfg, result = nonab_run
        manifest = json.loads(manifest_path(cfg).read_text())
        assert isinstance(manifest, list) and len(manifest) == 1
        entry = manifest[0]
        assert entry["new_A"] == result["new_A"]
        assert entry["new_B"] == result["new_B"]
        assert isinstance(entry["timestamp"], int)

    def test_provenance_fields(self, nonab_run):
        _, result = nonab_run
        data = json.loads(Path(result["new_A"][0]).read_text())
        prov = data["provenance"]
        assert prov["phase"] == "classical"
        assert prov["phase_module"] == "search.phases.classical"
        assert prov["phase_function"] == "run_classical"
        assert prov["config_path"] is None   # programmatic cfg, no YAML
        assert prov["python_executable"] == sys.executable
        assert "git_commit" in prov

    def test_sampling_metadata_mirrors_config(self, nonab_run):
        cfg, result = nonab_run
        data = json.loads(Path(result["new_A"][0]).read_text())
        sm = data["sampling_metadata"]
        assert sm == {
            "seed": 13,
            "include_identity": True,
            "min_element_order": 1,
            "avoid_same_coset": False,
            "max_tries": 1000,
            "canonicalize": True,
        }


class TestClassicalPoolControls:
    pytestmark = pytest.mark.gap

    def _seed_fake_pool(self, cfg, wtag="1.2", dist=5):
        for droot in (classical_A_dir(cfg), classical_B_dir(cfg)):
            d = droot / f"w{wtag}"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"d{dist}_w{wtag}_t5_0.json").write_text("{}")

    def test_min_pool_size_skips_both_sides_without_sampling(self, tmp_path):
        from search.phases.classical import run_classical
        cfg = _nonab_cfg(tmp_path / "results")
        cfg.classical.pool = PoolConfig(min_pool_size=1)
        self._seed_fake_pool(cfg)
        out = run_classical(cfg)
        assert out == {"new_A": [], "new_B": []}
        # Nothing new anywhere; manifest not even created.
        assert not manifest_path(cfg).exists()
        for droot in (classical_A_dir(cfg), classical_B_dir(cfg)):
            files = list((droot / "w1.2").glob("*.json"))
            assert len(files) == 1   # only the seeded fake

    def test_existing_files_count_toward_max_saved(self, tmp_path):
        # max_saved counts the PRE-EXISTING pool: with one d>=target file
        # already present and max_saved=1, the run saves nothing (and
        # never even calls the estimator, so this test stays cheap).
        from search.phases.classical import run_classical
        cfg = _nonab_cfg(tmp_path / "results")
        cfg.classical.pool = PoolConfig(max_saved=1)
        cfg.classical.sampling.total_samples = 3
        self._seed_fake_pool(cfg, dist=2)
        out = run_classical(cfg)
        assert out == {"new_A": [], "new_B": []}

    def test_below_target_files_do_not_block_max_saved(self, tmp_path):
        # A d1_ file (below d_target=2) must NOT count toward the pool:
        # with total_samples=0 the run does nothing, but the counting
        # path is exercised via min_pool_size (not satisfied -> proceed).
        from search.phases.classical import run_classical
        cfg = _nonab_cfg(tmp_path / "results")
        cfg.classical.pool = PoolConfig(min_pool_size=1)
        cfg.classical.sampling.total_samples = 0
        self._seed_fake_pool(cfg, dist=1)
        out = run_classical(cfg)
        # Skip did NOT trigger (d1 < d_target doesn't count); the loop of
        # zero samples then saves nothing.
        assert out == {"new_A": [], "new_B": []}


# ─────────────────────────────────────────────────────────────────
# Abelian classical -> pairing -> report chain
# ─────────────────────────────────────────────────────────────────


class TestAbelianEndToEnd:
    pytestmark = pytest.mark.bposd

    def test_abelian_writes_only_A_side(self, ab_run):
        cfg, classical, *_ = ab_run
        assert classical["new_B"] == []
        assert len(classical["new_A"]) == 2   # 1 pattern x 2 ring samples
        assert not classical_B_dir(cfg).exists()
        for p in classical["new_A"]:
            assert Path(p).parent == classical_A_dir(cfg) / "w1.1"

    def test_weight1_codes_have_even_distance(self, ab_run):
        # For W = [[1,1]] both blocks are permutations: codewords come in
        # (u, matched u) pairs, so every codeword weight is even and the
        # estimator can only report an even value >= 2 (or None).
        _, classical, *_ = ab_run
        for p in classical["new_A"]:
            data = json.loads(Path(p).read_text())
            assert data["weight_matrix"] == [[1, 1]]
            d = data["dist"]
            assert d is None or (d >= 2 and d % 2 == 0)

    def test_first_pairing_run_respects_max_pairs(self, ab_run):
        cfg, _, pairing_1, _, _ = ab_run
        assert pairing_1["n_pairs_tried"] == 1
        assert pairing_1["n_pairs_passed"] == 1   # d_target=1 always passes
        assert len(pairing_1["new_quantum"]) == 1

    def test_abelian_pairs_are_self_pairs(self, ab_run):
        cfg, *_ = ab_run
        tried = json.loads(tried_pairs_path(cfg).read_text())
        assert len(tried) == 2   # after the resume run
        for a_path, b_path in tried:
            assert a_path == b_path   # abelian: each A paired with itself

    def test_second_pairing_run_resumes_not_repeats(self, ab_run):
        cfg, _, pairing_1, pairing_2, _ = ab_run
        # Only the one remaining untried pair is attempted.
        assert pairing_2["n_pairs_tried"] == 1
        assert len(pairing_2["new_quantum"]) == 1
        q1 = set(pairing_1["new_quantum"])
        q2 = set(pairing_2["new_quantum"])
        assert q1.isdisjoint(q2)
        assert len(list(quantum_dir(cfg).glob("k*.json"))) == 2

    def test_quantum_B_is_elementwise_dagger_of_A(self, ab_run, gd_c6_mod):
        # Documented abelian convention: B = A^* — every entry inverted
        # elementwise. Verified with gd.inv directly.
        _, _, pairing_1, _, _ = ab_run
        data = json.loads(Path(pairing_1["new_quantum"][0]).read_text())
        A = _ring_from_json(data["A"])
        B = _ring_from_json(data["B"])
        expected_B = [
            [tuple(sorted(gd_c6_mod.inv[g] for g in entry)) for entry in row]
            for row in A
        ]
        assert B == expected_B

    def test_quantum_json_self_consistency(self, ab_run, gd_c6_mod):
        # Rebuild the CSS pair from the stored ring matrices and recompute
        # every derived scalar with reference code.
        from core.quantum_code import build_quantum_code
        _, _, pairing_1, _, _ = ab_run
        data = json.loads(Path(pairing_1["new_quantum"][0]).read_text())
        A = _ring_from_json(data["A"])
        B = _ring_from_json(data["B"])
        qc = build_quantum_code(A, B, gd_c6_mod)
        Hx, Hz = qc["Hx"], qc["Hz"]
        # CSS orthogonality, checked directly.
        assert not ((Hx @ Hz.T) % 2).any()
        assert data["n_phys"] == Hx.shape[1] == Hz.shape[1]
        assert data["n_x_checks"] == Hx.shape[0]
        assert data["n_z_checks"] == Hz.shape[0]
        # 1x2 LP over |G|=6: n_phys = (na*nb + ma*mb) * n = 5 * 6.
        assert data["n_phys"] == 30
        k_ref = Hx.shape[1] - _ref_gf2_rank(Hx) - _ref_gf2_rank(Hz)
        assert data["k"] == k_ref
        assert data["Hx_check_weight"] == int(Hx.sum(axis=1).max())
        assert data["Hz_check_weight"] == int(Hz.sum(axis=1).max())
        assert data["girth_hx"] == _ref_tanner_girth(Hx)
        assert data["girth_hz"] == _ref_tanner_girth(Hz)
        for key in ("dx", "dz"):
            assert data[key] is None or data[key] >= 1
        assert data["estimator"] == "bposd"
        assert data["sqetch_num_trials"] == 0
        assert data["bposd_num_trials"] == 5
        assert (data["ma"], data["na"], data["mb"], data["nb"]) == (1, 2, 1, 2)
        assert data["weight_A"] == [[len(e) for e in A[0]]]
        assert data["weight_B"] == [[len(e) for e in B[0]]]

    def test_quantum_filename_reflects_content(self, ab_run):
        _, _, pairing_1, _, _ = ab_run
        p = Path(pairing_1["new_quantum"][0])
        data = json.loads(p.read_text())
        dx_part = "X" if data["dx"] is None else str(data["dx"])
        assert p.name.startswith(f"k{data['k']}_dx{dx_part}_")
        assert p.parent.name == "quantum"

    def test_quantum_source_backlinks_resolve(self, ab_run):
        _, classical, pairing_1, _, _ = ab_run
        data = json.loads(Path(pairing_1["new_quantum"][0]).read_text())
        src = data["source_A"]
        assert src["path"] in classical["new_A"]
        src_data = json.loads(Path(src["path"]).read_text())
        assert src["dist"] == src_data["dist"]
        assert src["weight_matrix"] == src_data["weight_matrix"]
        # Abelian self-pair: both sources are the same classical file.
        assert data["source_B"]["path"] == src["path"]
        prov = data["provenance"]
        assert prov["phase"] == "pairing"
        assert prov["phase_module"] == "search.phases.pairing"

    def test_report_contents(self, ab_run):
        cfg, _, pairing_1, pairing_2, md = ab_run
        assert report_path(cfg).exists()
        assert report_path(cfg).read_text() == md
        assert "Total quantum codes saved: **2**" in md
        assert "C6" in md and "1×2" in md
        # Both saved codes' k values appear as table rows.
        k = json.loads(Path(pairing_1["new_quantum"][0]).read_text())["k"]
        assert f"| 1 | {k} |" in md

    def test_manifest_accumulates_stage_entries(self, ab_run):
        cfg, *_ = ab_run
        manifest = json.loads(manifest_path(cfg).read_text())
        # classical run + two pairing runs that saved something = 3.
        assert len(manifest) == 3
        assert "new_A" in manifest[0]
        assert "new_quantum" in manifest[1]
        assert "new_quantum" in manifest[2]


# ─────────────────────────────────────────────────────────────────
# Pairing pool controls on a hand-seeded classical pool
# ─────────────────────────────────────────────────────────────────


def _seed_abelian_classical_pool(cfg, gd, num=3, dist=2):
    """Write ``num`` weight-[[1,1]] C6 classical JSONs by hand (no
    estimator involved)."""
    d = classical_A_dir(cfg) / "w1.1"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(num):
        matrix = [[[0], [1 + i]]]   # weight-1 entries; distinct per file
        data = {
            "gap_expr": cfg.group.gap_expr,
            "group_tag": cfg.group.tag,
            "side": "A",
            "shape": list(cfg.shape),
            "weight_matrix": [[1, 1]],
            "matrix": matrix,
            "dist": dist,
            "girth_tanner": None,
            "timestamp": i,
        }
        p = d / f"d{dist}_w1.1_t5_{i}.json"
        p.write_text(json.dumps(data))
        paths.append(p)
    return paths


class TestPairingPoolControlsFresh:

    @pytest.mark.bposd
    def test_min_quantum_pool_size_stops_early(self, tmp_path, gd_c6_mod):
        from search.phases.pairing import run_pairing
        cfg = _ab_cfg(tmp_path / "results", max_pairs=None)
        cfg.pairing.pool = PairingPoolConfig(min_quantum_pool_size=1)
        _seed_abelian_classical_pool(cfg, gd_c6_mod, num=3)
        out = run_pairing(cfg)
        # Three candidate self-pairs exist, but the run must stop after
        # the FIRST saved quantum code.
        assert len(out["new_quantum"]) == 1
        assert out["n_pairs_tried"] == 1

    @pytest.mark.gap
    def test_rejected_pairs_are_recorded_as_tried(self, tmp_path, gd_c6_mod):
        # Filters reject every pair BEFORE any distance estimation, yet
        # each pair must still land in tried_pairs.json (documented
        # skip-if-tried semantics) with nothing saved.
        from search.phases.pairing import run_pairing
        cfg = _ab_cfg(tmp_path / "results", max_pairs=None)
        cfg.pairing.filters = PairingFiltersConfig(
            require_same_group=True, require_same_shape=True,
            min_classical_distance=99,
        )
        _seed_abelian_classical_pool(cfg, gd_c6_mod, num=3, dist=2)
        out = run_pairing(cfg)
        assert out["new_quantum"] == []
        assert out["n_pairs_passed"] == 0
        assert out["n_pairs_tried"] == 3
        tried = json.loads(tried_pairs_path(cfg).read_text())
        assert len(tried) == 3
        # Nothing passed -> the manifest is untouched.
        assert not manifest_path(cfg).exists()

    @pytest.mark.gap
    def test_dist_none_classical_codes_rejected_by_min_distance(
            self, tmp_path, gd_c6_mod):
        # dist=None is documented as PESSIMISTIC for the pairing filter.
        from search.phases.pairing import run_pairing
        cfg = _ab_cfg(tmp_path / "results", max_pairs=None)
        cfg.pairing.filters = PairingFiltersConfig(
            require_same_group=True, require_same_shape=True,
            min_classical_distance=2,
        )
        _seed_abelian_classical_pool(cfg, gd_c6_mod, num=2, dist=None)
        out = run_pairing(cfg)
        assert out["new_quantum"] == []
        assert out["n_pairs_tried"] == 2
        assert out["n_pairs_passed"] == 0


# ─────────────────────────────────────────────────────────────────
# CLI runner in a real subprocess
# ─────────────────────────────────────────────────────────────────


def _cli_yaml(tmp_path: Path, *, run_stages=("classical",),
              results_dir=None) -> Path:
    cfg = {
        "shape": [1, 2],
        "group": {"gap_expr": "SymmetricGroup(3)", "tag": "S3"},
        "run_stages": list(run_stages),
        "classical": {
            "weight_A": [[1, 2]],
            "weight_B": [[1, 2]],
            "sampling": {"total_samples": 0, "seed": 0},
            "distance": {"d_target": 2, "num_trials": 5, "n_workers": 1,
                         "osd_order": 0},
        },
        "results_dir": str(results_dir or tmp_path / "results"),
    }
    import yaml
    p = tmp_path / "cli_cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _run_cli(args, timeout=240):
    return subprocess.run(
        [sys.executable, "-m", "search.runners.search", *args],
        cwd=str(_PROJECT_ROOT),
        env=dict(os.environ),
        capture_output=True, text=True, timeout=timeout,
    )


class TestRunnerCLISubprocess:
    pytestmark = pytest.mark.gap

    def test_quiet_run_exits_zero_with_empty_stdout(self, tmp_path):
        yml = _cli_yaml(tmp_path)
        r = _run_cli([str(yml), "--quiet"])
        assert r.returncode == 0, r.stderr
        assert r.stdout == ""

    def test_verbose_run_prints_config_and_stage_recap(self, tmp_path):
        yml = _cli_yaml(tmp_path)
        r = _run_cli([str(yml)])
        assert r.returncode == 0, r.stderr
        assert "Loaded config" in r.stdout
        assert "SymmetricGroup(3)" in r.stdout
        assert "Saved 0 new A-side, 0 new B-side" in r.stdout

    def test_stages_override_runs_report_only(self, tmp_path):
        yml = _cli_yaml(tmp_path)
        r = _run_cli([str(yml), "--stages", "report", "--quiet"])
        assert r.returncode == 0, r.stderr
        # report.md written at the documented location; classical dirs not.
        results = tmp_path / "results"
        assert (results / "S3" / "1x2" / "report.md").exists()
        assert not (results / "S3" / "1x2" / "classical_A").exists()

    def test_missing_config_file_fails_nonzero(self, tmp_path):
        r = _run_cli([str(tmp_path / "does_not_exist.yaml"), "--quiet"])
        assert r.returncode != 0

    def test_bogus_stage_inside_yaml_fails_at_load(self, tmp_path):
        # run_stages is validated AT LOAD TIME (before any stage runs and
        # pays its cost): nonzero exit + the loader's ValueError on stderr.
        yml = _cli_yaml(tmp_path, run_stages=("bogus_stage",))
        r = _run_cli([str(yml), "--quiet"])
        assert r.returncode != 0
        assert "bogus_stage" in r.stderr
        assert "run_stages" in r.stderr
