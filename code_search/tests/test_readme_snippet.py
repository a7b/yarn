"""Guard: the top-level README's library snippet must build the code it
claims. The fresh-eye review caught an earlier version whose B had the wrong
shape and silently produced a k = 0 code (whose (None, None) estimate reads
as PASS) — this pins the corrected snippet's semantics.
"""

import pytest


@pytest.mark.gap
@pytest.mark.bposd
def test_readme_library_snippet_builds_a_30_6_code():
    from core.group import GroupData
    from core.quantum_code import build_quantum_code, compute_k
    from core.dist.quantum_bposd import estimate_quantum_distances_bposd

    gd = GroupData("SymmetricGroup(3)")
    A = [[(0, 1, 3), (0, 2, 4)]]
    B = [[(0, 1, 2), (0, 3, 5)]]
    qcode = build_quantum_code(A, B, gd)
    assert qcode["Hx"].shape[1] == 30
    k = compute_k(qcode["Hx"], qcode["Hz"])
    assert k == 6, "the README promises a [[30, 6]] code"
    dx, dz = estimate_quantum_distances_bposd(
        qcode["Hx"], qcode["Hz"], num_trials=2000, n_workers=1, osd_order=0)
    # logicals auto-derived; a genuine k>0 code must yield real distances
    assert isinstance(dx, int) and isinstance(dz, int)
    assert 1 <= dx <= 30 and 1 <= dz <= 30
