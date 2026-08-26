"""Memory experiment: sample shots, decode them, report the LER.

Samples detector outcomes from a DEM with stim, runs them through the
telescoping decoder, and prints the per-stage acceptance table and the
logical error rate with a 95% confidence interval.

    python examples/memory_experiment.py --shots 20000
    python examples/memory_experiment.py --shots 50000 --s2-batch 256   # 4 GB GPU
    python examples/memory_experiment.py --stem /path/to/my_code_stem

``--stem`` points at a prepared artifact set: ``<stem>.dem``,
``<stem>_matrices.npz`` and ``<stem>_gari_matrices.npz`` (what
``TelescopingDecoder.prepare()`` materializes). It defaults to the bundled
[[150,30,10]] example, r=10 at p=0.001.

Shots are decoded in chunks with global ``shot_ids``. This keeps S3 seeds
stable across process scheduling. S2 seeds its disordered gamma schedule from
the batch position, so changing ``--chunk`` or S2 batch settings can change
which S2 candidates are explored.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import stim

from telescoping_decoder import Stage, TelescopeConfig, TelescopingDecoder

DEFAULT_STEM = (Path(__file__).resolve().parent.parent / "tests" / "data"
                / "paper_nonab_150_30_10_xyz_p0.001_r10_xinit_L1_hookfree")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval — usable at k=0, unlike the normal one."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=int, default=20_000)
    ap.add_argument("--chunk", type=int, default=10_000,
                    help="shots per decode() call (progress + memory only)")
    ap.add_argument("--seed", type=int, default=1, help="stim sampler seed")
    ap.add_argument("--stem", type=Path, default=DEFAULT_STEM)
    ap.add_argument("--s1-batch", type=int, default=None,
                    help="s1.shots_per_batch (VRAM; default = config default)")
    ap.add_argument("--s2-batch", type=int, default=None,
                    help="s2.batch_size (VRAM; halve this first if you OOM)")
    ap.add_argument("--procs", type=int, default=None,
                    help="S3 worker processes (default: os.cpu_count())")
    ap.add_argument("--cpu-only", action="store_true",
                    help="skip the GPU stages (S1, S2)")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    stem = args.stem

    cfg = TelescopeConfig()
    if args.cpu_only:
        cfg.s1.enabled = cfg.s2.enabled = False
    if args.s1_batch:
        cfg.s1.shots_per_batch = args.s1_batch
    if args.s2_batch:
        cfg.s2.batch_size = cfg.s2.shots_per_batch = args.s2_batch
    if args.procs:
        cfg.s3.n_procs = args.procs

    print(f"sampling {args.shots} shots from {stem.name}.dem "
          f"(stim seed {args.seed})", flush=True)
    dem = stim.DetectorErrorModel.from_file(f"{stem}.dem")
    det, obs, _ = dem.compile_sampler(seed=args.seed).sample(
        shots=args.shots, bit_packed=False)
    syndromes = np.ascontiguousarray(det.astype(np.uint8))
    true_obs = np.ascontiguousarray(obs.astype(np.uint8))

    dec = TelescopingDecoder.from_npz(
        gari_npz=f"{stem}_gari_matrices.npz",
        matrices_npz=f"{stem}_matrices.npz",
        config=cfg,
    )

    stage = np.zeros(args.shots, dtype=np.int8)
    label = np.empty(args.shots, dtype=object)
    le = np.zeros(args.shots, dtype=bool)
    t_start = time.perf_counter()
    with dec:
        for lo in range(0, args.shots, args.chunk):
            hi = min(lo + args.chunk, args.shots)
            # Global shot IDs keep S3 seeds stable across decode calls.
            ids = np.arange(lo, hi, dtype=np.uint64)
            t0 = time.perf_counter()
            res = dec.decode(syndromes[lo:hi], true_obs=true_obs[lo:hi],
                             shot_ids=ids)
            stage[lo:hi], label[lo:hi], le[lo:hi] = res.stage, res.label, res.le
            print(f"  shots {lo}-{hi}: {time.perf_counter() - t0:6.1f}s  "
                  f"{res.counts_by_stage()}  LE {int(res.le.sum())}",
                  flush=True)
    wall = time.perf_counter() - t_start

    n = args.shots
    n_le = int(le.sum())
    lo_ci, hi_ci = wilson(n_le, n)
    print(f"\n{stem.name}  —  {n} shots")
    for code in (Stage.S1, Stage.S2, Stage.S3A, Stage.S3B, Stage.S3C,
                 Stage.S4, Stage.NC):
        count = int((stage == code).sum())
        if count:
            print(f"  {code.name:<4} accepted {count:>8}  ({count / n:9.5%})")
    print(f"  logical errors {n_le}   LER {n_le / n:.3e}   "
          f"[95% CI {lo_ci:.3e}, {hi_ci:.3e}]")

    # NC shots are scored against the zero prediction; report them separately.
    nc = stage == Stage.NC
    if nc.any():
        reasons: dict[str, int] = {}
        for text in label[nc]:
            reasons[str(text)] = reasons.get(str(text), 0) + 1
        print(f"  NC (uncertified) {int(nc.sum())}: {reasons}")
    print(f"  wall {wall:.1f}s ({n / wall:.1f} shots/s)")


if __name__ == "__main__":
    main()
