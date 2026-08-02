# Best HAL hardware layouts, one per code

Assembled from every HAL run under `hal_runs/` (`canonical/`, `superblock/`, `intra_experiments/`, 194 runs total). For each code, the run with the minimum `hardware_complexity` in its `benchmark.csv` was selected and its full run directory copied here verbatim.

`best_by_code.json` is the machine-readable index; each entry's `local_dir` names the folder in this directory, and `run_dir` records where it came from.

Re-check with `python verify_best_by_code.py` (rescans all runs, fails if any run beats the recorded winner).

## Index

| code | [[n,k,d]] | tiers | HC | avg coupler len | avg TSVs/edge | max avg face switches | folder |
|---|---|---|---|---|---|---|---|
| `paper_nonab_150_30_10` | [[150,30,10]] | 6 | **2.0211** | 6.491 | 3.583 | 4.119 | `20260526_10h18m01s_paper_nonab_150_30_10__canonical` |
| `paper_nonab_200_40_12` | [[200,40,12]] | 6 | **2.0890** | 8.384 | 3.766 | 4.121 | `20260701_20h03m45s_paper_nonab_200_40_12_reproduce_bbc` |
| `paper_nonab_300_60_14` | [[300,60,14]] | 8 | **2.3685** | 6.431 | 5.749 | 4.817 | `20260527_00h18m00s_paper_nonab_300_60_14__canonical` |
| `paper_nonab_420_84_16` | [[420,84,16]] | 9 | **2.6352** | 12.169 | 6.303 | 4.795 | `20260527_07h04m01s_paper_nonab_420_84_16__canonical` |
| `paper_nonab_500_100_16` | [[500,100,16]] | 10 | **2.8048** | 13.192 | 7.113 | 4.974 | `20260627_07h14m40s_paper_nonab_500_100_16__canonical` |
| `paper_nonab_540_108_18` | [[540,108,18]] | 10 | **2.7154** | 11.952 | 6.434 | 5.000 | `20260526_21h07m49s_paper_nonab_540_108_18__canonical` |
| `paper_nonab_600_120_20` | [[600,120,20]] | 10 | **2.7790** | 13.019 | 6.620 | 5.295 | `20260526_21h33m26s_paper_nonab_600_120_20__canonical` |
| `paper_nonab_630_126_20` | [[630,126,20]] | 14 | **3.3743** | 17.623 | 8.962 | 5.652 | `20260627_04h22m46s_paper_nonab_630_126_20__canonical` |
| `paper_nonab_750_150_22` | [[750,150,22]] | 12 | **3.0885** | 15.487 | 7.843 | 5.519 | `20260526_21h29m04s_paper_nonab_750_150_22__canonical` |
| `paper_nonab_780_156_22` | [[780,156,22]] | 14 | **3.4205** | 16.854 | 9.608 | 5.870 | `20260628_06h56m02s_paper_nonab_780_156_22__canonical` |
| `paper_nonab_900_180_24` | [[900,180,24]] | 12 | **3.2142** | 18.567 | 8.174 | 5.721 | `20260526_21h11m24s_paper_nonab_900_180_24__canonical` |
| `paper_nonab_975_195_24` | [[975,195,24]] | 17 | **3.9465** | 21.489 | 12.244 | 5.712 | `20260628_05h26m44s_paper_nonab_975_195_24__canonical` |

HC = `hardware_complexity` = sum of the four normalized metrics (tiers, avg coupler length,
max avg face switches, avg TSVs per edge), normalized against the `baseline_defaults` /
`bad_defaults` in each run's `settings.json`. Lower is better.

## Per-run contents

- `benchmark.csv` — the four metrics, their normalized values, and `hardware_complexity`
- `settings.json` — the exact HAL settings that produced the layout
- `perf.json` — placement/benchmark wall times
- `tiers/`, `layers/`, `grid_view/`, `tanner/` — per-tier metrics and layout data
- `tier_interactive_*.png` / `.svg` — rendered tier images

## Notes

- **`paper_nonab_200_40_12`** is the one code whose winner is not a plain canonical run. Its
  placement was recovered from `tiers/metrics_0.csv` of canonical run
  `20260527_00h56m18s` and re-run through current HAL, which routes better than it did on
  2026-05-27: 2.115002 -> 2.089023. The recovered placement is
  `paper_nonab_200_40_12_custom_pos.npz`; the reproduction script is
  `scripts/hal/reproduce_best_by_code_200.py`. The original placement-source run is kept here as
  `20260527_00h56m18s_paper_nonab_200_40_12__canonical__placement_source` and is *not* the winner.
- `superblock/` and `intra_experiments/` runs were scanned but none of them won for any code.
- `best_by_code.tex`, `placement_section.tex`, `spectral_ending.tex` are the paper text built off
  this table.

