"""Re-scan every HAL run under hal_runs/ and check best_by_code.json still records the winner.

Run from anywhere:  python hal_runs/best_by_code/verify_best_by_code.py
Exits nonzero if any run beats a recorded winner, or if a recorded local_dir is missing.
"""

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HAL_RUNS = os.path.dirname(HERE)


def scan(root):
    """Yield (hardware_complexity, run_dir, benchmark_row) for every run under root."""
    for dirpath, _dirs, files in os.walk(root):
        if "benchmark.csv" not in files:
            continue
        # skip the copies inside best_by_code itself
        if os.path.abspath(dirpath).startswith(HERE):
            continue
        rows = list(csv.DictReader(open(os.path.join(dirpath, "benchmark.csv"))))
        if not rows or not rows[0].get("hardware_complexity"):
            continue
        yield float(rows[0]["hardware_complexity"]), os.path.relpath(dirpath, HAL_RUNS), rows[0]


def main():
    index = json.load(open(os.path.join(HERE, "best_by_code.json")))["best_by_code"]
    runs = list(scan(HAL_RUNS))
    print(f"scanned {len(runs)} runs under {HAL_RUNS}")

    bad = []
    for entry in index:
        code = entry["code_name"]
        recorded = entry["hardware_complexity"]

        local = os.path.join(HERE, entry["local_dir"])
        if not os.path.isdir(local):
            bad.append(f"{code}: local_dir missing: {entry['local_dir']}")
            continue
        local_rows = list(csv.DictReader(open(os.path.join(local, "benchmark.csv"))))
        local_hc = float(local_rows[0]["hardware_complexity"])
        if abs(local_hc - recorded) > 1e-9:
            bad.append(f"{code}: local_dir HC {local_hc:.6f} != recorded {recorded:.6f}")

        # runs for this code: dir name starts with a timestamp then the code name
        mine = [r for r in runs if code in r[1]]
        better = [r for r in mine if r[0] < recorded - 1e-9]
        flag = "OK "
        if better:
            flag = "BEAT"
            best_hc, best_dir, _ = min(better)
            bad.append(f"{code}: {best_dir} has HC {best_hc:.6f} < recorded {recorded:.6f}")
        print(f"  {flag} {code:24s} HC={recorded:.6f}  ({len(mine)} runs scanned)")

    if bad:
        print("\nFAIL:")
        for b in bad:
            print("  " + b)
        return 1
    print("\nall winners confirmed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
