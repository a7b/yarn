"""Empirical per-detector X/Z typing for compatible stim circuits.

GARI needs every detector classified as X-type or Z-type. A detector error
model alone does not carry this, so it is derived from the circuit's
measurement structure. Direct ``MX``/``MRX`` records are X-type and ordinary
``M``/``MR`` records are Z-type. The traversal also recognizes the
Hadamard-wrapped ancilla measurements emitted by Stim's generated CSS memory
circuits: a reset ancilla touched by H gates before its M/MR is X-type. It
uses each detector's measurement-record targets and does not assume a detector
ordering.

:func:`circuit_to_dem` wraps the DEM-construction flags this typing assumes.
"""
from __future__ import annotations

import numpy as np
import stim

_X_MEAS = {"MX", "MRX"}
_Z_MEAS = {"M", "MZ", "MR", "MRZ"}
_ALL_MEAS = _X_MEAS | _Z_MEAS
_RESETS = {"R", "RX", "RY"}


def _instruction_measurement_count(ins) -> int:
    """Number of measurement-record bits produced by one flat instruction."""
    if ins.name in _ALL_MEAS:
        # Every target of the supported single-qubit measurement gates emits
        # one result. Avoid reparsing these overwhelmingly common cases.
        return len(ins.targets_copy())
    if not stim.gate_data(ins.name).produces_measurements:
        return 0
    # Unsupported measurement instructions still occupy record positions.
    # Ask Stim for the exact count (MPP and pair-product measurements do not
    # have one-result-per-target layouts), so later rec offsets stay correct.
    return int(stim.Circuit(str(ins)).num_measurements)


def circuit_xz_detector_mask(circuit: stim.Circuit) -> np.ndarray:
    """Per-detector X/Z type mask (True = X-type) from measurement bases.

    Walks the flattened circuit tracking the running measurement count; each
    DETECTOR's measurement-record targets are mapped back to the measurement
    that produced them. In addition to explicit MX/MRX measurements, this
    recognizes M/MR ancillas touched by H since their most recent reset. That
    is the X-stabilizer measurement pattern used by Stim's generated surface-
    code circuits. Mixed detectors are an error (the GARI typing is undefined
    for them).
    """
    flat = circuit.flattened()
    meas_basis: list[str] = []
    types: list[bool] = []
    # CSS syndrome circuits reset each ancilla between rounds. Stim implements
    # X-stabilizer readout as R; H; entangling gates; H; MR, whereas Z
    # stabilizers have no H. Remember whether an H occurred during the current
    # reset-to-measurement interval; parity is intentionally irrelevant here.
    had_h_since_reset: set[int] = set()
    for ins in flat:
        targets = ins.targets_copy()
        if ins.name in _RESETS:
            for target in targets:
                had_h_since_reset.discard(target.value)
            continue
        if ins.name == "H":
            for target in targets:
                had_h_since_reset.add(target.value)
            continue

        n_meas = _instruction_measurement_count(ins)
        if n_meas:
            if ins.name in _X_MEAS:
                meas_basis.extend(["X"] * n_meas)
            elif ins.name in _Z_MEAS:
                # A single instruction can measure both X- and Z-stabilizer
                # ancillas, so record the basis target by target.
                for target in targets:
                    meas_basis.append(
                        "X" if target.value in had_h_since_reset else "Z")
            else:
                meas_basis.extend([f"unsupported:{ins.name}"] * n_meas)

            # M/MX ends the current interval; MR/MRX also performs the reset.
            if ins.name in _ALL_MEAS:
                for target in targets:
                    had_h_since_reset.discard(target.value)
        elif ins.name == "DETECTOR":
            bases = set()
            for t in targets:
                if t.is_measurement_record_target:
                    record_index = len(meas_basis) + t.value
                    if not 0 <= record_index < len(meas_basis):
                        raise ValueError(
                            f"detector {len(types)} has out-of-range "
                            f"measurement-record target rec[{t.value}]")
                    bases.add(meas_basis[record_index])
            if bases == {"X"}:
                types.append(True)
            elif bases == {"Z"}:
                types.append(False)
            elif any(b.startswith("unsupported:") for b in bases):
                gates = sorted(
                    b.split(":", 1)[1] for b in bases
                    if b.startswith("unsupported:"))
                raise ValueError(
                    f"detector {len(types)} references measurement results "
                    f"from unsupported gate(s) {gates}; GARI detector typing "
                    "supports direct MX/MRX measurements and CSS ancilla "
                    "measurements using M/MR with optional H wrapping")
            else:
                raise ValueError(
                    f"detector {len(types)} mixes measurement bases {bases}; "
                    "GARI X/Z typing is undefined")
    return np.array(types, dtype=bool)


def circuit_to_dem(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """Build the DEM with the flags the rest of this package assumes.

    Flatten loops, approximate disjoint errors, no error decomposition —
    keeps every detector (both X- and Z-type families).
    """
    return circuit.detector_error_model(
        flatten_loops=True, approximate_disjoint_errors=True,
    )


__all__ = ["circuit_xz_detector_mask", "circuit_to_dem"]
