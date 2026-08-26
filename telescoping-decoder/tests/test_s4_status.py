"""A licensing failure must not look like a hard shot.

S4 reports a shot it could not certify as an uncertified deferral. When the
cause is the Gurobi license rather than the shot, the status has to say so —
otherwise a missing license reads as "the IP gave up on a hard syndrome",
which would incorrectly classify a licensing failure as a decoding result.
"""
import pytest

from telescoping_decoder.s4_ip import _license_status


class _FakeGurobiError(Exception):
    """Matched by class *name*, like the real gurobipy.GurobiError."""


_FakeGurobiError.__name__ = "GurobiError"


@pytest.mark.parametrize("message,expected", [
    ("Model too large for size-limited license; visit https://gurobi.com/...",
     "LICENSE_TOO_SMALL"),
    ("License 2810304 has expired", "LICENSE_EXPIRED"),
    ("No license found", "LICENSE_MISSING"),
    ("Failed to obtain a valid license", "LICENSE_MISSING"),
])
def test_license_errors_are_classified(message, expected):
    assert _license_status(_FakeGurobiError(message)) == expected


def test_solver_errors_are_not_classified_as_licensing():
    assert _license_status(_FakeGurobiError("Unable to retrieve attribute")) is None


def test_non_gurobi_exceptions_are_never_license_errors():
    assert _license_status(ValueError("size-limited license")) is None
