"""Smoke tests — confirms the test runner is wired up."""

import pytest

pytestmark = pytest.mark.fast


def test_pytest_runs():
    assert True


def test_fast_marker_recognised():
    assert True


def test_sys_path_includes_project_root():
    """conftest.py should have put the project root on sys.path."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    assert str(project_root) in sys.path
