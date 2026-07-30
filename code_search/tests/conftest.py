"""Pytest configuration.

Adds the project root to sys.path so top-level folders (core/, search/, ...)
are importable as packages when running the tests from any directory.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
