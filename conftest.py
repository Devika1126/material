"""Pytest configuration helpers for this repository.

This file ensures the repository root is on sys.path so tests can import the `src` package
when tests are executed from nested test folders (e.g., materials-gnn/tests).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
