"""`src` package for document dataset generator.

This file makes `src` an explicit Python package so scripts run
directly (e.g., `python scripts/generate_dataset.py`) can import
from `src` reliably.
"""

__all__ = ["data_generator"]
