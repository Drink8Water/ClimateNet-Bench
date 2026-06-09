"""Backward-compatible wrapper for ERA5-Land downloading.

Prefer the package entry point:
    python scripts/download_era5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_era5 import main


if __name__ == "__main__":
    main()
