from __future__ import annotations

import site
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT_DIR / ".vendor"

if VENDOR_DIR.exists():
    site.addsitedir(str(VENDOR_DIR))
