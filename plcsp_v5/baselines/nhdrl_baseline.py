# -*- coding: utf-8 -*-
"""Thin command-line wrapper for the adapted NHDRL baseline.

Run from any directory::

    D:/anaconda/python.exe plcsp_v5/baselines/nhdrl_baseline.py --help

The implementation lives in ``plcsp_v5/baselines/nhdrl/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "plcsp_v5" / "baselines"
for _path in (str(BASELINES), str(ROOT), str(ROOT / "plcsp_v5")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nhdrl.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
