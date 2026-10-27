"""Test path setup for the v5 package.

The training entry point runs from ``plcsp_v5/`` and uses absolute imports
such as ``from instance...`` and ``from trainer...``.  Keep the tests on the
same import path without requiring an editable install.
"""
from __future__ import annotations

import sys
from pathlib import Path

V5_DIR = Path(__file__).resolve().parents[1]
if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))
