#!/usr/bin/env python3
"""Forwarding shim — canonical source: wm4vla/scripts/precompute_libero_t5.py"""

import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from wm4vla.scripts.precompute_libero_t5 import main

if __name__ == "__main__":
    main()
