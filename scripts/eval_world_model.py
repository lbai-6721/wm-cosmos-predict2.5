#!/usr/bin/env python3
"""Forwarding shim — canonical source: wm4vla/scripts/eval_world_model.py"""

import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from wm4vla.scripts.eval_world_model import evaluate, parse_args

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
