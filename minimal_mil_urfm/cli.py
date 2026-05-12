"""Console-script entry points for `mmu-train` and `mmu-eval`.

These are thin wrappers around the scripts in ``scripts/``; we keep the
real logic in the scripts so they can also be run directly with python.
"""
from __future__ import annotations

import sys


def train_main():
    from scripts.train_on_public_bus_mil import main as _m
    sys.exit(_m())


def eval_main():
    # Reserved for a future scripts/eval_on_busi.py
    print("`mmu-eval` is not yet implemented; eval is included in the "
          "train_on_public_bus_mil.py script as a final step.")
    sys.exit(2)
