"""Backward-compatibility launcher shim for verify_phase6a.py.
Delegates to verifications/verify_phase6a.py.
"""
import os
import runpy
import sys

_TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verifications", "verify_phase6a.py")

if __name__ == "__main__":
    runpy.run_path(_TARGET, run_name="__main__")
