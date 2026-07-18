"""Regenerate the detection-power figure (delegates to the calibration demo,
which owns the linear calibration + plotting)."""
import pathlib
import runpy
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
runpy.run_module("experiments.demo_gut_brain", run_name="__main__")
