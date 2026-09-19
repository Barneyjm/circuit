"""Compatibility shim: gate evaluation now lives in `decision_circuits.gates`."""

from decision_circuits.gates import *
from decision_circuits.gates import Gate, GateResult, evaluate_gates

__all__ = ["Gate", "GateResult", "evaluate_gates"]
