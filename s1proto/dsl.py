"""Compatibility shim: the DSL now lives in the `decision-circuits` package
(../decision-circuits, published on PyPI). Everything is re-exported so
existing imports keep working."""

from decision_circuits.dsl import *
from decision_circuits.dsl import Circuit, G, Q, argmax, majority, order, render_mermaid, to_mermaid, verify

__all__ = ["Circuit", "G", "Q", "argmax", "majority", "order", "render_mermaid", "to_mermaid", "verify"]
