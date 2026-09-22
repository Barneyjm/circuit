"""A run's fitted per-type temperatures are applied by the scorer and multiply a caller's."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_lora import fit_temperatures, pick_checkpoint

from s1proto.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from s1proto.template import render


def test_prompts_carry_their_kind():
    assert render("s", NoulQuestion(type="noul", instructions="?")).kind == "noul"
    assert render("s", ChoiceQuestion(type="choice", instructions="?", criteria={"a": None, "b": None})).kind == "choice"
    assert render("s", ScoreQuestion(type="score", instructions="?", criteria=["low", "high"])).kind == "score"


def test_fit_finds_the_temperature_that_undoes_overconfidence():
    # logits twice as sharp as the truth warrants: the fit should land near T=2
    raw, kinds = [], []
    for i in range(200):
        ref = [0.8, 0.2] if i % 2 else [0.3, 0.7]
        import math

        logits = [2 * math.log(r) for r in ref]
        raw.append((logits, ref))
        kinds.append("choice")
    assert fit_temperatures(raw, kinds)["choice"] == 2.0
    assert fit_temperatures(raw, kinds)["noul"] == 1.0  # too few noul rows: left alone


def test_checkpoint_rule_reads_the_worst_type():
    a = {"step": 1, "accuracy": 0.90, "ece": 0.02, "ece_worst": 0.30}  # scores badly calibrated, hidden in the mean
    b = {"step": 2, "accuracy": 0.90, "ece": 0.03, "ece_worst": 0.05}
    assert pick_checkpoint([a, b], 0.02)["step"] == 2
