"""Which checkpoint a run keeps: calibrated, but only once it is also accurate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_lora import pick_checkpoint


def h(step, acc, ece):
    return {"step": step, "accuracy": acc, "ece": ece}


def test_a_calibrated_model_that_knows_nothing_is_not_kept():
    # the router run: step 100 had the lowest ECE at 26% accuracy
    assert pick_checkpoint([h(100, 0.26, 0.010), h(600, 0.93, 0.048)], 0.02)["step"] == 600


def test_among_accurate_checkpoints_the_best_calibrated_wins():
    # circuit-8b-v2's validation curve: lowest ECE outright is step 1200, 3.4 points short
    curve = [h(1200, 0.8987, 0.0203), h(2000, 0.9330, 0.0374), h(2400, 0.9233, 0.0210), h(4000, 0.9284, 0.0291), h(4446, 0.9171, 0.0284)]
    assert pick_checkpoint(curve, 0.02)["step"] == 2400
    assert pick_checkpoint(curve, 1.0)["step"] == 1200  # the old rule, still available


def test_the_floor_is_wide_enough_not_to_chase_noise():
    # circuit-1.7b-v2: step 8400 is 1.6 points more accurate and twice as badly calibrated
    assert pick_checkpoint([h(4400, 0.9008, 0.0209), h(8000, 0.9125, 0.0422), h(8400, 0.9171, 0.0485)], 0.02)["step"] == 4400
