"""Four example decision circuits, run end to end through a scorer.

Each circuit is a normal /v1/systemone request plus a `gates` block.
The model answers the questions; code evaluates the gates and prints
the path each state took. Patterns from "AI Decision Circuits"
(Barney, 2025): redundancy (majority over paraphrases), a negative
checker (verify), threshold gating with explicit escalation, and
ordinal bucketing.

    uv run python scripts/circuits_demo.py fake
    uv run python scripts/circuits_demo.py lora:runs/1.7b-r16
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from s1proto.scorer import load_scorer
from s1proto.service import create_app

# --------------------------------------------------------------------------
# 1. Support triage: route with a confidence floor, tier by urgency, and
#    an OR gate that escalates on anger or a critical tier.
TRIAGE = {
    "questions": {
        "dept": {
            "type": "choice",
            "instructions": "Which team should handle this message?",
            "criteria": {
                "billing": "Charges, invoices, refunds, payment methods, subscription changes",
                "technical": "Bugs, errors, outages, integrations, API, performance",
                "account": "Login, password, permissions, user management, data export",
                "sales": "Pricing questions, upgrades, quotes, enterprise, trials",
                "other": "None of these teams fits, or it is not a support request",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": [
                "Low: cosmetic or a question; can wait a week",
                "Medium: a workaround exists; this week",
                "High: blocks important work; today",
                "Critical: outage, data loss, or security; now",
            ],
        },
        "angry": {
            "type": "noul",
            "instructions": "Is the customer hostile or threatening to leave?",
            "criteria": {"true": "Hostile tone, threats to cancel, escalate, or dispute", "false": "Calm, annoyed, or frustrated but not hostile"},
        },
    },
    "gates": {
        "route": {"op": "argmax", "input": "dept", "min_confidence": 0.35, "on_uncertain": "abstain"},
        "tier": {"op": "order", "input": "urgency", "cutpoints": [1.0, 2.0, 2.6], "band": 0.15, "on_uncertain": "default", "default": 2},
        "critical": {"op": "threshold", "input": "urgency:3", "tau": 0.5},
        "human": {"op": "or", "inputs": ["angry", "critical"], "tau": 0.6, "band": 0.1, "on_uncertain": "escalate"},
    },
    "states": [
        "Your app deleted every invoice from March when I clicked export. I have an audit Friday. Fix this now or we cancel.",
        "Hi! Quick one: does the Pro plan include SSO? Trying to decide before our renewal next month.",
        "Login keeps saying my password is wrong even after I reset it twice. Not urgent, I can use the mobile app for now.",
    ],
}

# --------------------------------------------------------------------------
# 2. PII redaction: AND(pii present, NOT business-only) with escalation in
#    the uncertain band. False-accept math is explicit in the trace.
REDACT = {
    "questions": {
        "pii": {
            "type": "noul",
            "instructions": "Does this text contain personally identifiable information about a private individual?",
            "criteria": {
                "true": "Email, phone, home address, government ID, card number, or a name combined with such details",
                "false": "No PII, or only public/business information",
            },
        },
        "business": {
            "type": "noul",
            "instructions": "Are all the identifying details in this text about a business or public organization rather than a private person?",
            "criteria": {
                "true": "Company names, business addresses, support lines, public figures only",
                "false": "At least one detail belongs to a private individual",
            },
        },
    },
    "gates": {
        "has_pii": {"op": "threshold", "input": "pii", "tau": 0.7, "band": 0.1, "on_uncertain": "escalate"},
        "private": {"op": "not", "input": "business", "tau": 0.7, "band": 0.1, "on_uncertain": "escalate"},
        "redact": {"op": "and", "inputs": ["has_pii", "private"], "tau": 0.6, "band": 0.1, "on_uncertain": "escalate"},
    },
    "states": [
        "Order 48213 shipped to Maria Delgado, 14 Ash Grove Ln, Tulsa OK 74105, tel 918-555-0142.",
        "Contact Acme Support at 1-800-555-0199 or support@acme.example, 200 Market St, San Francisco.",
        "Reset requested for user j.patel — see ticket #9921 for details.",
    ],
}

# --------------------------------------------------------------------------
# 3. Refund decision with redundancy: majority over three phrasings of
#    the eligibility question, plus a negative checker that asks whether
#    the policy explicitly excludes the reason. Decide only when the vote
#    is unanimous-ish and the checker passes; otherwise escalate.
#    Date arithmetic is done in code and handed to the model as
#    `days_since_purchase`: the model is not asked to subtract dates.
REFUND = {
    "questions": {
        "elig_a": {
            "type": "choice",
            "instructions": "Under `policy`, is this refund request eligible?",
            "criteria": {"eligible": "Every condition in the policy is met", "not_eligible": "At least one condition is not met"},
        },
        "elig_b": {
            "type": "choice",
            "instructions": "Would a careful support agent approve this refund strictly following `policy`?",
            "criteria": {"eligible": "Yes, approve", "not_eligible": "No, decline"},
        },
        "elig_c": {
            "type": "choice",
            "instructions": "Check `days_since_purchase` against the policy window, and `reason`/`item_condition` against the policy's exclusions. Does the request comply?",
            "criteria": {"eligible": "Complies on every point", "not_eligible": "Fails at least one point"},
        },
        "excluded": {
            "type": "noul",
            "instructions": "Does `policy` explicitly exclude the customer's stated `reason` or `item_condition`?",
            "criteria": {"true": "The policy names this reason or condition as not refundable", "false": "The policy does not exclude it"},
        },
    },
    "gates": {
        "vote": {"op": "majority", "inputs": ["elig_a", "elig_b", "elig_c"], "min_confidence": 0.6, "on_uncertain": "escalate"},
        "not_excluded": {"op": "not", "input": "excluded", "tau": 0.6, "band": 0.1, "on_uncertain": "escalate"},
        "in_window": {"op": "threshold", "input": "within_window", "tau": 0.5},
        "approve": {"op": "and", "inputs": ["vote:eligible", "not_excluded", "in_window"], "tau": 0.6, "band": 0.1, "on_uncertain": "escalate"},
    },
    "states": [
        {
            "policy": "Full refund within 30 days of purchase for unopened items. Opened software and gift cards are not refundable.",
            "purchase_date": "2026-08-30",
            "request_date": "2026-09-12",
            "reason": "Ordered the wrong edition",
            "item_condition": "unopened, shrink-wrap intact",
        },
        {
            "policy": "Full refund within 30 days of purchase for unopened items. Opened software and gift cards are not refundable.",
            "purchase_date": "2026-07-01",
            "request_date": "2026-09-12",
            "reason": "Never used it",
            "item_condition": "unopened",
        },
        {
            "policy": "Full refund within 30 days of purchase for unopened items. Opened software and gift cards are not refundable.",
            "purchase_date": "2026-09-05",
            "request_date": "2026-09-12",
            "reason": "Didn't like it",
            "item_condition": "opened, license key activated",
        },
    ],
}

# --------------------------------------------------------------------------
# 4. Moderation with a negative checker: choose an outcome, then ask a
#    separate question whether the post is plainly within the rules; a
#    removal that the checker contradicts goes to a human.
MODERATE = {
    "questions": {
        "outcome": {
            "type": "choice",
            "instructions": "Which moderation outcome applies to this post?",
            "criteria": {
                "allow": "Ordinary post within the rules",
                "remove_spam": "Unsolicited promotion, link farming, or repetitive content",
                "remove_harassment": "Insults, threats, or targeting of a person or group",
                "off_topic": "On-rules but belongs in a different forum",
                "needs_human": "Genuinely ambiguous; a moderator should decide",
            },
        },
        "clean": {
            "type": "noul",
            "instructions": "Is this post plainly within ordinary community rules (no promotion, no harassment, on topic)?",
            "criteria": {"true": "A reasonable moderator would leave it up without a second look", "false": "Something about it would give a moderator pause"},
        },
    },
    "gates": {
        "decision": {"op": "argmax", "input": "outcome", "min_confidence": 0.4, "on_uncertain": "escalate"},
        "not_clean": {"op": "not", "input": "clean", "tau": 0.6, "band": 0.15, "on_uncertain": "escalate"},
        # the negative checker: a removal must be corroborated by the
        # independent "is it clean?" question; an allow must be too.
        "allow_ok": {"op": "and", "inputs": ["decision:allow", "clean"], "tau": 0.5, "band": 0.1, "on_uncertain": "escalate"},
        "is_removal": {"op": "not", "input": "decision:allow", "tau": 0.5, "band": 0.1},
        "remove_ok": {"op": "and", "inputs": ["is_removal", "not_clean"], "tau": 0.5, "band": 0.1, "on_uncertain": "escalate"},
    },
    "states": [
        "Anyone else's dishwasher tablets leave a white film? Switched brands twice, same thing. Hard water maybe?",
        "You people are morons, this whole thread is proof. Go back to whatever hole you crawled out of, Dave.",
        "Tired of white film on dishes?? Our tablets fix it GUARANTEED 👉 cleanbrite.example/promo use code FILM20",
    ],
}

CIRCUITS = {"support_triage": TRIAGE, "pii_redaction": REDACT, "refund_decision": REFUND, "moderation": MODERATE}


def code_facts(state):
    """Facts the circuit owns in code, not the model: date arithmetic here.
    Returned as extra answers (a noul the gates can read) and as fields
    added to the state so the model sees the number, not the dates."""
    if not isinstance(state, dict) or "purchase_date" not in state:
        return state, {}
    from datetime import date

    days = (date.fromisoformat(state["request_date"]) - date.fromisoformat(state["purchase_date"])).days
    window = 30  # parsed from the policy in a real system
    return {**state, "days_since_purchase": days}, {"within_window": {"type": "noul", "noul": 1.0 if days <= window else 0.0}}


def run(client: TestClient, name: str, circuit: dict) -> None:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    for state in circuit["states"]:
        state, facts = code_facts(state)
        req = {"state": state, "model": "demo", "questions": circuit["questions"]}
        if not facts:
            req["gates"] = circuit["gates"]
        r = client.post("/v1/systemone", json=req, headers={"Authorization": "Bearer x"})
        body = r.json()
        if r.status_code != 200:
            print("ERROR", body)
            continue
        if facts:
            from s1proto.circuits import Gate, evaluate_gates

            answers = {**body["answers"], **facts}
            body["gates"] = {k: v.model_dump() for k, v in evaluate_gates({k: Gate.model_validate(v) for k, v in circuit["gates"].items()}, answers).items()}
            for k, v in facts.items():
                body["answers"][k] = v
        s = state if isinstance(state, str) else json.dumps(state)
        print(f"\nSTATE: {s[:110]}{'…' if len(s) > 110 else ''}")
        for qid, a in body["answers"].items():
            if a["type"] == "noul":
                print(f"  {qid:<12} P(yes)={a['noul']:.2f}")
            elif a["type"] == "choice":
                top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:3]
                print(f"  {qid:<12} {a['choice']} conf={a['confidence']:.2f}  " + "  ".join(f"{k}={v:.2f}" for k, v in top))
            else:
                print(f"  {qid:<12} score={a['score']:.2f} conf={a['confidence']:.2f}")
        for gid, gr in body["gates"].items():
            tag = gr["outcome"].upper() if gr["outcome"] != "decided" else ""
            print(f"  => {gid:<12} {gr['value']!s:<12} {tag:<9} {' | '.join(gr['trace'])}")


def main() -> None:
    spec = sys.argv[1] if len(sys.argv) > 1 else "fake"
    scorer = load_scorer(spec)
    with TestClient(create_app(scorer=scorer)) as client:
        for name, circuit in CIRCUITS.items():
            run(client, name, circuit)


if __name__ == "__main__":
    main()
