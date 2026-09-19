"""The generalization grid: judgment *structure*, not subject matter.

The base model already knows about the world. The fine-tune teaches a
format: read a question over a state, put calibrated mass on options.
So the training distribution should span the structure of judgments:

    operation   what the question asks the model to do
    format      how the state is laid out
    options     what the answer space looks like

Every cell is (operation, format), each with its natural option shape,
and the labels are computed by code at generation time, so they are
exact, free of teacher noise, and free of dataset licenses. Holding out
whole cells (an operation, a format, or one cell) measures transfer
along each axis, which is the question "does it generalize" made
concrete.

Operations
    extract      does `text` mention a <phone / email / date / price / order id>?           noul
    classify     which category is this message?                                           choice (3 / 8 / 25 options, with or without descriptions, optional none-of-these)
    compare      is record A's <field> lower than record B's? / which of these is cheapest? noul / choice
    consistency  is `claim` consistent with `record`?                                       noul
    count        how many items in the list are <flagged>?                                  score (0, 1, 2, 3+)
    rule         does the record satisfy the stated rule (two or three clauses)?            noul
    temporal     did <event a> happen before <event b>? (mixed date formats)                noul
    negation     does the message decline / refuse / cancel?                                noul
    ordinal      how severe is the incident?                                                score (4 levels)

Formats
    string    prose with the fields written inline
    json      flat object
    nested    object with sub-objects
    list      a list of records (the question names one by id or position)
    thread    a chat transcript, the fact is in one turn
    document  the fields buried in unrelated filler paragraphs

Ambiguity: each operation emits ~10% items built to be undecidable (a
claim that is vague, a comparison of equal values with a strict
question, a date without a year) with a soft label of 0.5, so the model
learns that "I can't tell" is a valid distribution.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

OPERATIONS = ["extract", "classify", "compare", "consistency", "count", "rule", "temporal", "negation", "ordinal"]
FORMATS = ["string", "json", "nested", "list", "thread", "document"]

NAMES = ["Priya", "Marcus", "Elena", "Tomás", "Aisha", "Kenji", "Nora", "Dev", "Lucia", "Omar", "Hana", "Felix", "Ingrid", "Sam", "Yara"]
ITEMS = [
    "ceramic mug",
    "rain jacket",
    "wireless headphones",
    "desk lamp",
    "yoga mat",
    "blender",
    "phone case",
    "running shoes",
    "backpack",
    "coffee grinder",
    "throw blanket",
    "bike pump",
    "notebook set",
    "water bottle",
    "board game",
]
CITIES = ["Austin", "Lisbon", "Toronto", "Nairobi", "Osaka", "Denver", "Dublin", "Santiago", "Seoul", "Perth"]
FILLER = [
    "Thanks again for the quick turnaround last quarter; the team was impressed.",
    "As discussed, the office will be closed on the public holiday next week.",
    "Please note our support hours are 9 to 5 in your local time zone.",
    "The newsletter goes out on Thursdays and covers product updates and tips.",
    "We appreciate your patience while we migrate to the new billing system.",
    "Reminder: the survey closes at the end of the month and takes two minutes.",
    "Our warehouse team has expanded, so shipping estimates should improve.",
    "If you have feedback on the new dashboard, reply to this message.",
    "Weather permitting, deliveries in the region continue on the usual schedule.",
    "The knowledge base has a new article on setting up two-factor authentication.",
]
CATEGORIES = {
    "billing": (
        "Charges, invoices, refunds, payment methods",
        ["I was charged twice", "my invoice is wrong", "refund my last payment", "update my card on file", "why is there a fee"],
    ),
    "technical": (
        "Bugs, errors, outages, integrations",
        ["the app crashes on launch", "getting a 500 error", "the API returns nothing", "sync stopped working", "login page is blank"],
    ),
    "shipping": (
        "Delivery, tracking, lost or late packages",
        ["where is my package", "tracking hasn't updated", "delivered to the wrong address", "the box arrived damaged", "can I change the delivery date"],
    ),
    "account": ("Login, passwords, permissions, profile", ["reset my password", "add a teammate", "change my email", "I'm locked out", "delete my account"]),
    "sales": (
        "Pricing, upgrades, quotes, trials",
        ["how much is the pro plan", "extend my trial", "do you offer discounts", "send me a quote", "compare the plans"],
    ),
    "returns": (
        "Returning or exchanging an item",
        ["I want to return this", "exchange for a larger size", "return window question", "print a return label", "is this eligible for return"],
    ),
    "feedback": (
        "Suggestions, praise, complaints without a request",
        ["love the new design", "the checkout is confusing", "just wanted to say thanks", "your ads are annoying", "feature request: dark mode"],
    ),
    "legal": (
        "Privacy, terms, data requests, compliance",
        ["delete my personal data", "GDPR request", "where are your terms", "who do you share data with", "I need a data export"],
    ),
    "hr": (
        "Jobs, applications, employment",
        ["is the designer role open", "status of my application", "do you hire remote", "internship program", "benefits question"],
    ),
    "partnership": (
        "Reseller, affiliate, integration partnerships",
        ["become a reseller", "affiliate program", "co-marketing idea", "integrate with your platform", "referral terms"],
    ),
    "security": (
        "Suspicious activity, phishing, vulnerabilities",
        [
            "I got a phishing email",
            "someone logged in from another country",
            "reporting a vulnerability",
            "is this email really from you",
            "enable stricter security",
        ],
    ),
    "cancellation": (
        "Ending a subscription or order",
        ["cancel my subscription", "stop the renewal", "cancel order before it ships", "close my plan", "how do I unsubscribe"],
    ),
    "product_info": (
        "Specs, compatibility, availability",
        ["does it fit a 15 inch laptop", "is it dishwasher safe", "when is it back in stock", "what colors are available", "is it compatible with android"],
    ),
    "events": (
        "Webinars, meetups, conferences",
        ["register for the webinar", "is the meetup recorded", "conference ticket prices", "speaker application", "add the event to my calendar"],
    ),
    "press": (
        "Media, interviews, press kit",
        ["press kit request", "interview with your founder", "logo usage rights", "quote for an article", "media contact"],
    ),
    "abuse": (
        "Harassment, spam, policy violations by others",
        ["a user is harassing me", "reporting spam", "someone impersonates our brand", "fake reviews on the listing", "report a policy violation"],
    ),
    "accessibility": (
        "Screen readers, captions, accommodations",
        ["screen reader can't read the menu", "captions on videos", "keyboard navigation broken", "font is too small", "color contrast issue"],
    ),
    "localization": (
        "Languages, currencies, regions",
        ["is the app in Spanish", "charge me in euros", "ship to Brazil", "date format is wrong for my region", "translate the invoice"],
    ),
    "outage": (
        "Service down for everyone",
        ["is the service down", "status page says degraded", "nothing loads for our whole team", "outage since this morning", "when will it be back"],
    ),
    "training": (
        "Docs, tutorials, onboarding",
        ["is there a tutorial", "onboarding session", "where are the docs", "video walkthrough", "certification program"],
    ),
    "warranty": (
        "Repairs and warranty claims",
        ["is this under warranty", "repair request", "replacement part", "warranty period", "it broke after two months"],
    ),
    "gift": (
        "Gift cards, gift orders, wrapping",
        ["buy a gift card", "gift wrap option", "hide the price on the receipt", "gift card balance", "send to a different address as a gift"],
    ),
    "loyalty": ("Points, rewards, tiers", ["how many points do I have", "redeem rewards", "loyalty tier benefits", "points didn't post", "expiring rewards"]),
    "tax": (
        "Tax documents, VAT, exemptions",
        ["I need a VAT invoice", "tax exempt certificate", "sales tax charged twice", "year-end tax summary", "tax id on invoice"],
    ),
    "other": ("None of the above", ["what's the weather like", "tell me a joke", "wrong number", "hello?", "is anyone there"]),
}


@dataclass
class Item:
    cell: str
    op: str
    fmt: str
    kind: str
    state: Any
    question: dict[str, Any]
    ref: dict[str, float]
    ambiguous: bool = False


# ------------------------------------------------------------------ helpers
def _dstr(d: date, style: int) -> str:
    return [d.isoformat(), d.strftime("%B %-d, %Y"), d.strftime("%-m/%-d/%y"), d.strftime("%d %b %Y"), d.strftime("%A, %b %-d")][style]


def onehot(keys: list[str], k: str) -> dict[str, float]:
    return {x: (1.0 if x == k else 0.0) for x in keys}


def noul(p: float) -> dict[str, float]:
    return {"yes": p, "no": 1 - p}


def paraphrase(rng: random.Random, options: list[str]) -> str:
    return rng.choice(options)


# ------------------------------------------------------------- formatting
def render(
    rng: random.Random,
    fmt: str,
    fields: dict[str, Any],
    *,
    prose: Callable[[dict[str, Any]], str],
    nest: dict[str, list[str]] | None = None,
    others: list[dict[str, Any]] | None = None,
    focus_id: str | None = None,
) -> Any:
    """Lay the same fields out in one of the six formats."""
    if fmt == "string":
        return prose(fields)
    if fmt == "json":
        return dict(fields)
    if fmt == "nested":
        nest = nest or {"details": list(fields)[len(fields) // 2 :]}
        out: dict[str, Any] = {k: v for k, v in fields.items() if not any(k in ks for ks in nest.values())}
        for group, ks in nest.items():
            out[group] = {k: fields[k] for k in ks if k in fields}
        return out
    if fmt == "list":
        rows = [dict(fields, id=focus_id or "r1")] + [dict(o, id=f"r{i + 2}") for i, o in enumerate(others or [])]
        rng.shuffle(rows)
        return rows
    if fmt == "thread":
        fact = prose(fields)
        turns = [{"role": "agent", "text": "Hi, how can I help today?"}, {"role": "customer", "text": fact}]
        pre = rng.sample(FILLER, 2)
        turns = [{"role": "agent", "text": pre[0]}, {"role": "customer", "text": "Sure, one thing first."}, *turns, {"role": "agent", "text": pre[1]}]
        return turns
    if fmt == "document":
        paras = rng.sample(FILLER, 4)
        paras.insert(rng.randrange(1, 4), prose(fields))
        return "\n\n".join(paras)
    raise ValueError(fmt)


# ------------------------------------------------------------- operations
def gen_extract(rng: random.Random, fmt: str) -> Item:
    kinds = {
        "phone number": (lambda: f"{rng.randint(200, 989)}-555-{rng.randint(1000, 9999)}", "you can reach me at {v}"),
        "email address": (lambda: f"{rng.choice(NAMES).lower()}{rng.randint(1, 99)}@example.com", "my email is {v}"),
        "date": (lambda: _dstr(date(2026, rng.randint(1, 12), rng.randint(1, 28)), rng.randrange(5)), "it happened on {v}"),
        "price": (lambda: f"${rng.randint(5, 900)}.{rng.randint(0, 99):02d}", "the total was {v}"),
        "order id": (lambda: f"ORD-{rng.randint(10000, 99999)}", "the order number is {v}"),
    }
    target = rng.choice(list(kinds))
    present = rng.random() < 0.5
    ambiguous = rng.random() < 0.1
    decoy = rng.choice([k for k in kinds if k != target])
    parts = [rng.choice(FILLER)]
    if present:
        maker, tmpl = kinds[target]
        parts.append(tmpl.format(v=maker()))
    elif ambiguous:
        parts.append(
            {
                "phone number": "call me on my cell",
                "email address": "email me",
                "date": "it was sometime last spring",
                "price": "it cost a bit",
                "order id": "the order from last week",
            }[target]
        )
    else:
        maker, tmpl = kinds[decoy]
        parts.append(tmpl.format(v=maker()))
    rng.shuffle(parts)
    fields = {"name": rng.choice(NAMES), "text": " ".join(parts)}
    state = render(
        rng,
        fmt,
        fields,
        prose=lambda f: f"{f['name']} wrote: {f['text']}",
        others=[{"name": rng.choice(NAMES), "text": rng.choice(FILLER)} for _ in range(2)],
        focus_id="m1",
    )
    q = {
        "type": "noul",
        "instructions": paraphrase(
            rng, [f"Does the message contain a {target}?", f"Is a {target} given anywhere in the text?", f"Does the writer include a {target}?"]
        )
        + (" (the message with id m1)" if fmt == "list" else ""),
        "criteria": {"true": f"A specific {target} appears", "false": f"No {target}, or only a vague reference"},
    }
    return Item(f"extract/{fmt}", "extract", fmt, "noul", state, q, noul(0.5 if ambiguous else (1.0 if present else 0.0)), ambiguous)


def gen_classify(rng: random.Random, fmt: str) -> Item:
    n = rng.choice([3, 8, 24])  # 24 = every category but "other"
    with_desc = rng.random() < 0.6
    none_option = n < 24 and rng.random() < 0.3
    cats = rng.sample([c for c in CATEGORIES if c != "other"], n if not none_option else n - 1)
    if none_option:
        cats.append("other")
    truth = rng.choice(cats)
    if truth == "other" or (none_option and rng.random() < 0.15):
        truth_text = rng.choice(CATEGORIES["other"][1]) if truth == "other" else rng.choice(CATEGORIES[rng.choice([c for c in CATEGORIES if c not in cats])][1])
        truth = "other" if none_option else truth
    else:
        truth_text = rng.choice(CATEGORIES[truth][1])
    fields = {"from": rng.choice(NAMES), "message": truth_text[0].upper() + truth_text[1:] + rng.choice([".", "!", "?", ""])}
    state = render(
        rng,
        fmt,
        fields,
        prose=lambda f: f"Message from {f['from']}: {f['message']}",
        others=[{"from": rng.choice(NAMES), "message": rng.choice(FILLER)} for _ in range(2)],
        focus_id="m1",
    )
    crit = {c: (CATEGORIES[c][0] if with_desc else None) for c in cats}
    q = {
        "type": "choice",
        "instructions": paraphrase(rng, ["Which category does this message belong to?", "Classify the message.", "Pick the best category for the message."])
        + (" (the message with id m1)" if fmt == "list" else ""),
        "criteria": crit,
    }
    return Item(f"classify/{fmt}", "classify", fmt, "choice", state, q, onehot(cats, truth))


def gen_compare(rng: random.Random, fmt: str) -> Item:
    field_name = rng.choice(["price", "weight_kg", "delivery_days", "rating"])
    if fmt == "list" or rng.random() < 0.4:
        # which of these is lowest/highest
        k = rng.choice([3, 4, 5])
        vals = rng.sample(range(5, 500), k)
        ambiguous = rng.random() < 0.1
        if ambiguous:
            vals[1] = vals[0]
        rows = [{"item": rng.choice(ITEMS), field_name: v} for v in vals]
        want_low = rng.random() < 0.5
        target = min(vals) if want_low else max(vals)
        ids = [f"r{i + 1}" for i in range(k)]
        state = [dict(r, id=i) for r, i in zip(rows, ids, strict=True)]
        if fmt == "string":
            state = "; ".join(f"{r['id']}: {r['item']} with {field_name} {r[field_name]}" for r in state)
        elif fmt == "thread":
            state = [{"role": "customer", "text": "Here are the options."}] + [
                {"role": "customer", "text": f"{r['id']}: {r['item']}, {field_name} {r[field_name]}"} for r in state
            ]
        elif fmt == "document":
            state = "\n\n".join(
                [rng.choice(FILLER), "Options: " + "; ".join(f"{r['id']} ({r['item']}) {field_name}={r[field_name]}" for r in state), rng.choice(FILLER)]
            )
        elif fmt in ("json", "nested"):
            state = {"options": state} if fmt == "json" else {"request": {"kind": "compare"}, "catalog": {"options": state}}
        q = {"type": "choice", "instructions": f"Which option has the {'lowest' if want_low else 'highest'} {field_name}?", "criteria": {i: None for i in ids}}
        if ambiguous and target in (vals[0], vals[1]) and vals[0] == vals[1]:
            ref = {i: 0.0 for i in ids}
            ref[ids[0]] = ref[ids[1]] = 0.5
        else:
            ref = onehot(ids, ids[vals.index(target)])
        return Item(f"compare/{fmt}", "compare", fmt, "choice", state, q, ref, ambiguous)
    a, b = rng.sample(range(5, 500), 2)
    ambiguous = rng.random() < 0.1
    if ambiguous:
        b = a
    fields = {"a": {"item": rng.choice(ITEMS), field_name: a}, "b": {"item": rng.choice(ITEMS), field_name: b}}
    prose = lambda f: (
        f"Option A is a {f['a']['item']} with {field_name} {f['a'][field_name]}; option B is a {f['b']['item']} with {field_name} {f['b'][field_name]}."
    )
    state = render(rng, fmt, fields, prose=prose, nest={"a": ["a"], "b": ["b"]}) if fmt != "nested" else fields
    q = {
        "type": "noul",
        "instructions": paraphrase(
            rng,
            [
                f"Is option A's {field_name} strictly lower than option B's?",
                f"Does A have a lower {field_name} than B?",
                f"Comparing A and B, is A's {field_name} the smaller one (strictly)?",
            ],
        ),
        "criteria": {"true": "A is strictly lower", "false": "A is equal or higher"},
    }
    return Item(f"compare/{fmt}", "compare", fmt, "noul", state, q, noul(0.0 if ambiguous else (1.0 if a < b else 0.0)), False)


def _order(rng: random.Random) -> dict[str, Any]:
    d = date(2026, rng.randint(1, 9), rng.randint(1, 28))
    return {
        "item": rng.choice(ITEMS),
        "amount_usd": rng.randint(8, 400),
        "ordered_on": d.isoformat(),
        "delivered_on": (d + timedelta(days=rng.randint(2, 9))).isoformat(),
        "status": rng.choice(["delivered", "shipped", "processing"]),
        "city": rng.choice(CITIES),
        "prior_refunds": rng.randint(0, 3),
    }


def gen_consistency(rng: random.Random, fmt: str) -> Item:
    order = _order(rng)
    claims_true = [
        lambda o: f"I ordered a {o['item']}.",
        lambda o: f"It was delivered on {_dstr(date.fromisoformat(o['delivered_on']), rng.randrange(4))}.",
        lambda o: f"I paid ${o['amount_usd']} for it.",
        lambda o: f"It shipped to {o['city']}.",
        lambda o: f"The order status is {o['status']}.",
    ]
    claims_false = [
        lambda o: f"I ordered a {rng.choice([i for i in ITEMS if i != o['item']])}.",
        lambda o: f"It was delivered on {_dstr(date.fromisoformat(o['delivered_on']) + timedelta(days=rng.choice([-20, -9, 12, 30])), rng.randrange(4))}.",
        lambda o: f"I paid ${o['amount_usd'] + rng.choice([-60, -25, 40, 120])} for it.",
        lambda o: f"It shipped to {rng.choice([c for c in CITIES if c != o['city']])}.",
        lambda o: f"The order status is {rng.choice([s for s in ['delivered', 'shipped', 'processing', 'cancelled'] if s != o['status']])}.",
    ]
    claims_vague = ["It arrived recently.", "I paid a fair bit for it.", "It went to my usual address.", "I ordered one of the kitchen things."]
    ambiguous = rng.random() < 0.1
    consistent = rng.random() < 0.5
    claim = rng.choice(claims_vague) if ambiguous else (rng.choice(claims_true) if consistent else rng.choice(claims_false))(order)
    fields = {"claim": claim, "order": order}
    prose = lambda f: (
        f'Customer says: "{f["claim"]}" Our record: {f["order"]["item"]}, ${f["order"]["amount_usd"]}, ordered {f["order"]["ordered_on"]}, delivered {f["order"]["delivered_on"]} to {f["order"]["city"]}, status {f["order"]["status"]}.'
    )
    state = (
        fields
        if fmt in ("json", "nested")
        else render(
            rng,
            fmt,
            {"claim": claim, **order},
            prose=lambda f: prose({"claim": f["claim"], "order": f}),
            others=[dict(_order(rng), claim=rng.choice(claims_vague)) for _ in range(2)],
            focus_id="c1",
        )
    )
    q = {
        "type": "noul",
        "instructions": paraphrase(
            rng,
            [
                "Is the customer's claim consistent with the order record?",
                "Does the record support what the customer says?",
                "Check the claim against the record: does it hold?",
            ],
        )
        + (" (record c1)" if fmt == "list" else ""),
        "criteria": {"true": "The claim matches the record", "false": "The claim contradicts the record"},
    }
    return Item(f"consistency/{fmt}", "consistency", fmt, "noul", state, q, noul(0.5 if ambiguous else (1.0 if consistent else 0.0)), ambiguous)


def gen_count(rng: random.Random, fmt: str) -> Item:
    flag = rng.choice([("overdue", lambda: rng.random() < 0.4), ("flagged", lambda: rng.random() < 0.35), ("returned", lambda: rng.random() < 0.3)])
    n = rng.randint(3, 7)
    rows = [{"id": f"t{i + 1}", "item": rng.choice(ITEMS), flag[0]: flag[1]()} for i in range(n)]
    k = sum(1 for r in rows if r[flag[0]])
    levels = ["0", "1", "2", "3 or more"]
    truth = min(k, 3)
    if fmt in ("string", "document"):
        body = "; ".join(f"{r['id']} {r['item']} ({'is ' + flag[0] if r[flag[0]] else 'not ' + flag[0]})" for r in rows)
        state = body if fmt == "string" else "\n\n".join([rng.choice(FILLER), "Items: " + body, rng.choice(FILLER)])
    elif fmt == "thread":
        state = [{"role": "agent", "text": "Reading the list."}] + [
            {"role": "system", "text": f"{r['id']}: {r['item']}, {flag[0]}={str(r[flag[0]]).lower()}"} for r in rows
        ]
    elif fmt == "nested":
        state = {"batch": {"id": f"B{rng.randint(100, 999)}", "items": rows}}
    else:
        state = rows if fmt == "list" else {"items": rows}
    q = {
        "type": "score",
        "instructions": paraphrase(
            rng, [f"How many items are {flag[0]}?", f"Count the {flag[0]} items.", f"How many of the listed items are marked {flag[0]}?"]
        ),
        "criteria": levels,
    }
    return Item(f"count/{fmt}", "count", fmt, "score", state, q, onehot([str(i) for i in range(4)], str(truth)))


def gen_rule(rng: random.Random, fmt: str) -> Item:
    order = _order(rng)
    order["final_sale"] = rng.random() < 0.3
    order["days_since_delivery"] = rng.randint(1, 60)
    rules = [
        (
            "A refund is allowed if delivered within the last 30 days and the item is not final sale.",
            lambda o: o["days_since_delivery"] <= 30 and not o["final_sale"],
        ),
        ("Free reshipping applies if the status is delivered and the amount is under $100.", lambda o: o["status"] == "delivered" and o["amount_usd"] < 100),
        ("Escalate if there are 2 or more prior refunds or the amount is over $250.", lambda o: o["prior_refunds"] >= 2 or o["amount_usd"] > 250),
        (
            "A goodwill credit applies if the amount is at least $50, prior refunds are 0, and it is not final sale.",
            lambda o: o["amount_usd"] >= 50 and o["prior_refunds"] == 0 and not o["final_sale"],
        ),
    ]
    rule_text, fn = rng.choice(rules)
    fields = {"rule": rule_text, **order}
    prose = lambda f: (
        f"Rule: {f['rule']} Record: {f['item']}, ${f['amount_usd']}, status {f['status']}, delivered {f['days_since_delivery']} days ago, final sale {'yes' if f['final_sale'] else 'no'}, prior refunds {f['prior_refunds']}."
    )
    state = render(
        rng,
        fmt,
        fields,
        prose=prose,
        nest={"record": [k for k in fields if k != "rule"]},
        others=[dict(_order(rng), rule=rule_text, final_sale=False, days_since_delivery=rng.randint(1, 60)) for _ in range(2)],
        focus_id="o1",
    )
    q = {
        "type": "noul",
        "instructions": paraphrase(
            rng, ["Does the record satisfy the rule?", "Apply the rule to the record: does it hold?", "Given the rule, does this record qualify?"]
        )
        + (" (record o1)" if fmt == "list" else ""),
        "criteria": {"true": "Every condition of the rule is met", "false": "At least one condition fails"},
    }
    return Item(f"rule/{fmt}", "rule", fmt, "noul", state, q, noul(1.0 if fn(order) else 0.0))


def gen_temporal(rng: random.Random, fmt: str) -> Item:
    a = date(2026, rng.randint(1, 12), rng.randint(1, 28))
    b = a + timedelta(days=rng.choice([-90, -30, -7, -1, 1, 3, 14, 60]))
    ambiguous = rng.random() < 0.1
    ea, eb = rng.sample(["the payment", "the delivery", "the support call", "the refund request", "the account change", "the outage"], 2)
    sa, sb = _dstr(a, rng.randrange(5)), _dstr(b, rng.randrange(5))
    if ambiguous:
        sb = b.strftime("%B %-d")  # no year: undecidable across a year boundary only if it matters; keep simple: mark soft
    fields = {"event_a": ea, "when_a": sa, "event_b": eb, "when_b": sb}
    prose = lambda f: f"{f['event_a'].capitalize()} was on {f['when_a']}. {f['event_b'].capitalize()} was on {f['when_b']}."
    state = render(
        rng,
        fmt,
        fields,
        prose=prose,
        nest={"a": ["event_a", "when_a"], "b": ["event_b", "when_b"]},
        others=[{"event_a": "the survey", "when_a": _dstr(a, 0), "event_b": "the newsletter", "when_b": _dstr(b, 1)} for _ in range(2)],
        focus_id="e1",
    )
    q = {
        "type": "noul",
        "instructions": paraphrase(rng, [f"Did {ea} happen before {eb}?", f"Was {ea} earlier than {eb}?", f"In time order, does {ea} come first?"])
        + (" (entry e1)" if fmt == "list" else ""),
        "criteria": {"true": f"{ea} is strictly earlier", "false": f"{ea} is later or the same day"},
    }
    return Item(f"temporal/{fmt}", "temporal", fmt, "noul", state, q, noul(0.5 if ambiguous else (1.0 if a < b else 0.0)), ambiguous)


def gen_negation(rng: random.Random, fmt: str) -> Item:
    offer = rng.choice(["the upgrade", "the replacement", "the extended warranty", "the call back", "the discount"])
    accept = [f"Yes, let's go ahead with {offer}.", f"Sounds good, please proceed with {offer}.", f"I'll take {offer}, thanks.", f"Sign me up for {offer}."]
    decline = [
        f"No thanks, I don't want {offer}.",
        f"Please don't proceed with {offer}.",
        f"I'd rather not take {offer}.",
        f"Cancel {offer}, I've changed my mind.",
        f"Not {offer}, not now.",
    ]
    tricky_accept = [f"I can't say no to {offer}.", f"Don't skip {offer}, I want it.", f"It's not that I don't want {offer}; go ahead."]
    tricky_decline = [f"I'm not sure {offer} is for me; skip it.", f"Not interested in {offer}, sorry."]
    vague = [f"Let me think about {offer}.", f"What does {offer} involve?", f"Maybe later for {offer}."]
    ambiguous = rng.random() < 0.1
    r = rng.random()
    if ambiguous:
        text, p = rng.choice(vague), 0.5
    elif r < 0.35:
        text, p = rng.choice(accept), 0.0
    elif r < 0.55:
        text, p = rng.choice(tricky_accept), 0.0
    elif r < 0.85:
        text, p = rng.choice(decline), 1.0
    else:
        text, p = rng.choice(tricky_decline), 1.0
    fields = {"from": rng.choice(NAMES), "reply": text}
    state = render(
        rng,
        fmt,
        fields,
        prose=lambda f: f"{f['from']} replied: {f['reply']}",
        others=[{"from": rng.choice(NAMES), "reply": rng.choice(FILLER)} for _ in range(2)],
        focus_id="m1",
    )
    q = {
        "type": "noul",
        "instructions": paraphrase(rng, [f"Is the customer declining {offer}?", f"Does the reply turn down {offer}?", f"Is this a refusal of {offer}?"])
        + (" (message m1)" if fmt == "list" else ""),
        "criteria": {"true": "They decline or cancel it", "false": "They accept it, or ask for it"},
    }
    return Item(f"negation/{fmt}", "negation", fmt, "noul", state, q, noul(p), ambiguous)


def gen_ordinal(rng: random.Random, fmt: str) -> Item:
    tiers = [
        ("Cosmetic", ["a typo on the settings page", "the logo is slightly off-center", "a button label is lowercase", "the footer link color is wrong"]),
        (
            "Degraded",
            ["search results load slowly", "exports take twice as long as usual", "one report shows stale numbers", "email notifications are delayed"],
        ),
        (
            "Blocking",
            [
                "users can't complete checkout",
                "the login page errors for some accounts",
                "the API rejects all uploads",
                "invoices are generating with the wrong totals",
            ],
        ),
        ("Outage", ["the whole site is down", "no one can log in", "all API calls time out", "data is being lost on save"]),
    ]
    t = rng.randrange(4)
    fields = {
        "reporter": rng.choice(NAMES),
        "report": rng.choice(tiers[t][1]) + rng.choice([".", " since this morning.", ", affecting several customers.", ""]),
    }
    state = render(
        rng,
        fmt,
        fields,
        prose=lambda f: f"Incident reported by {f['reporter']}: {f['report']}",
        others=[{"reporter": rng.choice(NAMES), "report": rng.choice(FILLER)} for _ in range(2)],
        focus_id="i1",
    )
    q = {
        "type": "score",
        "instructions": paraphrase(rng, ["How severe is this incident?", "Rate the severity of the report.", "What severity level is this?"])
        + (" (incident i1)" if fmt == "list" else ""),
        "criteria": [f"{name}: {', '.join(ex[:2])}" for name, ex in tiers],
    }
    return Item(f"ordinal/{fmt}", "ordinal", fmt, "score", state, q, onehot([str(i) for i in range(4)], str(t)))


GENERATORS: dict[str, Callable[[random.Random, str], Item]] = {
    "extract": gen_extract,
    "classify": gen_classify,
    "compare": gen_compare,
    "consistency": gen_consistency,
    "count": gen_count,
    "rule": gen_rule,
    "temporal": gen_temporal,
    "negation": gen_negation,
    "ordinal": gen_ordinal,
}


def cells() -> list[str]:
    return [f"{op}/{fmt}" for op in OPERATIONS for fmt in FORMATS]


def to_record(it: Item, split: str) -> dict[str, Any]:
    keys = list(it.ref)
    return {
        "id": f"grid-{it.cell.replace('/', '-')}-{hashlib.sha1(json.dumps(it.state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]}",
        "family": f"ambiguous/{it.op}" if it.ambiguous else it.cell,  # ambiguous items are scored on calibration, not accuracy
        "operation": it.op,
        "format": it.fmt,
        "kind": it.kind,
        "heldout": split == "eval",
        "ambiguous": it.ambiguous,
        "state": it.state,
        "question": it.question,
        "refs": {"code": {k: it.ref[k] for k in keys}},
        "ref": {k: it.ref[k] for k in keys},
        "source": "grid",
    }


def generate(per_cell: int, seed: int, split: str = "train", only: set[str] | None = None) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}-{split}")
    out: list[dict[str, Any]] = []
    for cell in cells():
        if only and cell not in only:
            continue
        op, fmt = cell.split("/")
        seen: set[str] = set()
        tries = 0
        while len([r for r in out if cell_of(r) == cell]) < per_cell and tries < per_cell * 5:
            tries += 1
            rec = to_record(GENERATORS[op](rng, fmt), split)
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            out.append(rec)
    rng.shuffle(out)
    return out


def cell_of(rec: dict[str, Any]) -> str:
    return f"{rec['operation']}/{rec['format']}"
