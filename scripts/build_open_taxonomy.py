"""Choice rows over option lists the model has never seen: user-defined taxonomies.

A tagger built on circuits asks `choice` questions whose options are
the user's own labels with the user's own one-line descriptions, often with an "other"
catch-all. v2.0 and the 8B learned choice on fixed label sets (Banking77's 77 intents, every
time, in full) and answer an unfamiliar list by piling onto "other". These rows teach the
skill itself: read a list of labels with descriptions, pick the one that fits when it is
there, and pick "other" only when it is not.

    uv run python scripts/build_open_taxonomy.py
    # data/open_tax_train.jsonl, data/open_tax_eval.jsonl
    uv run python scripts/build_open_taxonomy.py
    # data/open_tax_tagger_eval.jsonl comes from the tagger's own repo (not in git: WildChat text, Jev labels as a reference)

Sources, all human or code labels (no teacher output anywhere in train):

  dolly_task      Dolly-15k       instruction (+ its human response) -> category (8)   CC BY-SA 3.0
  dbpedia_l1/l2   DBPedia_Classes article -> class, 9 top classes -> 70 subclasses    CC BY-SA 3.0 text
  massive_*       MASSIVE en-US   utterance -> scenario (18) -> intent                  CC BY 4.0
  banking77_sub   Banking77       message -> intent, a sampled subset of the 77         CC BY 4.0
  clinc_sub       CLINC150        utterance -> intent subset; out-of-scope ones are "other" CC BY 3.0
  dolly_workload  Dolly text      a person's question vs a program's templated call     code labels
  dolly_env       Dolly text      a real request vs test, placeholder or probing input  code labels
  dolly_data      Dolly text      general knowledge vs the same with personal data added code labels

Held out (never in train): no_robots categories (CC BY-NC 4.0, eval only), MMLU subjects
(MIT), two whole DBpedia top classes with their subclasses, and a Dolly slice by id hash.

How a row varies, so no single wording or list is learnt:
  - 4 to 14 options sampled from the label set (all of them when the set is smaller), in
    random order; keys are the dataset's label names in snake_case
  - descriptions from one of several templates over the label name, or none (bare keys);
    small sets (Dolly's 8) have written descriptions, varied by the same templates
  - an "other" option on most rows (key and wording varied); on ~1 row in 5 the gold label
    is removed and "other" is the answer. CLINC's out-of-scope utterances are always "other"
  - parent -> child rows ask the child over the options of the true parent only, with the
    parent named in the question ("This conversation is {parent} work. Which kind?")
  - states as a bare string, or as chat transcript lines ("user: ...", "assistant: ...")
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

GOLD_OUT = 0.2  # rows whose gold label is removed so "other" is right
OTHER_ON = 0.75  # rows (with the gold present) that still offer "other"
MAX_TEXT = 700  # a transcript line is cut to this, as a tagger's transcript does

OTHER_KEYS = [
    ("other", "None of the above"),
    ("other", "None of these"),
    ("other", "Something else"),
    ("none", "None of these fit"),
    ("other", None),
    ("not_listed", "Not in this list"),
]

INSTRUCTIONS = {
    "task": [
        "What kind of work was the assistant asked to do? Pick the single best fit.",
        "Which category best describes the request?",
        "What type of task is this?",
        "Classify the request by the kind of work it asks for.",
    ],
    "topic": [
        "Which area does this text belong to?",
        "What is this about? Pick the closest category.",
        "Which of these categories fits the text best?",
        "Tag this text with its category.",
    ],
    "intent": [
        "What does the person want?",
        "Which intent does this message express?",
        "Route this message: which of these is it?",
        "Which of these best matches what the user is asking for?",
    ],
    "child": [
        "This is {parent}. Which kind?",
        "Given that this is {parent}, which of these is it?",
        "This conversation is {parent} work. Which kind?",
        "Within {parent}, pick the best fit.",
    ],
}

DOLLY = {  # the category descriptions, written once; rows vary them with the templates below
    "open_qa": "Answering an open question from general knowledge",
    "general_qa": "A general question that needs some explanation or opinion",
    "closed_qa": "Answering a question from a passage that is given",
    "classification": "Sorting or labelling given items into categories",
    "information_extraction": "Pulling specific facts or items out of given text",
    "summarization": "Condensing a given passage",
    "brainstorming": "Coming up with a list of ideas or options",
    "creative_writing": "Writing a story, poem, letter or other original prose",
}

NO_ROBOTS = {
    "Generation": "Writing new text from a prompt",
    "Open QA": "Answering a question from general knowledge",
    "Brainstorm": "Listing ideas",
    "Chat": "Casual back-and-forth conversation",
    "Rewrite": "Rephrasing or editing given text",
    "Summarize": "Condensing given text",
    "Coding": "Writing or explaining code",
    "Classify": "Labelling given text",
    "Closed QA": "Answering from a given passage",
    "Extract": "Pulling items out of given text",
}


def snake(s: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)  # CamelCase (DBpedia's SportsManager) -> sports_manager
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def human(key: str) -> str:
    return key.replace("_", " ")


def describe(key: str, written: str | None, style: int) -> str | None:
    """One of several description styles, chosen per row so a list reads consistently."""
    h = human(key)
    if written:
        return [written, written, f"{h.capitalize()}: {written[0].lower() + written[1:]}", written, None][style % 5]
    return [None, h.capitalize(), f"About {h}", f"Anything to do with {h}", f"The text concerns {h}"][style % 5]


def cut(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_TEXT else text[:450] + " [...] " + text[-200:]


def as_state(rng: random.Random, user: str, assistant: str | None = None):
    """A bare string, or chat transcript lines."""
    if rng.random() < 0.5:
        return user if assistant is None or rng.random() < 0.5 else f"{user}\n\n{assistant}"
    lines = [f"user: {cut(user)}"]
    if assistant:
        lines.append(f"assistant: {cut(assistant)}")
    return {"transcript": lines}


def row(family: str, state, instructions: str, criteria: dict, gold: str, heldout: bool) -> dict:
    ref = {k: float(k == gold) for k in criteria}
    q = {"type": "choice", "instructions": instructions, "criteria": criteria}
    key = json.dumps([state, q], sort_keys=True, ensure_ascii=False)
    return {
        "id": f"{family}-{hashlib.sha1(key.encode()).hexdigest()[:10]}",
        "family": family,
        "kind": "choice",
        "heldout": heldout,
        "state": state,
        "question": q,
        "refs": {"jev": ref, "gemini": ref},  # no teachers; the human or code label stands in for both
        "ref": ref,
        "source": "open_taxonomy",
    }


def noul(family: str, state, instructions: str, true: str, false: str, answer: bool, heldout: bool) -> dict:
    ref = {"yes": float(answer), "no": float(not answer)}
    q = {"type": "noul", "instructions": instructions, "criteria": {"true": true, "false": false}}
    key = json.dumps([state, q], sort_keys=True, ensure_ascii=False)
    return {
        "id": f"{family}-{hashlib.sha1(key.encode()).hexdigest()[:10]}",
        "family": family,
        "kind": "noul",
        "heldout": heldout,
        "state": state,
        "question": q,
        "refs": {"code": ref},
        "ref": ref,
        "source": "open_taxonomy",
    }


def options(rng: random.Random, labels: list[str], gold: str | None, written: dict[str, str] | None = None, lo: int = 4, hi: int = 14, force_out: bool = False):
    """(criteria, answer key): a sampled list around the gold, maybe with "other", maybe without the gold."""
    written = written or {}
    out = force_out or gold is None or rng.random() < GOLD_OUT
    pool = [k for k in labels if k != gold]
    n = min(len(labels) - (0 if out else 1), rng.randint(lo, hi) - (0 if out else 1))
    picked = rng.sample(pool, min(n, len(pool)))
    if not out:
        picked.append(gold)
    rng.shuffle(picked)
    style = rng.randrange(5)
    crit = {k: describe(k, written.get(k), style) for k in picked}
    if out or rng.random() < OTHER_ON:
        ok, od = rng.choice(OTHER_KEYS)
        if ok in crit:
            ok = "none_of_these"
        crit[ok] = od if crit[picked[0]] is not None else None
        if rng.random() < 0.5:  # "other" last, as people write it, or anywhere
            items = list(crit.items())
            rng.shuffle(items)
            crit = dict(items)
        return crit, (ok if out else gold)
    return crit, gold


def split(key: str, heldout_frac: float = 0.1) -> bool:
    """True for the held-out slice, by a stable hash."""
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < heldout_frac


# ---------------------------------------------------------------- sources
def dolly() -> list[dict]:
    path = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    return [json.loads(line) for line in open(path)]


def dolly_rows(rng, data, n, heldout: bool) -> list[dict]:
    rows = []
    items = [d for d in data if split(d["instruction"]) == heldout]
    labels = list(DOLLY)
    for d in rng.sample(items, min(n, len(items))):
        user = d["instruction"] + (f"\n\n{d['context']}" if d["context"] else "")
        crit, gold = options(rng, labels, d["category"], DOLLY, lo=4, hi=8)
        rows.append(row("dolly_task", as_state(rng, user, d["response"]), rng.choice(INSTRUCTIONS["task"]), crit, gold, heldout))
    return rows


def dbpedia(split_name: str) -> list[dict]:
    path = hf_hub_download("DeveloperOats/DBPedia_Classes", f"DBPEDIA_{split_name}.csv", repo_type="dataset")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def dbpedia_rows(rng, data, n, heldout_l1: set[str], heldout: bool) -> list[dict]:
    rows = []
    l1s = sorted({d["l1"] for d in data})
    children: dict[str, list[str]] = collections.defaultdict(list)
    for d in data:
        if snake(d["l2"]) not in children[d["l1"]]:
            children[d["l1"]].append(snake(d["l2"]))
    keep = [d for d in data if (d["l1"] in heldout_l1) == heldout]
    all_l2 = sorted({snake(d["l2"]) for d in keep})
    for d in rng.sample(keep, min(n, len(keep))):
        text, l1, l2 = d["text"], snake(d["l1"]), snake(d["l2"])
        kind = rng.random()
        if kind < 0.3 and not heldout:  # top level: 9 classes, the held-out ones never offered in train
            crit, gold = options(rng, [snake(x) for x in l1s if x not in heldout_l1], l1, lo=4, hi=9)
            rows.append(row("dbpedia_l1", as_state(rng, text), rng.choice(INSTRUCTIONS["topic"]), crit, gold, heldout))
        elif kind < 0.7:  # child over the true parent's children
            kids = children[d["l1"]]
            if len(kids) < 2:
                continue
            crit, gold = options(rng, kids, l2, lo=3, hi=12)
            q = rng.choice(INSTRUCTIONS["child"]).format(parent=f"an article in the {human(l1)} category")
            rows.append(row("dbpedia_l2", as_state(rng, text), q, crit, gold, heldout))
        else:  # a flat list of subclasses from across the tree
            crit, gold = options(rng, all_l2, l2)
            rows.append(row("dbpedia_flat", as_state(rng, text), rng.choice(INSTRUCTIONS["topic"]), crit, gold, heldout))
    return rows


def massive_rows(rng, split_name: str, n: int) -> list[dict]:
    path = hf_hub_download("AmazonScience/massive", f"en-US/{split_name}/0000.parquet", repo_type="dataset", revision="refs/convert/parquet")
    ds = load_dataset("parquet", data_files=path, split="train")  # the parquet carries the ClassLabel names
    scen_names, intent_names = ds.features["scenario"].names, ds.features["intent"].names
    kids: dict[str, list[str]] = collections.defaultdict(list)
    for name in intent_names:
        s = name.split("_")[0]
        kids[s].append(name)
    rows = []
    for r in rng.sample(list(ds), min(n, len(ds))):
        scen, intent = scen_names[r["scenario"]], intent_names[r["intent"]]
        if rng.random() < 0.4:
            crit, gold = options(rng, scen_names, scen, lo=4, hi=12)
            rows.append(row("massive_scenario", as_state(rng, r["utt"]), rng.choice(INSTRUCTIONS["intent"]), crit, gold, False))
        else:
            sibs = [k.split("_", 1)[1] if "_" in k else k for k in kids[intent.split("_")[0]]]
            if len(sibs) < 2:
                continue
            child = intent.split("_", 1)[1] if "_" in intent else intent
            crit, gold = options(rng, sibs, child, lo=2, hi=10)
            q = rng.choice(INSTRUCTIONS["child"]).format(parent=f"a {human(scen)} request")
            rows.append(row("massive_intent_child", as_state(rng, r["utt"]), q, crit, gold, False))
    return rows


def banking_rows(rng, n: int) -> list[dict]:
    t = pq.read_table(hf_hub_download("mteb/banking77", "data/train-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    labels = sorted({snake(r["label_text"]) for r in t})
    return [
        row("banking77_sub", as_state(rng, r["text"]), rng.choice(INSTRUCTIONS["intent"]), *options(rng, labels, snake(r["label_text"])), False)
        for r in rng.sample(t, n)
    ]


def clinc_rows(rng, n: int) -> list[dict]:
    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = ds.features["intent"].names
    labels = [k for k in names if k != "oos"]
    rows = []
    for r in rng.sample(list(ds), n):
        gold = names[r["intent"]]
        crit, ans = options(rng, labels, None if gold == "oos" else gold)
        rows.append(row("clinc_sub", as_state(rng, r["text"]), rng.choice(INSTRUCTIONS["intent"]), crit, ans, False))
    return rows


def no_robots_rows(rng, n: int) -> list[dict]:
    t = pq.read_table(hf_hub_download("HuggingFaceH4/no_robots", "data/test-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    written = {snake(k): v for k, v in NO_ROBOTS.items()}
    rows = []
    for r in rng.sample(t, min(n, len(t))):
        msgs = r["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        reply = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        crit, gold = options(rng, list(written), snake(r["category"]), written, lo=4, hi=10)
        rows.append(row("no_robots_task", as_state(rng, user, reply), rng.choice(INSTRUCTIONS["task"]), crit, gold, True))
    return rows


def mmlu_rows(rng, n: int) -> list[dict]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    subjects = sorted(set(ds["subject"]))
    rows = []
    for r in rng.sample(list(ds), n):
        crit, gold = options(rng, subjects, r["subject"])
        rows.append(row("mmlu_subject", as_state(rng, r["question"]), rng.choice(INSTRUCTIONS["topic"]), crit, gold, True))
    return rows


# ---------------------------------------------------------------- code-labelled tags
AUTOMATED = [
    "Summarize the following text in three bullet points. Output only the bullets.\n\n###\n{text}\n###",
    'You are a classification engine. Classify the text into one of [positive, negative, neutral]. Respond with JSON {{"label": ...}}.\nText: {text}',
    "Extract every named entity from INPUT and return them as a JSON list of strings.\nINPUT: {text}",
    "Translate the text below into French. Return only the translation.\n\n{text}",
    "<task>rewrite</task><style>formal</style><input>{text}</input>",
    "SYSTEM: Generate 5 SEO keywords for the product description. Format: comma-separated, lowercase.\nDESCRIPTION: {text}",
    "Answer the question using only the context. If the answer is not there, reply NONE.\nContext: {text}\nQuestion: {question}\nAnswer:",
    "### Instruction\nWrite a one-sentence title for this document.\n### Document\n{text}\n### Title",
]
TESTS = [
    "test", "testing 123", "hello? test", "asdf", "ping", "can you see this", "lorem ipsum dolor sit amet, consectetur adipiscing elit",
    "TEST PROMPT PLEASE IGNORE", "this is a test of the system", "foo bar baz", "aaaa", "hi test test", "check check", "1 2 3",
    "{question} [test]", "{question} (just testing, ignore)", "Placeholder: {question}", "TODO replace with real prompt. {question}",
]  # fmt: skip
TEST_REPLIES = [
    "Hello! How can I help you today?",
    "It looks like you're testing. I'm here and working.",
    "Yes, I can see your message.",
    "Could you tell me more about what you need?",
]
FIRST = ["Maria", "James", "Aisha", "Wei", "Carlos", "Priya", "Tom", "Olga", "Kwame", "Sara", "Jonas", "Mei"]
LAST = ["Alvarez", "Okafor", "Chen", "Novak", "Smith", "Haddad", "Kowalski", "Tanaka", "Moreau", "Singh", "Brown", "Ivanova"]
PII = [
    "My name is {name}, born {dob}, SSN {ssn}. {question}",
    "Card {cc}, exp {exp}, CVV {cvv}, billing name {name}. {question}",
    "Patient {name} (MRN {mrn}, DOB {dob}) was diagnosed with {cond} last week. {question}",
    "{question} For reference my account number is {acct} and routing {routing}, name on the account {name}.",
    "Employee {name}, salary {salary}, home address {addr}. {question}",
]
CONDITIONS = ["type 2 diabetes", "major depressive disorder", "hypertension", "HIV", "asthma", "a herniated disc"]


def fake(rng) -> dict:
    d = lambda k: "".join(str(rng.randrange(10)) for _ in range(k))
    return {
        "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        "dob": f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(1950, 2005)}",
        "ssn": f"{d(3)}-{d(2)}-{d(4)}",
        "cc": f"4{d(3)} {d(4)} {d(4)} {d(4)}",
        "exp": f"{rng.randint(1, 12):02d}/{rng.randint(26, 31)}",
        "cvv": d(3),
        "mrn": d(8),
        "cond": rng.choice(CONDITIONS),
        "acct": d(10),
        "routing": d(9),
        "salary": f"${rng.randint(40, 250)},{d(3)}",
        "addr": f"{rng.randint(10, 9999)} {rng.choice(['Oak', 'Maple', 'Pine', 'Cedar', 'Elm'])} St, Apt {rng.randint(1, 40)}",
    }


WORKLOAD = [
    (
        {
            "interactive": "A person typing and reading the replies",
            "automated": "A program sending templated input: fixed instructions around swapped-in content",
        },
        "Was a person chatting, or a program sending templated input?",
    ),
    (
        {"human": "Someone chatting with the assistant", "pipeline": "An application calling the model with a fixed prompt"},
        "Who is on the other end of this request?",
    ),
    ({"interactive": None, "automated": None}, "Is this interactive use or an automated call?"),
    (
        {"chat": "Written by a person, for a person", "api_job": "Generated by software from a template", "other": "Can't tell"},
        "Where does this input come from?",
    ),
]
ENV = [
    (
        {"production": "Doing a real job: the output is meant to be used", "dev_test": "The content itself is a test: placeholder or dummy input, or 'test'"},
        "Is this doing a real job, or is the content itself a test?",
    ),
    ({"real": "A genuine request", "test": "A test, probe or placeholder, not a genuine request"}, "Is this a genuine request?"),
    ({"prod": None, "test": None}, "Production traffic or a test?"),
]
DATA = [
    (
        {"public": "Nothing that is not already public", "regulated": "Personal data about a person: health, payment, identity or account details"},
        "How sensitive is the most sensitive thing in this conversation?",
    ),
    (
        {
            "general": "General knowledge, nothing about a specific private person",
            "personal_data": "Identifies a private person or carries their health, financial or ID details",
        },
        "What kind of data does this contain?",
    ),
    ({"low": "No personal or confidential details", "pii": "Contains personally identifiable or financial information"}, "Classify the data sensitivity."),
]


def crit_of(rng, spec: dict) -> dict:
    items = list(spec.items())
    rng.shuffle(items)
    return dict(items)


def dolly_code_rows(rng, data, n: int, heldout: bool) -> list[dict]:
    rows = []
    items = [d for d in data if split(d["instruction"]) == heldout]
    for d in rng.sample(items, min(n, len(items))):
        q, ctx = d["instruction"], d["context"] or d["response"]
        which = rng.random()
        if which < 0.35:  # workload
            spec, instr = rng.choice(WORKLOAD)
            keys = list(spec)
            if rng.random() < 0.5:
                user, gold = rng.choice(AUTOMATED).format(text=ctx[:1500], question=q), keys[1]
            else:
                user, gold = q + (f"\n\n{d['context']}" if d["context"] and rng.random() < 0.5 else ""), keys[0]
            rows.append(row("dolly_workload", as_state(rng, user), instr, crit_of(rng, spec), gold, heldout))
        elif which < 0.65:  # environment
            spec, instr = rng.choice(ENV)
            keys = list(spec)
            if rng.random() < 0.45:  # a reply half the time on both sides, so its presence says nothing
                user, gold = rng.choice(TESTS).format(question=q), keys[1]
                reply = rng.choice(TEST_REPLIES + [d["response"]]) if rng.random() < 0.5 else None
            else:
                user, gold, reply = q, keys[0], d["response"] if rng.random() < 0.5 else None
            rows.append(row("dolly_env", as_state(rng, user, reply), instr, crit_of(rng, spec), gold, heldout))
        else:  # data class: the same request with and without personal data, labelled by construction
            if d["context"]:  # a passage may itself name private people; only bare questions are "public" by construction
                continue
            spec, instr = rng.choice(DATA)
            keys = list(spec)
            pii = rng.random() < 0.5
            user = rng.choice(PII).format(question=q, **fake(rng)) if pii else q
            gold = keys[1] if pii else keys[0]
            if rng.random() < 0.3:
                rows.append(
                    noul(
                        "dolly_pii",
                        as_state(rng, user),
                        "Does this contain personal data about a specific private person (identity, health, payment or account details)?",
                        "Yes, it does",
                        "No",
                        pii,
                        heldout,
                    )
                )
            else:
                rows.append(row("dolly_data", as_state(rng, user), instr, crit_of(rng, spec), gold, heldout))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every row count (0.05 for a smoke build)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    n = lambda k: max(4, int(k * args.scale))

    d = dolly()
    db_train, db_test = dbpedia("train"), dbpedia("test")
    heldout_l1 = {"Species", "Event"}
    train = [
        *dolly_rows(rng, d, n(3000), heldout=False),
        *dolly_code_rows(rng, d, n(3600), heldout=False),
        *dbpedia_rows(rng, db_train, n(4000), heldout_l1, heldout=False),
        *massive_rows(rng, "train", n(2500)),
        *banking_rows(rng, n(1200)),
        *clinc_rows(rng, n(1500)),
    ]
    evals = [
        *dolly_rows(rng, d, n(300), heldout=True),
        *dolly_code_rows(rng, d, n(450), heldout=True),
        *dbpedia_rows(rng, db_test, n(500), heldout_l1, heldout=True),
        *no_robots_rows(rng, n(500)),
        *mmlu_rows(rng, n(600)),
    ]
    rng.shuffle(train)
    out = Path("data")
    for name, rows in (("open_tax_train", train), ("open_tax_eval", evals)):
        with open(out / f"{name}.jsonl", "w") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        fam = collections.Counter(r["family"] for r in rows)
        other = sum(1 for r in rows if r["kind"] == "choice" and max(r["ref"], key=r["ref"].get) in {k for k, _ in OTHER_KEYS} | {"none_of_these"})
        print(f"{name}: {len(rows)} rows, 'other' is the answer on {other}; {dict(sorted(fam.items()))}")


if __name__ == "__main__":
    main()
