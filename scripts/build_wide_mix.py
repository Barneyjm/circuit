"""Build a wide, shallow training mix from many small public HF datasets,
plus a matching eval set, in our JSONL format (state, question, ref).

The point is task *diversity*: generalization to unseen judgment tasks
comes from seeing many different tasks, not from depth on a few. Every
task here is a supervised classification or judgment problem with human
labels; the reference is one-hot.

    uv run python scripts/build_wide_mix.py --per-task 300 --eval-per-task 100 \
        --train data/wide_train.jsonl --eval data/wide_eval.jsonl

Leave-one-task-out: train with `train_lora.py --exclude-family <task>` and
score `eval_set.py --families <task>`; `family` is the task name below.

Datasets are read with streaming so nothing large is downloaded; a task
that fails to load is skipped and reported. All ids are parquet-backed
(no dataset scripts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset


@dataclass
class Task:
    name: str
    dataset: str
    kind: str  # noul | choice | score
    instructions: str
    labels: list[str]  # option keys (choice), score levels (score), or ["yes","no"] (noul)
    label_field: str
    config: str | None = None
    train_split: str = "train"
    eval_split: str = "test"
    criteria: dict[str, str | None] | list[str] | None = None  # choice descriptions or score rubric
    state: Callable[[dict], Any] = field(default=lambda r: r["text"])
    label: Callable[[dict], int | None] | None = None  # returns index into labels, None to skip
    keep: Callable[[dict], bool] = field(default=lambda r: True)
    trust_remote_code: bool = False
    revision: str | None = None  # e.g. refs/convert/parquet for script-based datasets
    label_names_field: str | None = None  # column holding the label's text when the label column is a plain int
    license: str = "unclear"  # ok | nc (non-commercial) | unclear; see docs/wide-mix.md


def idx(field_name: str):
    return lambda r: int(r[field_name]) if r[field_name] is not None and int(r[field_name]) >= 0 else None


def choice(keys: list[str], desc: dict[str, str | None] | None = None):
    return {k: (desc or {}).get(k) for k in keys}


TASKS: list[Task] = [
    # --- sentiment / polarity
    Task("sst2", "nyu-mll/glue", "noul", "Is the sentiment of this movie-review sentence positive?", ["yes", "no"], "label", config="sst2", eval_split="validation",
         criteria={"true": "Positive", "false": "Negative"}, state=lambda r: r["sentence"], label=lambda r: [1, 0][r["label"]] if r["label"] >= 0 else None, license="unclear"),
    Task("imdb", "stanfordnlp/imdb", "noul", "Is this movie review positive overall?", ["yes", "no"], "label",
         criteria={"true": "Positive", "false": "Negative"}, state=lambda r: r["text"][:1500], label=lambda r: [1, 0][r["label"]], license="unclear"),
    Task("rotten_tomatoes", "cornell-movie-review-data/rotten_tomatoes", "noul", "Is this review snippet positive?", ["yes", "no"], "label",
         criteria={"true": "Positive", "false": "Negative"}, label=lambda r: [1, 0][r["label"]], license="unclear"),
    Task("amazon_polarity", "fancyzhx/amazon_polarity", "noul", "Is this product review positive?", ["yes", "no"], "label",
         criteria={"true": "Positive", "false": "Negative"}, state=lambda r: {"title": r["title"], "review": r["content"][:1500]}, label=lambda r: [1, 0][r["label"]], license="unclear"),
    Task("yelp_stars", "Yelp/yelp_review_full", "score", "How many stars did this reviewer give?", ["0", "1", "2", "3", "4"], "label",
         criteria=["1 star: terrible", "2 stars: poor", "3 stars: okay", "4 stars: good", "5 stars: excellent"], state=lambda r: r["text"][:1500], label=idx("label"), license="nc"),
    Task("sst5", "SetFit/sst5", "score", "How positive is this sentence?", ["0", "1", "2", "3", "4"], "label",
         criteria=["Very negative", "Negative", "Neutral", "Positive", "Very positive"], label=idx("label"), license="unclear"),
    Task("financial_sentiment", "FinanceMTEB/financial_phrasebank", "choice", "What is the sentiment of this financial news sentence from an investor's point of view?", [], "label", eval_split="train",
         label=idx("label"), label_names_field="label_text", license="nc"),
    Task("tweet_sentiment", "cardiffnlp/tweet_eval", "choice", "What is the sentiment of this tweet?", ["negative", "neutral", "positive"], "label", config="sentiment",
         criteria=choice(["negative", "neutral", "positive"]), label=idx("label"), license="nc"),
    # --- emotion
    Task("emotion", "dair-ai/emotion", "choice", "Which emotion does this text express?", ["sadness", "joy", "love", "anger", "fear", "surprise"], "label", config="split",
         criteria=choice(["sadness", "joy", "love", "anger", "fear", "surprise"]), label=idx("label"), license="nc"),
    Task("tweet_emotion", "cardiffnlp/tweet_eval", "choice", "Which emotion does this tweet express?", ["anger", "joy", "optimism", "sadness"], "label", config="emotion",
         criteria=choice(["anger", "joy", "optimism", "sadness"]), label=idx("label"), license="nc"),
    # --- toxicity / abuse / stance
    Task("tweet_hate", "cardiffnlp/tweet_eval", "noul", "Is this tweet hateful (attacks a group on identity)?", ["yes", "no"], "label", config="hate",
         criteria={"true": "Hateful", "false": "Not hateful"}, label=lambda r: [1, 0][r["label"]], license="nc"),
    Task("tweet_offensive", "cardiffnlp/tweet_eval", "noul", "Is this tweet offensive?", ["yes", "no"], "label", config="offensive",
         criteria={"true": "Offensive", "false": "Not offensive"}, label=lambda r: [1, 0][r["label"]], license="nc"),
    Task("tweet_irony", "cardiffnlp/tweet_eval", "noul", "Is this tweet ironic?", ["yes", "no"], "label", config="irony",
         criteria={"true": "Ironic", "false": "Literal"}, label=lambda r: [1, 0][r["label"]], license="nc"),
    Task("stance_abortion", "cardiffnlp/tweet_eval", "choice", "What is this tweet's stance toward legalized abortion?", ["none", "against", "favor"], "label", config="stance_abortion",
         criteria=choice(["none", "against", "favor"], {"none": "No stance expressed"}), label=idx("label"), license="nc"),
    Task("stance_climate", "cardiffnlp/tweet_eval", "choice", "What is this tweet's stance toward the claim that climate change is a real concern?", ["none", "against", "favor"], "label", config="stance_climate",
         criteria=choice(["none", "against", "favor"], {"none": "No stance expressed"}), label=idx("label"), license="nc"),
    # --- NLI / entailment / paraphrase
    Task("rte", "nyu-mll/glue", "noul", "Does `premise` entail `hypothesis`?", ["yes", "no"], "label", config="rte", eval_split="validation",
         criteria={"true": "Entailed", "false": "Not entailed"}, state=lambda r: {"premise": r["sentence1"], "hypothesis": r["sentence2"]}, label=lambda r: [1, 0][r["label"]] if r["label"] >= 0 else None, license="unclear"),
    Task("snli", "stanfordnlp/snli", "choice", "Given `premise`, what is the status of `hypothesis`?", ["entailed", "neutral", "contradicted"], "label",
         criteria={"entailed": "Must be true if the premise is true", "neutral": "The premise does not settle it", "contradicted": "Must be false if the premise is true"},
         state=lambda r: {"premise": r["premise"], "hypothesis": r["hypothesis"]}, label=idx("label"), license="ok"),
    Task("qqp", "nyu-mll/glue", "noul", "Are these two questions asking the same thing?", ["yes", "no"], "label", config="qqp", eval_split="validation",
         criteria={"true": "Duplicates", "false": "Different questions"}, state=lambda r: {"question1": r["question1"], "question2": r["question2"]}, label=lambda r: [1, 0][r["label"]] if r["label"] >= 0 else None, license="nc"),
    Task("mrpc", "nyu-mll/glue", "noul", "Are these two sentences paraphrases of each other?", ["yes", "no"], "label", config="mrpc", eval_split="validation",
         criteria={"true": "Paraphrases", "false": "Not paraphrases"}, state=lambda r: {"sentence1": r["sentence1"], "sentence2": r["sentence2"]}, label=lambda r: [1, 0][r["label"]], license="nc"),
    Task("paws", "google-research-datasets/paws", "noul", "Do these two sentences mean the same thing?", ["yes", "no"], "label", config="labeled_final",
         criteria={"true": "Same meaning", "false": "Different meaning (word overlap is not enough)"}, state=lambda r: {"sentence1": r["sentence1"], "sentence2": r["sentence2"]}, label=lambda r: [1, 0][r["label"]], license="ok"),
    Task("qnli", "nyu-mll/glue", "noul", "Does `sentence` contain the answer to `question`?", ["yes", "no"], "label", config="qnli", eval_split="validation",
         criteria={"true": "Contains the answer", "false": "Does not"}, state=lambda r: {"question": r["question"], "sentence": r["sentence"]}, label=lambda r: [1, 0][r["label"]] if r["label"] >= 0 else None, license="ok"),
    Task("boolq", "google/boolq", "noul", "Based on `passage`, is the answer to `question` yes?", ["yes", "no"], "answer", eval_split="validation",
         criteria={"true": "Yes", "false": "No"}, state=lambda r: {"question": r["question"], "passage": r["passage"][:1500]}, label=lambda r: 0 if r["answer"] else 1, license="ok"),
    Task("wiki_qa", "microsoft/wiki_qa", "noul", "Does `sentence` answer `question`?", ["yes", "no"], "label",
         criteria={"true": "Answers it", "false": "Does not answer it"}, state=lambda r: {"question": r["question"], "sentence": r["answer"]}, label=lambda r: [1, 0][r["label"]], license="unclear"),
    # --- acceptability / subjectivity / question type
    Task("cola", "nyu-mll/glue", "noul", "Is this sentence grammatically acceptable English?", ["yes", "no"], "label", config="cola", eval_split="validation",
         criteria={"true": "Acceptable", "false": "Unacceptable"}, state=lambda r: r["sentence"], label=lambda r: [1, 0][r["label"]] if r["label"] >= 0 else None, license="unclear"),
    Task("subj", "SetFit/subj", "noul", "Is this sentence subjective (an opinion) rather than objective (a fact)?", ["yes", "no"], "label",
         criteria={"true": "Subjective", "false": "Objective"}, label=lambda r: [1, 0][r["label"]], license="unclear"),
    Task("trec", "CogComp/trec", "choice", "What kind of answer is this question looking for?", ["abbreviation", "entity", "description", "human", "location", "numeric"], "coarse_label",
         criteria={"abbreviation": "An abbreviation or its expansion", "entity": "A thing (animal, color, product, ...)", "description": "A definition, reason, or explanation", "human": "A person or group", "location": "A place", "numeric": "A number, date, or amount"},
         label=idx("coarse_label"), revision="refs/convert/parquet", license="unclear"),
    # --- topic (wider option lists)
    Task("ag_news", "fancyzhx/ag_news", "choice", "Which section does this news article belong in?", ["world", "sports", "business", "sci_tech"], "label",
         criteria=choice(["world", "sports", "business", "sci_tech"]), label=idx("label"), license="nc"),
    Task("dbpedia", "fancyzhx/dbpedia_14", "choice", "What kind of entity is this encyclopedia article about?",
         ["company", "educational_institution", "artist", "athlete", "office_holder", "mean_of_transportation", "building", "natural_place", "village", "animal", "plant", "album", "film", "written_work"], "label",
         criteria=choice(["company", "educational_institution", "artist", "athlete", "office_holder", "mean_of_transportation", "building", "natural_place", "village", "animal", "plant", "album", "film", "written_work"]),
         state=lambda r: {"title": r["title"], "content": r["content"][:1200]}, label=idx("label"), license="ok"),
    Task("yahoo_topics", "community-datasets/yahoo_answers_topics", "choice", "Which topic does this question belong to?",
         ["society_culture", "science_math", "health", "education_reference", "computers_internet", "sports", "business_finance", "entertainment_music", "family_relationships", "politics_government"], "topic",
         criteria=choice(["society_culture", "science_math", "health", "education_reference", "computers_internet", "sports", "business_finance", "entertainment_music", "family_relationships", "politics_government"]),
         state=lambda r: {"title": r["question_title"], "body": r["question_content"][:800]}, label=idx("topic"), license="nc"),
    Task("newsgroups", "SetFit/20_newsgroups", "choice", "Which newsgroup was this post made to?", [], "label",
         state=lambda r: r["text"][:1500], label=idx("label"), label_names_field="label_text", license="unclear"),
    Task("banking77", "mteb/banking77", "choice", "Which banking intent does this customer message express?", [], "label",
         label=idx("label"), label_names_field="label_text", license="ok"),
    Task("massive_intent", "AmazonScience/massive", "choice", "Which assistant intent does this utterance express?", [], "intent", eval_split="validation",
         state=lambda r: r["utt"], label=idx("intent"), keep=lambda r: r["locale"] == "en-US", revision="refs/convert/parquet", license="ok"),
]


def item(task: Task, state: Any, ref_idx: int, criteria, split: str) -> dict:
    keys = task.labels
    ref = {k: (1.0 if i == ref_idx else 0.0) for i, k in enumerate(keys)}
    q: dict[str, Any] = {"type": task.kind, "instructions": task.instructions, "criteria": criteria}
    return {
        "id": f"{task.name}-{hashlib.sha1(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]}",
        "family": task.name,
        "kind": task.kind,
        "heldout": split == "eval",
        "state": state,
        "question": q,
        "refs": {"human": ref},
        "ref": ref,
        "source": f"hf:{task.dataset}" + (f":{task.config}" if task.config else ""),
    }


def resolve_labels(task: Task, ds) -> None:
    """Fill in label names and criteria from the dataset's ClassLabel when the task left them empty."""
    if task.labels:
        return
    if task.label_names_field:
        pairs: dict[int, str] = {}
        for row in ds.take(5000):
            pairs[int(row[task.label_field])] = str(row[task.label_names_field])
        raw = [pairs[i] for i in range(max(pairs) + 1)]
    else:
        raw = ds.features[task.label_field].names
    names = [n.replace(" ", "_").replace("/", "_").replace(".", "_").replace("-", "_") for n in raw]
    task.labels = names
    if task.criteria is None:
        task.criteria = choice(names)


def draw(task: Task, split: str, n: int, seed: int, exclude_ids: set[str]) -> list[dict]:
    ds = load_dataset(task.dataset, task.config, split=split, streaming=True, trust_remote_code=task.trust_remote_code, revision=task.revision)
    resolve_labels(task, ds)
    ds = ds.shuffle(seed=seed, buffer_size=20_000)
    out: list[dict] = []
    seen: set[str] = set()
    for row in ds:
        if not task.keep(row):
            continue
        li = task.label(row) if task.label else idx(task.label_field)(row)
        if li is None or li < 0 or li >= len(task.labels):
            continue
        state = task.state(row)
        if not state or (isinstance(state, str) and len(state.strip()) < 3):
            continue
        criteria = task.criteria if task.criteria is not None else ({"true": "Yes", "false": "No"} if task.kind == "noul" else choice(task.labels))
        it = item(task, state, li, criteria, "eval" if split != task.train_split else "train")
        if it["id"] in exclude_ids or it["id"] in seen:
            continue
        seen.add(it["id"])
        out.append(it)
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=300)
    ap.add_argument("--eval-per-task", type=int, default=100)
    ap.add_argument("--train", default="data/wide_train.jsonl")
    ap.add_argument("--eval", default="data/wide_eval.jsonl")
    ap.add_argument("--only", default=None, help="comma-separated task names")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--commercial", action="store_true", help="train only on tasks with license=ok; everything still goes to eval")
    args = ap.parse_args()

    tasks = TASKS if not args.only else [t for t in TASKS if t.name in set(args.only.split(","))]
    train: list[dict] = []
    ev: list[dict] = []
    failed: list[tuple[str, str]] = []
    for t in tasks:
        try:
            e = draw(t, t.eval_split, args.eval_per_task, args.seed, set())
            if t.eval_split == t.train_split:  # single-split dataset: carve eval out of train
                tr = draw(t, t.train_split, args.per_task, args.seed + 1, {x["id"] for x in e})
            else:
                tr = draw(t, t.train_split, args.per_task, args.seed + 1, {x["id"] for x in e})
            for x in e:
                x["heldout"] = True
            print(f"{t.name:<20} {t.kind:<6} {len(t.labels):>3} opts  train={len(tr):<4} eval={len(e)}", flush=True)
            if not args.commercial or t.license == "ok":
                train += tr
            ev += e
        except Exception as exc:
            failed.append((t.name, f"{type(exc).__name__}: {str(exc)[:160]}"))
            print(f"{t.name:<20} FAILED {type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr, flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(ev)
    for path, rows in ((args.train, train), (args.eval, ev)):
        with open(path, "w") as f:
            for it in rows:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\ntrain {len(train)} items over {len(Counter(i['family'] for i in train))} tasks -> {args.train}")
    print(f"eval  {len(ev)} items -> {args.eval}")
    if failed:
        print("failed:")
        for name, why in failed:
            print(f"  {name}: {why}")


if __name__ == "__main__":
    main()
