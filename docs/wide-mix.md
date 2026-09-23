# Wide-mix training data and its licenses

`scripts/build_wide_mix.py` draws 300 train + 100 eval items from each
of 32 small public tasks (9,600 / 3,200 total, `data/wide_train.jsonl`,
`data/wide_eval.jsonl`). The point is task diversity: generalization to
unseen judgment tasks comes from many tasks seen shallowly, not from
depth on a few. Leave-one-task-out is `train_lora.py --exclude-family X`
then `eval_set.py --families X`.

## License audit (2026-09-18)

Verdicts are for **training a model that could ship commercially**.
For an internal research prototype every source below is fine to use.
Hub `license` tags are incomplete (many say `other` or `unknown`), so
the verdict comes from the original release where the tag is silent.
`ok` = permissive or CC-BY family. `nc` = the release says
non-commercial or research-only. `unclear` = no license stated by the
original authors; widely used in papers, but nothing grants commercial
use. Non-commercial and unclear sources are **eval-only** under
`--commercial`, which is how the safe file was cut:
`data/wide_train_commercial.jsonl` (2,100 items, 7 tasks).

| task | source | verdict | basis |
|---|---|---|---|
| snli | stanfordnlp/snli | ok | CC BY-SA 4.0 |
| qnli | nyu-mll/glue | ok | SQuAD-derived, CC BY-SA 4.0 |
| paws | google-research-datasets/paws | ok | Google: free for any purpose |
| boolq | google/boolq | ok | CC BY-SA 3.0 |
| dbpedia | fancyzhx/dbpedia_14 | ok | CC BY-SA 3.0 |
| banking77 | mteb/banking77 | ok | CC BY 4.0 (PolyAI), MIT mirror |
| massive_intent | AmazonScience/massive | ok | CC BY 4.0 |
| yelp_stars | Yelp/yelp_review_full | nc | Yelp dataset terms: academic use only |
| financial_sentiment | FinanceMTEB/financial_phrasebank | nc | CC BY-NC-SA 3.0 |
| tweet_* and stance_* (7 tasks) | cardiffnlp/tweet_eval | nc | Twitter content; redistributed for research |
| emotion | dair-ai/emotion | nc | card: educational and research purposes only |
| qqp | nyu-mll/glue | nc | Quora release: non-commercial |
| mrpc | nyu-mll/glue | nc | Microsoft Research data license, non-commercial |
| ag_news | fancyzhx/ag_news | nc | AG corpus: non-commercial use |
| yahoo_topics | community-datasets/yahoo_answers_topics | nc | Yahoo Webscope: research only |
| sst2, sst5, rotten_tomatoes, subj | Pang & Lee / Stanford | unclear | no license stated |
| imdb | stanfordnlp/imdb | unclear | scraped reviews, no license stated |
| amazon_polarity | fancyzhx/amazon_polarity | unclear | mirror says Apache 2.0; underlying Amazon reviews have no grant |
| rte, cola | nyu-mll/glue | unclear | mixed / no license stated |
| wiki_qa | microsoft/wiki_qa | unclear | Microsoft Research data license; read it before commercial use |
| trec | CogComp/trec | unclear | no license stated |
| newsgroups | SetFit/20_newsgroups | unclear | public Usenet posts, no license stated |
| maud_deal (hard tier) | theatticusproject/maud | ok | CC BY 4.0 |
| sharc_policy (hard tier) | UCLNLP/sharc | ok | CC BY-SA 3.0 |
| cuad_clause (hard tier) | theatticusproject/cuad-qa | ok | CC BY 4.0 |
| musique_hops (hard tier) | dgslibisey/MuSiQue | ok | CC BY 4.0 |
| wiki2_hops (hard tier) | xanhho/2WikiMultihopQA | ok | Apache 2.0 |
| musique_locate, wiki2_locate, sharc_locate, cuad_locate, cuad_multi (v2 types) | as the hard tier | ok | as the hard tier |
| squad_locate (v2 types) | rajpurkar/squad_v2 | ok | CC BY-SA 4.0 |
| goemo_multi (v2 types) | google-research-datasets/go_emotions | ok | Apache 2.0 (card); Reddit comments |
| esci_rank, esci_match (v2.1 types) | tasksource/esci (Amazon Shopping Queries) | ok | Apache 2.0; annotators' E/S/C/I labels |
| stackx_rank (v2.1 types) | HuggingFaceH4/stack-exchange-preferences | ok | CC BY-SA 4.0; answer text and vote scores only, no author fields |
| cuad_match, musique_match (v2.1 types) | as the hard tier | ok | as the hard tier |
| cf_snli (counterfactuals) | acmi-lab/counterfactually-augmented-data, NLI half | ok | Apache 2.0 edits of SNLI (CC BY-SA 4.0); the IMDb half is not used, its reviews carry no licence |
| cf_wiki2 (counterfactuals) | xanhho/2WikiMultihopQA | ok | Apache-2.0; edits are code (answer swapped, unused paragraph removed) |

## Hard tier (`scripts/build_hard_tier.py`)

Long documents and multi-hop reading, aimed at the two JevBench tiers where the open
models trail Jev (long_policy, multi_hop). Nothing is taken from JevBench itself. Rows
run up to about 4,000 Qwen3 tokens, so the training pipeline uses `--max-length 4096`.
The built JSONL is not committed; rebuild it with the script (fixed seeds).

- maud_deal: MAUD merger-agreement excerpts, one question per single-answer deal point, choice over that point's answers.
- sharc_policy: ShARC, with the policy widened to every rule snippet from the same government page (shuffled), choice of yes / no / not covered / need more information, balanced by label.
- cuad_clause: CUAD contracts cut into 13k-character windows, noul "does this part contain a {clause} provision?", half positive.
- musique_hops: MuSiQue answerable questions, choice of supported / wrong (an intermediate hop's answer is proposed) / not enough information (one hop's paragraph removed).
- wiki2_hops: 2WikiMultihopQA, noul on the gold answer or a flipped one.

## v2 question types (`scripts/build_v2_types.py`)

Rows for `locate` (point at the part of the state that answers, or none) and `multi`
(every option that applies, each with its own probability). Built from the hard-tier
sources plus SQuAD 2.0 and GoEmotions; GoEmotions keeps the share of raters per emotion
as the reference, so `multi` is trained against soft labels where they exist. About one
locate row in six has its answer removed from the state. Not committed; rebuild with the
script.

## v2.1 question types (`scripts/build_v21_types.py`)

`rank` (order the options; the reference is a grade per option) and `match` (pair each item
with an option or "none"). Rank: ESCI shopping searches with their products graded by
Amazon's annotators, and Stack Exchange answers graded by votes, eval on sites the train rows
never saw. Match: ESCI searches against their exact products with other searches'
substitutes as hard negatives, CUAD clause types against a contract's clauses, and MuSiQue
question steps against their paragraphs. Human or code labels throughout; no teacher output.

## Counterfactual groups (`scripts/build_counterfactuals.py`)

One item in several versions that differ only in the fact that decides it, so the answer has
to move with that fact and stay put when anything else changes. SNLI pairs with Kaushik et
al.'s four human edits each (premise or hypothesis rewritten to each other label), in the
wide mix's `snli` wording; 2Wiki paragraphs with the answer swapped for another answer of the
same relation everywhere it appears, and with a paragraph no hop uses removed. Rows carry a
`group`; `eval_set.py` reports the share of groups answered entirely right.

## Injection groups (`scripts/build_injections.py`)

Hard-tier items with an instruction planted in the state (in a document, in the person's own
words, or as a field of its own) pushing a wrong option; the label does not change, since the
planted text is addressed to the model, not about the case. Train rows are the injected copies
only (the clean items are already hard-tier rows); the eval keeps each clean item beside two
attacks, in phrasings the train rows never use, and `eval_set.py` reports how often the planted
text hijacked or flipped the answer. Derived from the hard tier's sources; no new licence.

## The earlier real-data set (`data/hf_train.jsonl`)

| source | verdict | basis |
|---|---|---|
| nyu-mll/multi_nli | ok | CC BY-SA 3.0 / OANC public domain by genre |
| google/civil_comments | ok | CC0 |
| ucirvine/sms_spam | ok | CC BY 4.0 (UCI) |
| clinc/clinc_oos | ok | CC BY 3.0 |
| tals/vitaminc | ok | CC BY-SA 3.0; human-written claims against Wikipedia revisions (`build_grounded_tools.py`) |
| rajpurkar/squad_v2 | ok | CC BY-SA 4.0; crowd-written questions and spans |
| benayas/snips | ok | SNIPS NLU benchmark, CC0; this mirror is tagged Apache-2.0 |
| Tobi-Bueck/customer-support-tickets | nc | CC BY-NC 4.0, eval only, never trained on |
| gorilla-llm/Berkeley-Function-Calling-Leaderboard | eval | Apache-2.0; a benchmark, so eval only by convention (`build_unseen_eval.py`) |
| pminervini/HaluEval | eval | Apache-2.0 mirror of the MIT original; eval only |
| metaeval/chaos-mnli-ambiguity | nc | ChaosNLI is CC BY-NC 4.0; eval only, output not committed |
| FastFit/hwu_64 | eval | HWU64 (NLU-Evaluation-Data) is CC BY 4.0; kept unseen on purpose, eval only |
| lmms-lab-encoder/POPE | eval | MIT; images are COCO val2014 under their photographers' licences, so not committed |

## Two things that are not dataset licenses

- **Jev as a teacher.** The synthetic families' soft targets are an
  average of Jev's and Gemini's distributions. Training a model on
  another provider's outputs is commonly restricted by that provider's
  terms (Google's terms allow it for Gemini output with conditions;
  TypeSafe's should be read before any model trained on Jev outputs is
  distributed or sold). The human-labeled sets above carry no such
  question. For a commercial model, drop the Jev-derived targets or get
  TypeSafe's consent.
- **Base model.** Qwen3 weights are Apache 2.0.
