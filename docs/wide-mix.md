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
