# Interview notes

Plain-English explanations of the concepts in this project, and ten questions you are likely to be
asked with answers grounded in this codebase.

Everything here points at code you can open. If an interviewer asks "where?", there is a file.

---

## Precision and recall, and why accuracy is useless here

Imagine 1,000 payments, 35 of them fraudulent.

- **Accuracy** — what fraction did I get right overall? Answer "not fraud" to all 1,000 and you are
  96.5% accurate, having caught nothing. That is why the word *accuracy* does not appear in any
  metric this project reports.
- **Precision** — of the payments I blocked, how many really were fraud? Low precision means real
  customers being stopped at checkout.
- **Recall** — of the fraud that existed, how much did I catch? Low recall means fraud getting
  through.

They trade against each other. Block everything scoring above 0.99 and precision is high, recall is
terrible. Block everything above 0.05 and you catch almost all the fraud, and block thousands of
real customers.

**PR-AUC** (precision-recall AUC, also called average precision) summarises that trade across every
threshold, and unlike ROC-AUC it does not flatter you when positives are rare — with 96.5%
negatives, the enormous true-negative count drowns out the false positives in ROC-AUC.

**And recall by count is not recall by money.** A model can catch 75% of fraudulent payments while
missing the expensive ones. `training/evaluate.py` reports fraud *value* caught versus missed
alongside recall, and `tests/test_evaluate.py::test_recall_by_count_and_by_value_can_disagree` pins
a case where 75% recall loses over 90% of the money.

## Data leakage and point-in-time correctness

Leakage is training on information you would not have had at the moment of the decision. It makes
offline metrics look wonderful and production fail. There are two doors it comes through, and this
project closes both:

**Feature leakage** — a feature computed from events that happened *after* the payment. A "card's
average amount" computed over the whole file includes the future. Every feature here is computed
from a window ending at the payment's own timestamp (`features/offline.py`), and
`tests/test_point_in_time.py::test_features_do_not_change_when_later_payments_arrive` builds the
Gold table over 150 payments and again over 300, requiring the first 150 rows to be identical.

**Label leakage** — training on a label that had not arrived yet. This is the subtle one in fraud,
because the label genuinely does not exist for weeks (see below).

## Delayed labels — the fact that shapes everything

Nobody knows a payment was fraudulent when it happens. They find out when the cardholder disputes
it and a chargeback works through the scheme — typically 7 to 60 days later. Legitimate payments
are never confirmed at all; they are *presumed* good once the dispute window closes.

`training/chargebacks.py` puts that delay back into the dataset. The consequence, measured on the
fixture:

| Training date | Labels known | Fraud rate among known |
|---|---|---|
| day of the last payment | 0 | — |
| + 30 days | 16 (1.6%) | **1.000** |
| + 61 days | 1000 (100%) | 0.035 |

Thirty days out, *everything you know is fraud*, because chargebacks are the only labels that have
arrived. Train on "all the labels we have" and the model learns that every payment is fraudulent.
The true rate is 3.5%.

`training/dataset.py` therefore includes a payment only when it has happened **and** its label had
arrived by the training date. It offers two policies for the immature remainder — exclude them
(honest, throws data away) or assume them legitimate (more data, mislabels recent fraud) — and
flags every assumed label so the choice stays visible.

## Online and offline features, and why they must agree

The model is trained on features computed in batch over history. It is *served* features computed
live, per payment. If those two disagree, the model is being fed something it never learned on and
nothing fails loudly — it just quietly gets worse. That is **training/serving skew**.

This project defines every feature once, in `features/definitions.py`. The online path folds
payments into a state object one at a time; the offline path expresses the same definitions as
Spark window functions, because folding 590k rows in Python would take minutes.

Two implementations of one definition is the classic way skew creeps in, so
`tests/test_feature_consistency.py` computes the features for the same payments **both ways** and
asserts all 18 agree to within 1e-6 — one parametrised test per feature, so a failure names the one
that drifted.

One detail worth being able to explain: **Redis stores state, not features.** A feature like
"payments on this card in the last 10 minutes" includes the payment being scored, and that payment
does not exist until the request arrives. So the cache holds the card's history and the API
computes on top of it.

## How SHAP works, at the level you need

A model score is a number. SHAP turns it into *contributions*: for this specific payment, how much
did each feature push the score up or down, and they sum to the score itself.

The intuition is a fair way of splitting credit. Ask: if the model knew nothing, then learned the
features one at a time in every possible order, how much did adding *this* feature move the
prediction on average? That average is the feature's contribution. Doing it exactly is
exponentially expensive in general — but for tree models there is an exact algorithm that is fast,
which is why it is usable on the scoring path rather than only in a notebook.

In this project (`training/explain.py`) only the risk-*increasing* contributions are surfaced: an
analyst reviewing a flagged payment wants to know what made it look bad. Reasons are computed for
review and block decisions only — nobody reads the explanation of a payment that went through, and
it costs ~7 ms.

## How the RAG retrieval works

1. A confirmed past case is turned into a **sentence** (`describe_case`): *"very large payment,
   decision block; first payment seen on this card; device not seen on this card before"*. Prose,
   not a feature dump, because the embedder was trained on prose.
2. An embedding model (`all-MiniLM-L6-v2`) turns that sentence into 384 numbers — a point in space
   where similar meanings sit close together.
3. Those vectors live in Postgres with the **pgvector** extension.
4. When a new payment is flagged, it gets the same treatment, and the database returns the nearest
   points by cosine distance — the most similar past cases.

Two things worth knowing that are easy to get wrong:

- **Only confirmed cases go in.** A case with no outcome has nothing to teach; retrieving it lets
  one unresolved payment vouch for another.
- **A payment never retrieves itself.** An identical match is a perfect, useless neighbour — and
  during evaluation it would quietly turn retrieval quality into 100%.

## How the LLM is kept honest, and how that is measured

**The LLM never computes a number.** Every figure — amount, velocity count, score, multiple of the
card's average — is calculated by the pipeline, placed in a facts object (`genai/facts.py`), and
handed to the model to write prose around. A 3B model asked to divide 847.20 by 92.15 will produce
a number; it will be wrong often enough to matter, and wrong in a way that reads as authoritative
on an analyst's screen at 2am.

Because the facts are an explicit object, *"did it stick to them?"* is a question code can answer.
Every response is:

1. parsed as JSON,
2. validated against a Pydantic schema,
3. **grounded** — every number in the text is checked against the facts, and every cited case id
   against what was actually retrieved,
4. given exactly one repair attempt, quoting the specific problem back,
5. and if it still fails, **thrown away**.

Nothing is better than something wrong. The app falls back to the facts and the SHAP reasons. An
analyst who sees no summary knows where they stand; one who sees an invented figure does not.

The evaluation (`eval/llm_eval.py`) measures factual accuracy, schema validity, retrieval quality
(do the retrieved cases share the outcome?) and latency, over a fixed set of flagged cases.

## Choosing the thresholds

There is no statistically correct place to put a threshold — only a business one. So the operating
point minimises expected cost: a missed fraud costs the transaction amount, a wrongly blocked
payment costs a fixed friction charge, a review costs an analyst's handling fee.

The asymmetry is the point. A €20 fraud and a €2,000 fraud are not the same miss, so "optimise F1"
is the wrong tool — it treats every error as equal when the business does not.

Two details worth defending:

- **Thresholds are tuned on the validation window, never on test.** Tuning on test means choosing
  your operating point using the data you then report on.
- **Three decisions, not two.** Between allow and block sits *review*, where a human looks. That is
  what makes a fraud system usable: the model does not have to be confident about everything.

---

# Ten questions, with answers

**1. Walk me through what happens when a payment arrives.**

It lands on a Kafka topic, keyed by a card identity proxy so one card's payments stay ordered. A
Spark Structured Streaming job writes the raw event to a Bronze Delta table and folds it into that
card's state in Redis. In parallel, a consumer calls `POST /score`: the API reads the card's state,
computes features with the shared definitions — including this payment — runs a YAML rules engine,
scores with an XGBoost model loaded from the MLflow registry, and returns allow, review or block
with SHAP reasons. Every decision is written to an audit log. Measured end to end: **p50 12.7 ms**
at concurrency 1.

**2. Why rules *and* a model? Isn't the model enough?**

Some decisions are not the model's to make. A card the fraud team has confirmed as compromised must
be blocked whatever the model thinks. A regulatory or commercial limit is policy, not prediction.
And when a new attack pattern appears on a Tuesday, an analyst can add a rule that afternoon
instead of waiting a week for a retrain. A blocking rule short-circuits the model entirely; a
review rule is a floor, not a veto — it can lift an allow but cannot stop the model blocking.

**3. How do you know your features aren't leaking?**

Two tests. One builds the feature table over 150 payments and then over 300, and requires the first
150 rows to be byte-identical — any feature that looked forward would move. The other takes a fraud
confirmed 50 days after the payment and asserts it is absent from a model trained at 30 days and
present at 61. Deferred, not discarded.

**4. Your model looks good offline. Why should I believe it works in production?**

You shouldn't, on that evidence alone — that's the point of the online/offline consistency test.
Both paths compute every feature and must agree to 1e-6. Beyond that: the split is time-based, the
thresholds are tuned on validation rather than test, and the labels respect the chargeback delay.
The honest caveat is that no model in this repository has been trained on the real dataset yet —
the README results table says *not yet measured on the real dataset* rather than showing you the
synthetic figure.

**5. You got p50 latency from 526 ms to 12.7 ms. How?**

By profiling rather than guessing. Two causes. `build_matrix` was written for training sets —
vectorised pandas over hundreds of thousands of rows — and on a single row it spent 21.5 ms
constructing 57 Series to hold one value each; `build_row` does the same work in 1.0 ms, with a
test asserting the two produce identical output. And XGBoost defaulted to one thread per core per
request, so eight concurrent requests on four cores produced 32 threads competing for them; the
served model is pinned to one thread. Throughput then stays flat at ~71 req/s across concurrency
levels, which tells you the service is CPU-bound in a single Python process — past that point you
are measuring the queue, not the service, and the fix is more workers.

**6. Tell me about a bug you found.**

`mlflow.xgboost.load_model` does not round-trip `enable_categorical` — a model logged with it
`True` comes back `False`. Predictions are unaffected (I verified them identical to 1e-6, because
the booster already knows its categorical splits), but `shap.TreeExplainer` reads that flag when it
is constructed and then builds its own DMatrix without it. So **every flagged payment reached the
analyst with an empty reasons list**, and the only evidence was a warning in a log nobody reads.
Six phases of green unit tests never caught it, because it takes a model that has actually been
through a registry. Two regression tests now cover it.

**7. How does the system handle failure?**

It degrades rather than stopping. If Redis is unreachable the card is treated as unknown, which
biases towards review — the safe direction — and the response says so. If the model registry is
unreachable the service starts in rules-only mode and `/health` reports it. If the LLM cannot
produce a grounded summary, the analyst sees the facts and SHAP reasons instead. If Kafka is not
reachable, decisions fall back to a local JSONL file so the audit trail survives. The principle: a
fraud platform that refuses to answer is worse than one that answers conservatively.

**8. How would you know the model has gone stale?**

Three signals, in order of how quickly they arrive. Prediction drift — the score distribution moves
— needs no labels at all, which matters because labels are weeks away. Feature drift says the
payments themselves have changed. And once chargebacks land, precision and recall per model version
over time (`agg_model_performance`), computed on matured labels only so recent days are honestly
absent rather than flattering. A weekly Airflow DAG retrains, and promotes the challenger to
champion **only if** it beats the incumbent's PR-AUC on the latest test window.

**9. Where is the PII, and what did you do about it?**

The dataset has no card number, but `card1 + addr1 + P_emaildomain` together identify a card, so
that proxy is tokenised with an HMAC under a secret salt in the Silver layer — HMAC rather than a
plain hash because the key space is small enough to brute-force. Tokenising is pointless if the
ingredients survive, so Silver drops `card1`, `card2`, `addr1` and `addr2`. **This costs model
performance** — `card1` is one of the stronger raw features in this dataset — and it is the right
trade. The strongest guard is a data-quality expectation on the exact column set: if a raw
identifier ever reappears in Silver, the job refuses to write and CI goes red.

**10. What would you do differently in production?**

Flink or Kafka Streams instead of Spark micro-batches, for genuinely per-event latency. A managed
feature store rather than Redis plus my own consistency test. More uvicorn workers — the current
throughput ceiling is one Python process. A real analyst-labelling workflow rather than my
simulated chargebacks. A considerably stronger model behind the assistant than a 3B local one, with
the same grounding checks kept in place, because those are what make its output safe rather than
the model's size. And the review threshold set by how many analysts you actually have, which is a
staffing decision the cost model should take as an input rather than discover.
