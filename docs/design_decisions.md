# Design decisions

Every entry records what was chosen, what it was chosen over, and what it costs. Decisions are
added as the phase that makes them lands, so this file grows through the project.

---

## Phase 1 — Foundation

### Kafka in KRaft mode, not ZooKeeper

KRaft has been production-ready since Kafka 3.3 and removes an entire container from the stack.
On a 16 GB laptop that is roughly 500 MB and one more thing to wait for on startup. The cost is
that KRaft is newer, so some older tutorials will not match this configuration.

### Redis as the online feature store

Feature lookups sit directly on the latency path: the API has a ~100 ms budget for the whole
request, so the feature read needs to be a sub-millisecond point lookup. Redis gives that with a
trivial operational model.

It is treated as a **cache, not a system of record** — the Delta Gold tables are the truth. That is
why the container runs with `maxmemory-policy allkeys-lru` and features carry a TTL: a card profile
that has not been updated in two days should expire rather than be scored against stale.

Rejected: a real feature store (Feast) — more concepts to explain than the problem needs here;
Postgres — fine, but a point lookup in Redis is an order of magnitude cheaper.

### DuckDB as the warehouse

Free, embedded, no server, and it reads Parquet directly. For analytical queries over a few hundred
thousand rows it is genuinely fast. The cost is real and documented: **DuckDB allows one writing
process at a time**, which is why the dashboard reads a snapshot copy rather than the live file
(see phase 9).

### The dataset is not committed; a synthetic fixture is

The IEEE-CIS data is competition data and its licence does not allow redistribution, so `data/` is
gitignored. But a test suite that cannot run without a Kaggle account is a test suite nobody runs.

`tests/synthetic.py` generates data with the **exact same schema** — 394 transaction columns and 41
identity columns, in the real order — so code written against the fixture works against the real
files. It is seeded, so the committed fixture is reproducible and CI fails if the generator and the
fixture drift apart.

It is explicitly **not** a statistical clone. Any model metric computed on it is meaningless and is
never published as a result.

### One typed Settings object

Configuration is read once, in `common/config.py`, and validated by pydantic. A test asserts that
`.env.example` and the `Settings` fields match in both directions — no undocumented variable, no
dead one. This is the cheapest possible defence against the classic Compose bug where a service
silently falls back to a default nobody knew existed.

### Exact version pins everywhere

Every Docker image and every Python package is pinned with `==`. A portfolio project's whole value
is that someone else can run it and get the same result; a floating dependency destroys that
quietly, months later.

One pin needs explaining: **numpy is held at 1.26.4**, below 2.0, because PySpark 3.5 and several of
the ML dependencies still build against the numpy 1.x ABI.

### Compose profiles instead of one big stack

The full system (Kafka, Spark, Redis, MLflow, Airflow, Postgres, Ollama, Superset) does not fit in
16 GB. Profiles (`core`, `ai`, `orchestration`, `dashboard`) let you run the part you are working
on. Each service also has an explicit `mem_limit`, so one runaway JVM cannot take the laptop down.

Services are added to the Compose file **in the phase that builds the code they run**, not up
front. A profile that starts a container with nothing to do is worse than no profile.

### Apache Superset / Streamlit for dashboards, not Power BI

Power BI was the obvious candidate given the author's background, and it was rejected for four
reasons:

1. **Sharing is not free.** Desktop is free, but publishing a dashboard so a reviewer can open a
   link requires a Pro licence. That breaks the project's "everything free" constraint at precisely
   the point that matters.
2. **It cannot be part of `docker compose up`.** Power BI Desktop is a Windows application; a
   reviewer on macOS or Linux gets nothing.
3. **No clean DuckDB connector.** It would mean a third-party ODBC driver — exactly the fragile
   dependency that makes a project fail on someone else's machine.
4. **`.pbix` is an opaque binary in Git.** It cannot be diffed or reviewed. A Superset dashboard
   exports as YAML and a Streamlit page is Python; both are reviewable code.

Power BI is still supported as an *optional extra*: the Gold tables are exported to Parquet, which
Power BI Desktop reads natively with no driver, and `dashboard/powerbi/` holds the `.pbix` and a
screenshot. See [dashboard/powerbi/README.md](../dashboard/powerbi/README.md).

---

## Phase 2 — Streaming ingestion

### The payment event carries no label

A real payment message cannot contain `isFraud`: at authorisation time nobody knows. Confirmation
arrives weeks later as a chargeback. So `PaymentEvent` has no label field at all, and the streaming
and scoring paths **cannot** leak it — not as a matter of discipline, but because the column is not
in the message. `tests/test_events.py::test_the_event_carries_no_label` pins that down.

Labels enter the platform separately, through the simulated chargeback table in phase 4.

### Kafka messages are keyed by the card identity proxy

Kafka guarantees ordering *within a partition*, not across a topic. Velocity features ("how many
payments has this card made in the last 10 minutes") are wrong if two payments on the same card are
processed out of order, so the card key is the partition key. Three partitions give some
parallelism while keeping each card's history sequential.

### Sparse column blocks travel as maps, not as 339 columns

The V block is about 90% empty. Sending 339 nulls per message wastes bandwidth, and a 339-column
Spark schema is unreadable. The populated values travel as `map<string,double>` instead. The same
applies to C, D, M and the identity columns, which are split into a numeric map and a categorical
map so that the id_01 scores do not have to be stringified and cast back.

### The Bronze schema is declared, not inferred

Schema inference on a stream looks at the first micro-batch only. If a later batch happens to have
a field the first one lacked, the table's schema and the data quietly diverge. The Bronze job
therefore declares the schema explicitly — and a test compares it field by field against the
Pydantic model, so a field added to the producer cannot silently go missing downstream.

### Bronze keeps the raw payload

Alongside the typed columns, every Bronze row keeps the original JSON string and its Kafka topic,
partition and offset. Bronze is the replay point: if a parsing assumption turns out to be wrong,
the raw payload is still there. Dropping it would make the mistake unrecoverable.

### Replay pacing is scheduled, not chained

The producer computes each event's send time against the *start* of the run, rather than sleeping
for the full gap after each send. Chaining sleeps makes a slow broker stretch the entire replay,
and the drift compounds. There is a test that proves the difference.

### Spark 3.5 needs Java 17

Spark 3.5 supports Java 8, 11 and 17 — Java 21 support landed in Spark 4. The Docker image pins the
JRE explicitly and CI installs Temurin 17 for the Spark job, rather than inheriting whatever the
base image or runner ships.

---

## Phase 3 — Features

### One definition, two implementations, reconciled by a test

`features/definitions.py` is the only place a feature is defined. The online path folds payments
into a state object one at a time; the offline path expresses the same definitions as Spark window
functions, because folding 590k rows in Python would take minutes and would not parallelise.

Two implementations of one definition is exactly how training/serving skew gets into a system: the
model learns from one set of numbers and is served another, and nothing fails — the model just
quietly gets worse. So `tests/test_feature_consistency.py` computes the features for the same
payments both ways and asserts every one of the 18 matches to within 1e-6, per feature, so a
failure names the one that drifted.

### Redis stores *state*, not features

The obvious design is to precompute features and cache them. It is wrong here: a feature like
"payments on this card in the last 10 minutes" includes the payment being scored, and that payment
does not exist until the request arrives. So Redis holds the card's history, and the API computes
the features on top of it with the same function the batch path uses.

This is also why the scoring path is not a straight cache read — it is a cache read plus a
deterministic computation, which is the part the consistency test protects.

### Sentinel values rather than NaN

`seconds_since_card_last_txn` is `-1` on a card's first payment. NaN would hide the difference
between "this card has no history" and "this value is missing", and those mean very different
things to a fraud model. XGBoost splits on a distinctive sentinel perfectly well.

### "New device" means nothing on a card's first payment

On a card the platform has never seen, every device is unfamiliar. Flagging that would make every
new card look suspicious, which is both useless and unfair to new customers. The novelty features
only fire once the card has history.

### Tokenisation: HMAC, and the identifying columns go too

The card key comes from a small space — a few thousand `card1` values, a few hundred `addr1`, about
sixty email domains. A plain SHA-256 of that is reversible by anyone willing to generate every
combination, so the token is an **HMAC** under a secret salt.

Tokenising is pointless if the columns the key was built from survive alongside it, so Silver drops
`card1`, `card2`, `addr1` and `addr2`. **This costs model performance** — `card1` is one of the
stronger raw features in this dataset — and it is the right trade for a system handling real
payments: the model gets the card's *behaviour* through the per-card features instead of its
identity. The email domain stays, because a domain on its own (sixty values, one of them
`anonymous.com`) identifies nobody and is genuinely predictive.

### The known divergence: bounded device tracking

Redis keeps the 20 most recent distinct devices per card; the offline window keeps all of them. For
a card with more than 20 distinct devices the two implementations could disagree.

Rather than pretend this is not there, a test asserts that no card in the data comes near the
bound. If that ever changes, the test fails and the divergence gets dealt with instead of silently
skewing a model.

### One streaming job writes Bronze *and* the online store

Two separate jobs would mean two Spark drivers, two JVM heaps and two reads of the same topic —
which does not fit on a 16 GB laptop. So one `foreachBatch` does both, Bronze first: if the Redis
update fails, the raw events are already durable and the state can be rebuilt from them. On a real
cluster these would be separate applications, so that a problem in the feature writer could not
stall ingestion.

State updates are not commutative ("was this device seen before?" depends on what came earlier), so
each batch is repartitioned by card and sorted within the partition before the fold.

### Both ends of a feature window are bounded

The trailing windows exclude events *after* the payment being scored, not just before the cutoff.
In the online path state only ever holds earlier payments, so the upper bound looks redundant — but
"cannot happen" is how leaks get in, and a late-arriving payment folded in after a newer one would
otherwise see its own future.

### Great Expectations: the column set is the real check

The most valuable expectation in the Silver suite is `expect_table_columns_to_match_set` with
`exact_match=True`. Value-level checks would never notice `card1` reappearing in Silver; the column
contract fails immediately. A data-protection regression becomes a red CI job.

The Silver job refuses to write a table that fails its suite, rather than writing it and warning.

### Two Spark gotchas, both now handled in `common/spark.py`

- **`PYSPARK_PYTHON`.** Spark launches Python workers with whatever `python3` is on PATH, which in
  a virtualenv is the wrong interpreter; the first pandas UDF then dies with "No module named
  pandas". The session builder points both ends at `sys.executable`.
- **Java 17, verified.** On Java 21 a session starts and ordinary queries work, but Arrow-based
  pandas UDFs fail with `sun.misc.Unsafe or java.nio.DirectByteBuffer.<init>(long, int) not
  available` — confirmed by running the Silver tests on both JVMs. The Docker image and CI pin
  Temurin 17, and the session builder warns if it finds anything newer.

---

## Phase 4 — Labels and training data

### Labels arrive weeks late, and the platform models that

In production nobody knows a payment was fraudulent when it happens. They find out when the
cardholder disputes it and the chargeback works through the scheme — typically 7 to 60 days later.
Legitimate payments are never confirmed at all; they are *presumed* good once the dispute window
closes.

The dataset hands us `isFraud` immediately, so `training/chargebacks.py` puts the delay back: fraud
gets a chargeback date inside the dispute window, everything else matures after 60 quiet days. The
label table is the only place in the platform that reads `isFraud`, and nothing on the streaming or
scoring path can see it.

Measured on the 1000-row fixture, which is exactly the behaviour being modelled:

| Training date | Labels known | Fraud rate among known |
|---|---|---|
| day of the last payment | 0 (0%) | — |
| + 10 days | 2 (0.2%) | 1.000 |
| + 30 days | 16 (1.6%) | 1.000 |
| + 61 days | 1000 (100%) | 0.035 |

The middle rows are the point. Thirty days after the fact, the only labels in hand are chargebacks
— so a training set built from "everything we know" is **100% fraud**. Train on that and the model
learns that every payment is fraudulent. The true rate is 3.5%.

### Delays are derived from the transaction ID, not drawn from a generator

Airflow loads this table a day at a time (phase 7). If a payment's delay came from a running RNG,
the timeline would shift every time the table was rebuilt, and every point-in-time guarantee with
it. Hashing `(seed, transaction_id)` gives the same answer whether the table is built in one pass
or in sixty daily chunks — there is a test for exactly that.

### Two policies for immature payments, because real teams disagree

On any training date, recent payments have no usable label yet. Two defensible choices, both
implemented:

* **`exclude`** (default) — leave them out. Honest, throws data away.
* **`assume_legitimate`** — include them as non-fraud, which is what a system that trusts "no
  dispute yet" effectively does.

Measured on the fixture at +30 days: `exclude` gives 16 rows at a 100% fraud rate; 
`assume_legitimate` gives 1000 rows at 1.6%, of which 984 labels are assumed — and 19 genuine
frauds are sitting in there labelled legitimate. Neither is free. The assumed rows carry a
`label_is_assumed` flag so the choice stays visible instead of being baked into the numbers.

### Point-in-time correctness has two halves

Feature leakage and label leakage are different defects and both are tested:

* **Features** — everything in Gold comes from a window ending at the payment's own timestamp.
  `test_features_do_not_change_when_later_payments_arrive` builds Gold over 150 payments and then
  over 300, and requires the first 150 rows to be byte-identical. A feature that looked forward —
  a card average over the whole file, a count with no upper bound — would move.
* **Labels** — a row is only included when `label_available_at <= as_of`. The test finds a fraud
  confirmed 50 days out, asserts it is absent from a model trained at 30 days, and then asserts it
  *is* present at 61 days: deferred, not discarded.

---

## Phase 5 — The model

### Accuracy is not reported anywhere

Fraud is 3.5% of payments, so a model that answers "not fraud" to everything is 96.5% accurate and
catches nothing. What is reported instead: **PR-AUC** (the summary metric for a rare positive
class), **precision and recall at the chosen operating points** rather than at 0.5, and **money** —
the fraud value caught versus missed.

That last one matters more than it looks. A model can catch 75% of fraudulent *payments* while
missing the expensive ones; there is a test (`test_recall_by_count_and_by_value_can_disagree`) that
pins exactly that case, where 75% recall lets over 90% of the fraud value walk out of the door.

ROC-AUC is logged for reference but not headlined: with 96.5% negatives, the huge true-negative
count drowns out the false positives and it flatters the model.

### Thresholds are a business decision, so they are chosen by cost

There is no statistically correct place to put a threshold, only a business one. So the operating
point minimises expected cost: a missed fraud costs the transaction amount, a wrongly blocked
payment costs a fixed friction charge, and a review costs an analyst's handling fee.

The asymmetry is the point. A 20 euro fraud and a 2,000 euro fraud are not the same miss, so
"optimal F1" is the wrong tool — it treats every error as equal when the business does not. Tests
assert the tuner responds the way a business would expect: raise the cost of annoying a customer
and fewer payments get blocked; raise the cost of an analyst and the review queue shrinks.

One optimistic assumption is baked in and worth saying out loud: a payment sent to review is
assumed to be resolved correctly. Real analysts are not perfect, so the review band looks slightly
cheaper here than it would in production.

The search is exact over a candidate grid rather than approximate: sort the scores once, precompute
cumulative sums, and the cost of any threshold pair becomes three array lookups — which makes a
100×100 grid instant even on a million rows.

### Thresholds are tuned on validation, never on test

Tuning the operating point on the test window means choosing it using the data the result is then
reported on. The number would be real but unreproducible in production. So: train on the training
window, tune thresholds on validation, report on a test window that neither the model nor the
thresholds ever saw.

### Imbalance: `scale_pos_weight`, not resampling

With 3.5% positives an unweighted model reaches a good loss by predicting "legitimate" almost
everywhere. Weighting the positive class by the imbalance ratio makes a missed fraud as expensive
to the loss function as it is to the business.

Chosen over SMOTE or undersampling because it changes no data: synthesising fraud rows would
invent card histories that never happened, and undersampling throws away the majority-class
detail that makes the rare signal stand out.

### Categorical vocabularies live in code, not in a fitted encoder

An encoder artifact can drift out of step with the model file and has to be loaded and versioned
separately at serving time. A vocabulary in code is versioned with the code, reviewable in a diff,
and impossible to forget to ship. The cost is that a genuinely new email domain is encoded as
"other" until somebody updates the list — a visible change rather than a silent re-encoding on the
next retrain.

Unknown and absent values are kept distinct (`other` vs `missing`): a payment with an unfamiliar
email domain and a payment with no email domain at all are not the same event.

### SHAP, because "0.87" is not a reason

A fraud analyst cannot act on a score. SHAP gives each feature a signed contribution to *this*
payment's score, and for tree models it is exact and fast — no sampling — which is what makes it
usable on the scoring path rather than only in a notebook. Only risk-*increasing* contributions are
surfaced: someone reviewing a flagged payment wants to know what made it look bad.

### Promotion rule: PR-AUC on the most recent test window

A challenger replaces the champion only if it beats the incumbent on the latest test window.
Explicit, checkable, and boring on purpose — phase 7 runs exactly this from Airflow each week.

### The fixture was made harder on purpose

The first training run on the synthetic fixture returned a PR-AUC of **1.000**. Perfect separation
tests nothing: every threshold is equally good, the cost model has no trade-off to make, and
anybody reading the output would reasonably assume the number was fabricated.

The generator's fraud and legitimate distributions now overlap heavily, and one outright giveaway
was removed (the recipient email domain was present for every fraud and absent for most legitimate
payments, which handed the model the label). Fixture numbers remain meaningless as a measure of
real performance — but the pipeline now has to do real work to produce them.

---

## Phase 6 — Serving

### Rules run before the model, and a block short-circuits it

Some decisions are not the model's to make: a card the fraud team has confirmed as compromised, a
policy limit, a brand-new attack pattern that cannot wait a week for a retrain. Those are rules,
they live in YAML, and an analyst can add one without a deployment.

A blocking rule skips the model entirely — it saves several milliseconds, and the model's opinion
was irrelevant anyway. A `review` rule is a **floor, not a veto**: it can lift an allow to a review,
but it cannot stop the model blocking outright.

The condition language has no `eval` in it. A rules file is configuration, and configuration that
can execute arbitrary Python is a remote code execution bug waiting for someone to edit the wrong
file. Unknown operators fail at load time rather than silently never matching.

### The API never writes to the feature store

The streaming job owns card state. If the API updated it too, a retried request would count the
same payment twice in its own card's history — and retries are normal. Scoring is read-only, so a
re-delivery produces the same decision twice rather than corrupting anything.

### Redis being down degrades the score, not the service

A fraud platform that refuses to answer is worse than one that answers conservatively. If the
feature store is unreachable the card is treated as unknown, which biases towards review — the safe
direction — and the degradation is explicit in the response and the audit record.

The same applies to the model: if the registry is unreachable or there is no champion yet, the
service starts in degraded mode with rules only, and says so in `/health`.

### Thresholds travel with the model version

They were tuned on that model's own validation window, so pairing version 7's model with version
4's thresholds is an operating point nobody ever evaluated. The loader reads both from the registry
together.

### SHAP runs only on flagged payments

Explanations cost about 7 ms. Nobody reads the reasons for an allowed payment, and ~97% of payments
are allowed, so explaining them would spend a fifth of the latency budget on output nobody looks
at. Reasons are computed for review and block decisions only — the same principle that keeps the
LLM off the common path in phase 8.

### Decisions go to Kafka, not to Delta inline

A synchronous table write on the scoring path puts file-system latency between a customer and their
payment. The topic is durable; phase 7's consumer lands it in Delta. When no broker is reachable
the sink falls back to a local JSONL file, so a laptop demo still leaves a complete audit trail
rather than silently discarding decisions.

### The latency work: 526 ms → 11 ms, by measuring

The first load test returned a p50 of **526 ms** against a 100 ms target. Profiling the scoring
path rather than guessing found two causes:

| Stage | Before | After |
|---|---|---|
| `build_matrix` on one row | 21.5 ms | 1.0 ms (`build_row`) |
| Model threads | 4 per request | 1 |
| Full `score()`, single thread | 36 ms | ~12 ms |

1. **`build_matrix` is written for a training set** — vectorised pandas over hundreds of thousands
   of rows. On a single row it spends 21.5 ms constructing 57 Series and six categorical dtypes to
   hold one value each. `build_row` builds the same one-row matrix directly, and a test asserts the
   two produce identical output — the same "two implementations, one contract" pattern as the
   online/offline features.
2. **XGBoost defaults to one thread per core, per request.** Eight in-flight requests on four cores
   produced 32 threads competing for them. A single-row prediction gains nothing from parallelism,
   so the served model is pinned to one thread.

Measured afterwards against **real services** - a real Redis, the champion model loaded from a real
MLflow registry, decisions published to a real Kafka broker, all on one four-core host:

| Concurrency | p50 | p95 | p99 | req/s |
|---|---|---|---|---|
| 1 | 12.7 ms | 20.3 ms | 21.4 ms | 71 |
| 2 | 27.0 ms | 42.8 ms | 49.4 ms | 68 |
| 4 | 62.9 ms | 94.9 ms | 125.9 ms | 62 |
| 8 | 135.3 ms | 195.9 ms | 290.9 ms | 58 |

The real Redis hop costs about 1.3 ms against an in-process fake (11.4 ms → 12.7 ms at concurrency
1), which is the honest price of a network round trip on the scoring path.

Throughput is flat at ~80 req/s across all of them, which says the service is CPU-bound in a single
Python process: past that point the latency being measured is the **queue**, not the service. The
production fix is more uvicorn workers, not more optimisation. This is why the load test sweeps
concurrency by default and names the unqueued run explicitly — quoting a saturated p50 as "our
latency" is the most common way a load test result misleads.

### The bug only a real registry could show: empty reasons

Running the platform against a real MLflow server turned up something no unit test had:
**`mlflow.xgboost.load_model` does not round-trip `enable_categorical`.** A model logged with it
set to `True` comes back with `False`.

Predictions are unaffected - verified by comparing scores from the in-process model against the
same model loaded from the registry, identical to within 1e-6, because the booster already knows
its categorical splits. But `shap.TreeExplainer` reads that flag when it is *constructed* and then
builds its own DMatrix without it. So every flagged payment reached the analyst with an empty
reasons list, and the only evidence was a warning in the API log that nobody reads.

The fix restores the parameter before the explainer is built. Two regression tests now cover it -
one asserting a registry-loaded model can still explain itself, one asserting its predictions are
bit-identical - and both would have caught this. The in-process tests could not: it takes a model
that has actually been through the registry.

This is the argument for integration testing against real infrastructure in one paragraph. Six
phases of green unit tests, a feature silently not working.

---

## Phase 7 — Orchestration and analytics

### Every Airflow task is a subprocess, not an import

The DAGs run `python -m streaming.silver` rather than importing the pipeline into the Airflow
worker. Three reasons, all practical:

* the Spark jobs need their own JVM and their own process anyway;
* Airflow's dependency set is large and opinionated — keeping the platform's dependencies out of it
  means an Airflow upgrade cannot break the model, and a pandas upgrade cannot break the scheduler;
* every job already has a command-line entry point, so **the DAG runs exactly what a human would
  type** at 3am when something has gone wrong.

### The quality gate is the pipeline's safety catch

`streaming/silver.py` refuses to write a table that fails its Great Expectations suite, so
everything downstream of it is either built on checked data or not built at all. A pipeline that
carries on past a failed quality gate — leaving the dashboard showing yesterday's numbers next to
today's date — is worse than one that stops.

### The warehouse reads a snapshot, not the live tables

DuckDB has a Delta extension and it is the obvious thing to reach for. This project publishes
Parquet snapshots instead, because the extension is downloaded at runtime: a laptop that is offline
— or behind a proxy that blocks the extension host, which is exactly what happened here — gets a
warehouse that cannot read anything.

It is also the more honest architecture. The Delta tables are the operational store; the warehouse
holds a snapshot that analysts and dashboards query without contending with the jobs that write it.
That separation is what makes the read-only Superset and Power BI paths possible.

### The warehouse distinguishes "decided" from "was right"

`fct_decisions` leaves the label null until `label_available_at` has passed, so every downstream
model has to be explicit about which question it is answering. Volumes — how many payments were
reviewed, how big the queue was, what the latency was — are known immediately. Accuracy is known
weeks later. Reporting the two as though they arrive together is the most common way a fraud
dashboard misleads, and `agg_daily_kpis` splits them into separate column groups for exactly that
reason.

A dbt test (`assert_labels_are_not_used_early.sql`) enforces it: if a label ever appears before its
chargeback arrived, the build fails. That is the reporting equivalent of training on the future.

### A rule that fires constantly and is rarely right is a tax

`agg_rule_effectiveness` exists because rules get written under time pressure and then never
revisited. Hit rate per rule, against matured labels, is what tells you a rule is protecting the
business rather than just filling the analyst queue.

### `dbt build`, not `dbt run`

The DAG runs `dbt build`, which runs the models *and* their tests, so a failing test stops the
pipeline instead of publishing a dashboard nobody should trust.

The range test is written out as a local macro rather than pulled from `dbt_utils`. That would be a
package download at build time, and this project is meant to run on a laptop with no network — one
macro is cheaper than a dependency.

### Retraining promotes on merit, or not at all

The weekly DAG rebuilds the training set **as of the run date**, so a re-run of an old week
reproduces that week's data exactly and a model trained there has not seen the future. It registers
the challenger every time; the `champion` alias moves only if PR-AUC on the latest test window
improves. A retrain that quietly promoted a worse model every Sunday would be worse than no
retrain, because nobody would be looking.

Verified in an actual DAG run: version 2 trained, scored identically to the incumbent, and was
**left as a challenger**.

### Drift produces a report, not an alert that blocks

Feature drift and prediction drift both mean *somebody should look*, not *stop the pipeline*.
Prediction drift is the earlier signal of the two because it needs no labels — and labels are weeks
away.

A second bug worth recording: Evidently's `DataDriftPreset` emits two metrics, a summary and a
per-column table, and an earlier version of the code read the summary's position for both. The
report said **"0 of 18 columns drifted (66.7%)"** — internally contradictory, and exactly what a
monitoring tool must never say. A monitoring tool that reports nonsense is worse than no monitoring,
so there is now a test asserting the count and the share describe the same thing.

---

## Decisions already taken for later phases

Recorded here so the reasoning is not lost; the implementation arrives with its phase.

### Synthetic timeline for `TransactionDT` (implemented in phase 2)

`TransactionDT` is a seconds offset from an unknown origin, not a timestamp. It is mapped onto a
fixed reference date (`SYNTHETIC_EPOCH`, default 2023-01-01) so the data has a real calendar, which
the windowed features, the chargeback delay simulation and the time-based split all need. The
absolute dates are fictional; the intervals between transactions are real.

### Card identity proxy (implemented in phase 3)

The dataset has no card identifier. The pipeline uses `card1 + addr1 + P_emaildomain`, a widely
used proxy for this dataset. It is imperfect in both directions — one card with two billing
addresses looks like two cards, and two cards sharing a household address and email domain can
collapse into one. It is documented as an approximation rather than presented as ground truth.

### Feature freshness at scoring time (implemented in phases 3 and 6)

Spark writes rolling per-card aggregates to Redis in micro-batches, so there is always a few
seconds of lag. If `/score` read Redis alone, the feature "transactions on this card in the last
10 minutes" would **exclude the transaction being scored** — which is exactly the signal that
catches a velocity attack.

So: Spark owns the historical aggregate, and the API applies the current transaction's delta on top
using the *same* function from `features/definitions.py`. One definition, two call sites. This is
what makes the online/offline consistency test meaningful rather than trivially true.
