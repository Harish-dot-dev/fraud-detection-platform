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

### Feature freshness at scoring time (implemented in phase 3; used by the API in phase 6)

Spark writes rolling per-card aggregates to Redis in micro-batches, so there is always a few
seconds of lag. If `/score` read Redis alone, the feature "transactions on this card in the last
10 minutes" would **exclude the transaction being scored** — which is exactly the signal that
catches a velocity attack.

So: Spark owns the historical aggregate, and the API applies the current transaction's delta on top
using the *same* function from `features/definitions.py`. One definition, two call sites. This is
what makes the online/offline consistency test meaningful rather than trivially true.
