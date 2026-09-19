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

## Decisions already taken for later phases

Recorded here so the reasoning is not lost; the implementation arrives with its phase.

### Synthetic timeline for `TransactionDT` (phase 2)

`TransactionDT` is a seconds offset from an unknown origin, not a timestamp. It is mapped onto a
fixed reference date (`SYNTHETIC_EPOCH`, default 2023-01-01) so the data has a real calendar, which
the windowed features, the chargeback delay simulation and the time-based split all need. The
absolute dates are fictional; the intervals between transactions are real.

### Card identity proxy (phase 3)

The dataset has no card identifier. The pipeline uses `card1 + addr1 + P_emaildomain`, a widely
used proxy for this dataset. It is imperfect in both directions — one card with two billing
addresses looks like two cards, and two cards sharing a household address and email domain can
collapse into one. It is documented as an approximation rather than presented as ground truth.

### Feature freshness at scoring time (phase 3 / 6)

Spark writes rolling per-card aggregates to Redis in micro-batches, so there is always a few
seconds of lag. If `/score` read Redis alone, the feature "transactions on this card in the last
10 minutes" would **exclude the transaction being scored** — which is exactly the signal that
catches a velocity attack.

So: Spark owns the historical aggregate, and the API applies the current transaction's delta on top
using the *same* function from `features/definitions.py`. One definition, two call sites. This is
what makes the online/offline consistency test meaningful rather than trivially true.
