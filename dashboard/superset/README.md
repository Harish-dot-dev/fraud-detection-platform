# Superset (optional)

The project's primary dashboard is the **Streamlit metrics page** (`make app`, then the *Metrics*
page). It is the one that is tested, screenshotted and guaranteed to run.

Superset is offered as an alternative for anyone who wants a click-to-build BI tool rather than a
Python page. It is **not** required, and it is the one component of the stack that has never been
run in this repository's own verification — the reasoning and the honest status are both below.

## Why it is optional rather than primary

DuckDB allows one writing process and takes a file lock. A BI tool holding an open connection
while the daily Airflow batch rebuilds the models produces either a locked warehouse or a failed
dbt run, and that is the usual reason "Superset over DuckDB" is described as fragile.

The fix is not clever, it is procedural: **dashboards read a copy.** `make publish` snapshots the
warehouse to `data/warehouse/fraud_readonly.duckdb` (written beside and renamed, so a reader never
sees a half-written file), and Superset points at the snapshot. The live file stays with the jobs
that write it.

## Connecting

1. `make publish` — create the read-only snapshot.
2. `make dashboard` — start Superset on <http://localhost:8088>.
3. Log in with the credentials in `.env` (`SUPERSET_ADMIN` / `SUPERSET_PASSWORD`).
4. **Data → Databases → + Database → Other**, with this SQLAlchemy URI:

   ```
   duckdb:////app/data/warehouse/fraud_readonly.duckdb
   ```

   Four slashes: three for the URI scheme, one for the absolute path inside the container.

5. **Data → Datasets → + Dataset** for each of:
   `agg_daily_kpis`, `agg_model_performance`, `agg_rule_effectiveness`, `fct_decisions`.

## Charts worth building first

These mirror the Streamlit page, which is the quickest way to check the two agree:

| Chart | Dataset | Notes |
|---|---|---|
| Decisions per day | `agg_daily_kpis` | `allowed` / `sent_to_review` / `blocked` |
| Fraud value caught vs missed | `agg_daily_kpis` | the pair that exposes an expensive-miss model |
| Review queue size | `agg_daily_kpis` | the number that decides whether thresholds are affordable |
| False positive rate | `agg_daily_kpis` | null while labels are pending — do not render null as zero |
| Precision / recall over time | `agg_model_performance` | matured labels only |
| Latency p50 / p95 / p99 | `agg_daily_kpis` | one axis; three series in the same unit |
| Rule hit rate | `agg_rule_effectiveness` | a rule that fires constantly and is rarely right is a tax |

## Status: unverified

The Compose service and this configuration have **not been run**. The environment this repository
was built in blocks Docker Hub image layers, so the Superset image could never be pulled. Every
other component here was executed end to end; this one is written from the documented
configuration and is the most likely thing in the repository to need a fix on first run.

If it does not work, the Streamlit metrics page covers the same ground and the export in
`dashboard/powerbi/` covers the BI-tool case.
