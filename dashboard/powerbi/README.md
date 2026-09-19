# Power BI (optional)

The project's dashboards are built with Superset and Streamlit so that a reviewer can run them for
free on any operating system (the reasoning is in
[docs/design_decisions.md](../../docs/design_decisions.md)). Power BI is supported here as an
optional extra for anyone who wants to build the same views in a tool they already know.

## Connecting

Power BI Desktop has no DuckDB connector, so the pipeline exports the Gold tables to Parquet, which
Power BI reads natively with no driver:

1. Run the batch pipeline so the Gold layer exists (phase 7).
2. Export to Parquet — `make export-powerbi` (arrives in phase 9); output lands in
   `data/exports/powerbi/`.
3. In Power BI Desktop: **Get Data → Parquet**, point at the exported files.

## What goes in this folder

- `fraud_dashboard.pbix` — your report file (not committed by default; `data/` and large binaries
  stay out of Git, but a `.pbix` here is small enough to commit if you want it reviewed)
- `screenshot.png` — an image for the README

Both are placeholders for now.
