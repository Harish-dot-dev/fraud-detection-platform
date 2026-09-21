#!/usr/bin/env python3
"""Build the README results table from the files a run produced.

    make readme-metrics

This exists to enforce one of the project's rules: **no number reaches the
README that did not come out of a measurement.** The table is generated from
whatever is in ``reports/``; anything missing is printed as *not yet measured*
rather than filled in from memory or from a previous run's recollection.

Each row also carries the conditions it was measured under, where the report
recorded them - a latency figure without its concurrency, or an LLM score
without knowing which model produced it, is not a measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS = REPO_ROOT / "reports"
START_MARKER = "<!-- METRICS:START -->"
END_MARKER = "<!-- METRICS:END -->"

NOT_MEASURED = "*not yet measured*"
NOT_ON_REAL_DATA = "*not yet measured on the real dataset*"

# Model quality is entirely data-dependent. A PR-AUC from the synthetic fixture
# says nothing about how this model would perform on real payments, and putting
# one in a results table - however footnoted - invites exactly the wrong
# reading. Those rows stay empty until the run used the real dataset; the
# fixture figures live in PROGRESS.md with the context that makes them
# meaningful.
REAL_DATASET = "IEEE-CIS (real)"


def _load(name: str) -> dict[str, Any] | None:
    path = REPORTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def _percent(value: Any) -> str:
    return NOT_MEASURED if value is None else f"{float(value):.1%}"


def _number(value: Any, places: int = 4) -> str:
    return NOT_MEASURED if value is None else f"{float(value):.{places}f}"


def _money(value: Any) -> str:
    return NOT_MEASURED if value is None else f"{float(value):,.0f}"


def build_rows() -> list[tuple[str, str, str]]:
    """(metric, value, source) for every published number."""
    metrics = _load("metrics.json")
    thresholds = _load("thresholds.json")
    latency = _load("latency.json")
    llm = _load("llm_eval.json")
    drift = _load("drift.json")

    rows: list[tuple[str, str, str]] = []

    # --- Model quality ------------------------------------------------------
    # Withheld unless the run used real payments (see REAL_DATASET).
    on_real_data = bool(metrics) and metrics.get("dataset") == REAL_DATASET
    if not on_real_data:
        # The thresholds were tuned on the same run, so they are withheld with
        # it: an operating point chosen against synthetic costs is not the one
        # this model would use.
        metrics = thresholds = None
    rows.append(
        (
            "PR-AUC (test window)",
            _number(metrics.get("pr_auc")) if metrics else NOT_ON_REAL_DATA,
            "`make train`",
        )
    )
    rows.append(
        (
            "Precision at the block threshold",
            _number(metrics.get("precision_at_block"), 3) if metrics else NOT_ON_REAL_DATA,
            "`make train`",
        )
    )
    rows.append(
        (
            "Recall at the block threshold",
            _number(metrics.get("recall_at_block"), 3) if metrics else NOT_ON_REAL_DATA,
            "`make train`",
        )
    )
    rows.append(
        (
            "Recall including the review queue",
            _number(metrics.get("recall_including_review"), 3) if metrics else NOT_ON_REAL_DATA,
            "`make train`",
        )
    )
    rows.append(
        (
            "False positive rate",
            _number(metrics.get("false_positive_rate"), 5) if metrics else NOT_ON_REAL_DATA,
            "`make train`",
        )
    )

    # --- Money --------------------------------------------------------------
    if metrics:
        caught = metrics.get("fraud_value_blocked", 0) + metrics.get("fraud_value_reviewed", 0)
        value = (
            f"{_money(caught)} caught / {_money(metrics.get('fraud_value_missed'))} missed "
            f"({_percent(metrics.get('fraud_value_caught_share'))} of value)"
        )
    else:
        value = NOT_ON_REAL_DATA
    rows.append(("Fraud value caught vs missed (test window)", value, "`make train`"))

    # --- Operating point ----------------------------------------------------
    if thresholds:
        value = (
            f"review >= {thresholds['review_threshold']:.3f}, "
            f"block >= {thresholds['block_threshold']:.3f} "
            f"({_percent(thresholds.get('cost_reduction'))} lower expected cost than allowing all)"
        )
    else:
        value = NOT_ON_REAL_DATA
    rows.append(("Chosen thresholds (cost-tuned)", value, "`make train`"))

    # --- Latency ------------------------------------------------------------
    if latency and latency.get("service_latency"):
        service = latency["service_latency"]
        value = (
            f"{service['p50_ms']:.1f} / {service['p95_ms']:.1f} / {service['p99_ms']:.1f} ms "
            f"at concurrency {service['concurrency']}, "
            f"peak {latency['peak_throughput_rps']:.0f} req/s"
        )
        source = f"`make load-test` — {latency.get('label', 'load test')}"
    else:
        value, source = NOT_MEASURED, "`make load-test`"
    rows.append(("/score latency p50 / p95 / p99", value, source))

    # --- The assistant ------------------------------------------------------
    # Same rule as model quality. A scripted generator measures the evaluation
    # harness, and the hashing stub measures the retrieval plumbing; neither
    # says anything about llama3.2 or all-MiniLM, and "100% factual accuracy"
    # with a footnote is still read as a result. They are withheld until a real
    # model produced them - the harness figures are in PROGRESS.md.
    generator = (llm or {}).get("generator", "")
    embedder = (llm or {}).get("embedder", "")
    real_generator = bool(llm) and generator not in {"scripted", "unknown", ""}
    real_embedder = bool(llm) and embedder not in {"hashing-stub", "none", "unknown", ""}

    needs_model = "*not yet measured — needs a local LLM*"
    needs_embedder = "*not yet measured — needs the real embedder*"

    rows.append(
        (
            "LLM factual accuracy",
            _percent(llm["factual_accuracy"]) if real_generator else needs_model,
            f"`make llm-eval`{f' — {generator}' if real_generator else ''}",
        )
    )
    rows.append(
        (
            "LLM schema validity",
            _percent(llm["schema_validity"]) if real_generator else needs_model,
            f"`make llm-eval`{f' — {generator}' if real_generator else ''}",
        )
    )
    rows.append(
        (
            "Retrieval quality (same confirmed outcome)",
            _percent(llm["retrieval_precision"]) if real_embedder else needs_embedder,
            f"`make llm-eval`{f' — {embedder}' if real_embedder else ''}",
        )
    )
    rows.append(
        (
            "LLM latency per summary (p50)",
            f"{llm['latency_p50_ms']:.0f} ms" if real_generator else needs_model,
            f"`make llm-eval`{f' — {generator}' if real_generator else ''}",
        )
    )

    # --- Drift --------------------------------------------------------------
    if drift:
        value = (
            f"{drift['columns_drifted']}/{drift['columns_checked']} features "
            f"({_percent(drift.get('drift_share'))})"
        )
    else:
        value = NOT_MEASURED
    rows.append(("Feature drift, recent vs reference window", value, "`make drift`"))

    return rows


def render_table(rows: list[tuple[str, str, str]]) -> str:
    lines = ["| Metric | Value | Produced by |", "|---|---|---|"]
    lines.extend(f"| {metric} | {value} | {source} |" for metric, value, source in rows)
    return "\n".join(lines)


def update_readme(readme: Path, table: str) -> bool:
    """Replace the table between the markers. Returns True if it changed."""
    text = readme.read_text()
    if START_MARKER not in text or END_MARKER not in text:
        raise ValueError(f"{readme} has no {START_MARKER} / {END_MARKER} block")

    before, rest = text.split(START_MARKER, 1)
    _, after = rest.split(END_MARKER, 1)
    updated = f"{before}{START_MARKER}\n\n{table}\n\n{END_MARKER}{after}"

    if updated == text:
        return False
    readme.write_text(updated)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--readme", default=str(REPO_ROOT / "README.md"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="print the table without writing it (used to see what a run produced)",
    )
    args = parser.parse_args(argv)

    rows = build_rows()
    table = render_table(rows)
    measured = sum(1 for _, value, _ in rows if not value.startswith("*not yet measured"))

    print(table)
    print(f"\n{measured} of {len(rows)} metrics measured.")

    if not args.check:
        changed = update_readme(Path(args.readme), table)
        print("README updated." if changed else "README already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
