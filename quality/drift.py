"""Drift monitoring with Evidently.

A fraud model decays in two different ways, and they need watching separately:

* **Feature drift** - the payments themselves change. A new merchant category,
  a marketing push into a new country, a payment method that suddenly takes
  off. The model is now being asked about a population it was not trained on.
* **Prediction drift** - the distribution of scores moves. This is the earlier
  warning of the two, because it shows up without waiting for labels, and
  labels are weeks away (``training/chargebacks.py``).

Neither one means the model is wrong. Both mean somebody should look, which is
why this produces a report rather than an alert that blocks a pipeline.

    make drift

Writes an HTML report a human can read and a JSON summary Airflow can branch
on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from common.config import REPO_ROOT

logger = logging.getLogger("drift")

DEFAULT_EXPORT_DIR = REPO_ROOT / "data" / "warehouse" / "export"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"

# Drift on a handful of noisy columns is normal; this is the share of columns
# that have to drift before the run is called drifted overall.
DRIFT_SHARE_THRESHOLD = 0.3


@dataclass
class DriftResult:
    """What the drift run found."""

    reference_rows: int
    current_rows: int
    columns_checked: int
    columns_drifted: int
    drift_share: float
    dataset_drift: bool
    drifted_columns: list[str] = field(default_factory=list)
    prediction_drift: float | None = None
    html_report: str = ""

    def describe(self) -> str:
        headline = "DRIFT DETECTED" if self.dataset_drift else "no dataset drift"
        return (
            f"{headline}: {self.columns_drifted}/{self.columns_checked} columns "
            f"({self.drift_share:.1%}) over {self.current_rows} recent payments "
            f"vs {self.reference_rows} reference"
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2))
        return destination


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    html_path: str | Path | None = None,
    drift_share_threshold: float = DRIFT_SHARE_THRESHOLD,
) -> DriftResult:
    """Compare two windows of payments and report what moved.

    Args:
        reference: the window the model was trained on.
        current: recent payments.
        html_path: where to write the human-readable report.
    """
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    shared = [column for column in reference.columns if column in current.columns]
    if not shared:
        raise ValueError("reference and current share no columns")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[shared], current_data=current[shared])

    # DataDriftPreset emits two metrics: DatasetDriftMetric carries the summary
    # counts, DataDriftTable carries the per-column detail. They are looked up
    # by name rather than by position - an earlier version read index 0 for
    # both and reported "0 of 18 columns drifted (66.7%)", which is nonsense on
    # its face and exactly what a monitoring tool must never say.
    results = {metric["metric"]: metric["result"] for metric in report.as_dict()["metrics"]}
    summary = results.get("DatasetDriftMetric", {})
    table = results.get("DataDriftTable", {})

    by_column = table.get("drift_by_columns", {})
    drifted = sorted(name for name, details in by_column.items() if details.get("drift_detected"))
    checked = int(summary.get("number_of_columns") or len(shared))
    share = float(summary.get("share_of_drifted_columns") or 0.0)

    written = ""
    if html_path:
        destination = Path(html_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(destination))
        written = str(destination)

    if by_column and len(drifted) != int(summary.get("number_of_drifted_columns") or len(drifted)):
        logger.warning(
            "drift summary and per-column detail disagree (%s vs %s)",
            summary.get("number_of_drifted_columns"),
            len(drifted),
        )

    return DriftResult(
        reference_rows=len(reference),
        current_rows=len(current),
        columns_checked=checked,
        columns_drifted=len(drifted),
        drift_share=share,
        dataset_drift=share >= drift_share_threshold,
        drifted_columns=drifted,
        html_report=written,
    )


def split_by_time(
    frame: pd.DataFrame, time_column: str = "event_time", reference_fraction: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a table into an older reference window and a recent one.

    Time-ordered, for the same reason the training split is: comparing a random
    half against another random half would find no drift by construction.
    """
    ordered = frame.sort_values(time_column, kind="stable")
    cut = int(len(ordered) * reference_fraction)
    return ordered.iloc[:cut], ordered.iloc[cut:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument(
        "--reference-fraction",
        type=float,
        default=0.7,
        help="share of the history treated as the reference window",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    import duckdb

    from features.definitions import FEATURE_NAMES

    export_dir = Path(args.export_dir)
    gold_glob = str(export_dir / "gold" / "*.parquet")
    connection = duckdb.connect()

    columns = ", ".join(["event_time", *FEATURE_NAMES])
    gold = connection.execute(f"SELECT {columns} FROM read_parquet('{gold_glob}')").df()
    if gold.empty:
        logger.error("no rows in %s - run `make export` first", gold_glob)
        return 1

    reference, current = split_by_time(gold, reference_fraction=args.reference_fraction)
    result = build_drift_report(
        reference.drop(columns=["event_time"]),
        current.drop(columns=["event_time"]),
        html_path=Path(args.report_dir) / "drift_features.html",
    )

    # Prediction drift: the score distribution, which moves before labels do.
    decisions_glob = str(export_dir / "decisions" / "*.parquet")
    try:
        scores = connection.execute(
            f"SELECT decided_at, score FROM read_parquet('{decisions_glob}') "
            "WHERE score IS NOT NULL"
        ).df()
        if len(scores) > 20:
            score_reference, score_current = split_by_time(scores, "decided_at")
            prediction = build_drift_report(
                score_reference[["score"]],
                score_current[["score"]],
                html_path=Path(args.report_dir) / "drift_predictions.html",
            )
            result.prediction_drift = prediction.drift_share
    except Exception as error:  # noqa: BLE001 - decisions may not exist yet
        logger.info("no prediction drift computed (%s)", error)

    print(result.describe())
    if result.drifted_columns:
        print(f"  drifted: {', '.join(result.drifted_columns[:10])}")
    print(f"  report:  {result.html_report}")
    print(f"  summary: {result.save(Path(args.report_dir) / 'drift.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
