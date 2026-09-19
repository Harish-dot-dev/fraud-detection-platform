"""Great Expectations suite for the Silver layer.

The checks here are the contract between ingestion and everything downstream.
They are deliberately blunt: a Silver table that fails any of them is not worth
building features from, so ``streaming/silver.py`` refuses to write it.

The single most important expectation is the column set. Declaring it exactly
means the suite fails if a raw identifier such as ``card1`` ever leaks back
into Silver - a data-protection regression that no amount of value-level
checking would catch.

Expectations are declared as data rather than as a chain of validator calls,
so the whole contract can be read in one place and the runner stays generic.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

logger = logging.getLogger("quality")

# The exact Silver schema. Anything missing, and anything extra, fails.
SILVER_COLUMNS = [
    "transaction_id",
    "event_time",
    "event_date",
    "transaction_dt",
    "card_token",
    "amount",
    "product_cd",
    "card3",
    "card4",
    "card5",
    "card6",
    "dist1",
    "dist2",
    "p_emaildomain",
    "r_emaildomain",
    "counts",
    "deltas",
    "match_flags",
    "vesta",
    "identity_numeric",
    "identity_categorical",
    "device_type",
    "device_info",
    "ingested_at",
]

# A payment above this is not plausible in this dataset (the maximum in the
# real data is about 32,000) and almost certainly indicates a units bug.
MAX_PLAUSIBLE_AMOUNT = 100_000.0

SILVER_EXPECTATIONS: list[tuple[str, dict[str, Any]]] = [
    # --- Structure -----------------------------------------------------------
    ("expect_table_row_count_to_be_between", {"min_value": 1}),
    # The schema contract, and the check that keeps raw card identifiers out.
    (
        "expect_table_columns_to_match_set",
        {"column_set": SILVER_COLUMNS, "exact_match": True},
    ),
    # --- Keys ----------------------------------------------------------------
    ("expect_column_values_to_not_be_null", {"column": "transaction_id"}),
    # Duplicate payments would corrupt every velocity feature downstream.
    ("expect_column_values_to_be_unique", {"column": "transaction_id"}),
    ("expect_column_values_to_not_be_null", {"column": "event_time"}),
    # --- Tokenisation --------------------------------------------------------
    ("expect_column_values_to_not_be_null", {"column": "card_token"}),
    # 32 hex characters: proof the tokeniser ran, not just that a value exists.
    (
        "expect_column_values_to_match_regex",
        {"column": "card_token", "regex": "^[0-9a-f]{32}$"},
    ),
    # --- Values --------------------------------------------------------------
    ("expect_column_values_to_not_be_null", {"column": "amount"}),
    (
        "expect_column_values_to_be_between",
        {"column": "amount", "min_value": 0.0001, "max_value": MAX_PLAUSIBLE_AMOUNT},
    ),
    (
        "expect_column_values_to_be_in_set",
        {"column": "product_cd", "value_set": ["W", "C", "R", "H", "S"], "mostly": 0.99},
    ),
    (
        "expect_column_values_to_be_in_set",
        {
            "column": "card6",
            "value_set": ["debit", "credit", "charge card", "debit or credit"],
            "mostly": 0.99,
        },
    ),
    (
        "expect_column_values_to_be_in_set",
        {"column": "device_type", "value_set": ["desktop", "mobile"], "mostly": 0.99},
    ),
]


@dataclass
class QualityResult:
    """The outcome of running a suite, in a form worth logging and storing."""

    success: bool
    evaluated: int
    successful: int
    failures: list[dict[str, Any]] = field(default_factory=list)
    suite_name: str = ""

    def describe(self) -> str:
        return (
            f"{self.suite_name}: {self.successful}/{self.evaluated} expectations passed"
            f"{'' if self.success else ' - FAILED'}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite_name,
            "success": self.success,
            "expectations_evaluated": self.evaluated,
            "expectations_passed": self.successful,
            "failures": self.failures,
        }

    def save(self, path: str | Path) -> Path:
        """Write the result to JSON so a run leaves evidence behind."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return destination


def _quiet_progress_bars(context: Any) -> None:
    """Great Expectations prints a progress bar per metric; in a log it is noise."""
    try:
        from great_expectations.data_context.types.base import ProgressBarsConfig

        context.variables.progress_bars = ProgressBarsConfig(
            globally=False, metric_calculations=False
        )
    except Exception:  # pragma: no cover - cosmetic only
        logger.debug("could not disable Great Expectations progress bars")


def validate(
    dataframe: DataFrame,
    expectations: list[tuple[str, dict[str, Any]]],
    suite_name: str,
) -> QualityResult:
    """Run a declared set of expectations against a Spark DataFrame."""
    import great_expectations as gx

    context = gx.get_context(mode="ephemeral")
    _quiet_progress_bars(context)

    datasource = context.sources.add_or_update_spark(f"{suite_name}_source")
    asset = datasource.add_dataframe_asset(suite_name)
    batch_request = asset.build_batch_request(dataframe=dataframe)
    suite = context.add_or_update_expectation_suite(f"{suite_name}_suite")
    validator = context.get_validator(batch_request=batch_request, expectation_suite=suite)

    for expectation_type, kwargs in expectations:
        getattr(validator, expectation_type)(**kwargs)

    validation = validator.validate()
    statistics = validation.statistics

    failures = [
        {
            "expectation": result.expectation_config.expectation_type,
            "column": result.expectation_config.kwargs.get("column"),
            "observed": result.result.get("observed_value")
            or result.result.get("partial_unexpected_list"),
            "unexpected_count": result.result.get("unexpected_count"),
        }
        for result in validation.results
        if not result.success
    ]

    return QualityResult(
        success=bool(validation.success),
        evaluated=int(statistics["evaluated_expectations"]),
        successful=int(statistics["successful_expectations"]),
        failures=failures,
        suite_name=suite_name,
    )


def validate_silver(dataframe: DataFrame) -> QualityResult:
    """Run the Silver suite."""
    return validate(dataframe, SILVER_EXPECTATIONS, suite_name="silver")


def main(argv: list[str] | None = None) -> int:
    """Validate the Silver table on disk and write the result to reports/."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from common.config import REPO_ROOT, get_settings
    from common.spark import build_spark_session

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--silver-path", default=str(settings.path(settings.delta_silver_path)))
    parser.add_argument("--report", default=str(REPO_ROOT / "reports" / "quality_silver.json"))
    args = parser.parse_args(argv)

    spark = build_spark_session("silver-quality")
    spark.sparkContext.setLogLevel("ERROR")

    result = validate_silver(spark.read.format("delta").load(args.silver_path))
    print(result.describe())
    for failure in result.failures:
        print(f"  FAILED {failure['expectation']} on {failure['column']}: {failure['observed']}")
    print(f"  report: {result.save(args.report)}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
