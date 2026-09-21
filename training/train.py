"""Train the fraud model: XGBoost, time-split, cost-tuned thresholds, MLflow.

    make train

The run does five things, in an order that matters:

1. **Split on time**, never at random (``training/split.py``).
2. **Train on the training window**, with ``scale_pos_weight`` to stop the
   model collapsing onto the majority class, and early stopping on the
   validation window's PR-AUC.
3. **Tune the thresholds on the validation window** by expected cost. Tuning
   them on the test window would be choosing the operating point using the data
   the result is then reported on - a quiet way of reporting a number nobody
   can reproduce in production.
4. **Evaluate on the test window**, which the model has never seen and the
   thresholds were not fitted to.
5. **Log everything to MLflow** and register the model, promoting it to
   champion only if it beats the incumbent on the test window.

Every number the README quotes comes from step 4 and lands in
``reports/metrics.json``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.config import REPO_ROOT, get_settings
from training.evaluate import ModelMetrics, evaluate_model
from training.explain import Explainer, global_importance
from training.preprocessing import MODEL_COLUMNS, build_matrix
from training.split import SplitBoundaries, time_based_split
from training.thresholds import CostModel, ThresholdChoice, tune_thresholds

logger = logging.getLogger("train")

LABEL_COLUMN = "is_fraud"
AMOUNT_COLUMN = "amount"

# Deliberately modest: depth 6 on ~50 features, and early stopping decides the
# tree count. A deeper model memorises card tokens in the training window and
# looks better offline than it is.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    # Area under the precision-recall curve: the metric that means something
    # when positives are rare.
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "enable_categorical": True,
}


@dataclass
class TrainingResult:
    """Everything a run produced, ready to be logged or asserted on."""

    model: Any
    thresholds: ThresholdChoice
    test_metrics: ModelMetrics
    validation_metrics: ModelMetrics
    boundaries: SplitBoundaries
    params: dict[str, Any]
    scale_pos_weight: float
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)

    def describe(self) -> str:
        return (
            f"{self.boundaries.describe()}\n"
            f"  thresholds: {self.thresholds.describe()}\n"
            f"  test:       {self.test_metrics.describe()}"
        )


def _labels_and_amounts(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return frame[LABEL_COLUMN].to_numpy().astype(int), frame[AMOUNT_COLUMN].to_numpy(dtype=float)


def train_model(
    training_set: pd.DataFrame,
    costs: CostModel | None = None,
    params: dict[str, Any] | None = None,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    early_stopping_rounds: int = 40,
    explain: bool = True,
) -> TrainingResult:
    """Train, tune thresholds and evaluate - no MLflow, no side effects.

    Keeping this pure is what lets the tests run a full training cycle in
    seconds without a tracking server.
    """
    from xgboost import XGBClassifier

    costs = costs or CostModel()
    params = {**DEFAULT_PARAMS, **(params or {})}

    train, validation, test, boundaries = time_based_split(
        training_set, train_fraction, validation_fraction
    )
    for name, part in (("validation", validation), ("test", test)):
        if part.empty:
            raise ValueError(f"the {name} window is empty - not enough data to train on")

    x_train, x_validation, x_test = (build_matrix(part) for part in (train, validation, test))
    y_train, _ = _labels_and_amounts(train)
    y_validation, validation_amounts = _labels_and_amounts(validation)
    y_test, test_amounts = _labels_and_amounts(test)

    # With ~3.5% positives, an unweighted model can reach a good loss by
    # predicting "legitimate" almost everywhere. Weighting the positive class
    # by the imbalance ratio makes a missed fraud as expensive to the loss
    # function as it is to the business.
    positives = max(int(y_train.sum()), 1)
    scale_pos_weight = float((len(y_train) - positives) / positives)

    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early_stopping_rounds,
        random_state=get_settings().random_seed,
    )
    model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)

    # Thresholds come from validation; the test window stays untouched.
    validation_scores = model.predict_proba(x_validation)[:, 1]
    thresholds = tune_thresholds(y_validation, validation_scores, validation_amounts, costs)

    validation_metrics = evaluate_model(
        y_validation,
        validation_scores,
        validation_amounts,
        thresholds.review_threshold,
        thresholds.block_threshold,
        costs,
    )
    test_metrics = evaluate_model(
        y_test,
        model.predict_proba(x_test)[:, 1],
        test_amounts,
        thresholds.review_threshold,
        thresholds.block_threshold,
        costs,
    )

    importance = pd.DataFrame()
    if explain:
        # Sampled: SHAP over the full test window is slow and the ranking is
        # stable well before then.
        sample = x_test.head(min(len(x_test), 2000))
        importance = global_importance(Explainer(model), sample)

    return TrainingResult(
        model=model,
        thresholds=thresholds,
        test_metrics=test_metrics,
        validation_metrics=validation_metrics,
        boundaries=boundaries,
        params=params,
        scale_pos_weight=scale_pos_weight,
        feature_importance=importance,
    )


def log_to_mlflow(
    result: TrainingResult,
    tracking_uri: str,
    experiment: str,
    registered_model: str | None = None,
    champion_alias: str = "champion",
    promote: bool = True,
) -> dict[str, Any]:
    """Log the run, register the model, and promote it if it earned it.

    The promotion rule is deliberately explicit: a challenger replaces the
    champion only if its PR-AUC on the most recent test window is higher. Phase
    7 runs exactly this from Airflow every week.
    """
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    outcome: dict[str, Any] = {"promoted": False, "version": None}

    with mlflow.start_run() as run:
        mlflow.log_params({**result.params, "scale_pos_weight": result.scale_pos_weight})
        mlflow.log_params(
            {
                "train_rows": result.boundaries.train_rows,
                "validation_rows": result.boundaries.validation_rows,
                "test_rows": result.boundaries.test_rows,
                "train_end": result.boundaries.train_end,
                "validation_end": result.boundaries.validation_end,
            }
        )
        mlflow.log_metrics(
            {
                f"test_{key}": value
                for key, value in result.test_metrics.to_dict().items()
                if isinstance(value, int | float)
            }
        )
        mlflow.log_metrics(
            {
                "validation_pr_auc": result.validation_metrics.pr_auc,
                "review_threshold": result.thresholds.review_threshold,
                "block_threshold": result.thresholds.block_threshold,
                "expected_cost": result.thresholds.expected_cost,
            }
        )
        if not result.feature_importance.empty:
            mlflow.log_table(result.feature_importance, "feature_importance.json")

        example = pd.DataFrame(columns=MODEL_COLUMNS).astype("float64").head(0)
        signature = infer_signature(example, np.array([0.0]))
        mlflow.xgboost.log_model(result.model, artifact_path="model", signature=signature)

        outcome["run_id"] = run.info.run_id
        model_uri = f"runs:/{run.info.run_id}/model"

    if registered_model:
        client = MlflowClient(tracking_uri=tracking_uri)
        version = mlflow.register_model(model_uri, registered_model).version
        outcome["version"] = version

        champion_pr_auc = _champion_pr_auc(client, registered_model, champion_alias)
        challenger_pr_auc = result.test_metrics.pr_auc
        if promote and (champion_pr_auc is None or challenger_pr_auc > champion_pr_auc):
            client.set_registered_model_alias(registered_model, champion_alias, version)
            outcome["promoted"] = True
            logger.info(
                "promoted version %s to %s (PR-AUC %.4f vs %s)",
                version,
                champion_alias,
                challenger_pr_auc,
                "no incumbent" if champion_pr_auc is None else f"{champion_pr_auc:.4f}",
            )
        else:
            logger.info(
                "left version %s as challenger (PR-AUC %.4f vs champion %.4f)",
                version,
                challenger_pr_auc,
                champion_pr_auc,
            )

    return outcome


def _champion_pr_auc(client: Any, registered_model: str, alias: str) -> float | None:
    """The incumbent's PR-AUC, or None when there is no champion yet."""
    try:
        champion = client.get_model_version_by_alias(registered_model, alias)
    except Exception:
        return None

    run = client.get_run(champion.run_id)
    return run.data.metrics.get("test_pr_auc")


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--training-path", default=str(settings.path(settings.delta_training_path)))
    parser.add_argument("--report-dir", default=str(REPO_ROOT / "reports"))
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--no-mlflow", action="store_true", help="train and report without a tracking server"
    )
    parser.add_argument(
        "--no-promote", action="store_true", help="register the model but never set the alias"
    )
    return parser


def load_training_set(training_path: str) -> pd.DataFrame:
    """Read the point-in-time training set into pandas via Spark."""
    from common.spark import build_spark_session

    spark = build_spark_session("load-training-set")
    spark.sparkContext.setLogLevel("WARN")
    frame = spark.read.format("delta").load(training_path).toPandas()
    spark.stop()
    return frame


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    settings = get_settings()

    training_set = load_training_set(args.training_path)
    logger.info(
        "training set: %s rows, %s fraud", len(training_set), int(training_set.is_fraud.sum())
    )

    result = train_model(
        training_set,
        costs=CostModel(false_block=settings.cost_false_block, review=settings.cost_review),
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    print(result.describe())

    # Record the provenance of the numbers alongside them.
    from producer.replay import resolve_source_paths

    _, _, is_real_data = resolve_source_paths()
    result.test_metrics.dataset = "IEEE-CIS (real)" if is_real_data else "synthetic fixture"

    report_dir = Path(args.report_dir)
    metrics_path = result.test_metrics.save(report_dir / "metrics.json")
    thresholds_path = result.thresholds.save(report_dir / "thresholds.json")
    logger.info("wrote %s and %s", metrics_path, thresholds_path)

    if not args.no_mlflow:
        outcome = log_to_mlflow(
            result,
            tracking_uri=settings.mlflow_tracking_uri,
            experiment=settings.mlflow_experiment_name,
            registered_model=settings.mlflow_registered_model,
            champion_alias=settings.mlflow_champion_alias,
            promote=not args.no_promote,
        )
        logger.info("mlflow run %s, version %s", outcome.get("run_id"), outcome.get("version"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
