"""Tests for the training pipeline.

These run a real XGBoost fit - small, but real - because the things most worth
protecting (imbalance handling, thresholds tuned on validation rather than
test, a registry promotion rule) are not visible in a mock.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.training_helpers import synthetic_training_set
from training.preprocessing import MODEL_COLUMNS, build_matrix
from training.thresholds import CostModel
from training.train import train_model

# Small enough to keep the fast suite fast, big enough that the test window
# holds more than a handful of frauds - PR-AUC is a noisy statistic when it is
# computed over six positives.
TRAINING_ROWS = 6000
FAST_PARAMS = {"n_estimators": 60, "max_depth": 4}


@pytest.fixture(scope="module")
def training_set() -> pd.DataFrame:
    return synthetic_training_set(n_rows=TRAINING_ROWS, seed=11)


@pytest.fixture(scope="module")
def trained(training_set: pd.DataFrame):
    return train_model(training_set, params=FAST_PARAMS, early_stopping_rounds=20)


def test_the_model_learns_something(trained) -> None:
    """A sanity floor, not a performance claim.

    The number itself is meaningless - it comes from synthetic data - but a
    model that cannot beat random on data built to contain a signal means the
    pipeline is broken.
    """
    assert trained.test_metrics.pr_auc > 0.3
    assert trained.test_metrics.roc_auc > 0.8
    assert trained.test_metrics.recall_including_review > 0.2


def test_the_split_is_chronological_end_to_end(trained) -> None:
    assert trained.boundaries.train_end < trained.boundaries.validation_end
    assert trained.boundaries.train_rows > trained.boundaries.test_rows


def test_the_imbalance_is_handled(trained) -> None:
    """With ~3.5% positives an unweighted model can ignore fraud entirely."""
    assert trained.scale_pos_weight > 10
    assert trained.model.get_params()["scale_pos_weight"] == trained.scale_pos_weight


def test_thresholds_are_tuned_on_validation_not_test(training_set: pd.DataFrame) -> None:
    """Otherwise the reported result is fitted to the data it is reported on.

    Changing the cost model must move the thresholds; the *test* window plays
    no part in choosing them, so a model tuned under two different cost models
    is still the same model.
    """
    cheap = train_model(
        training_set, costs=CostModel(false_block=5.0, review=1.0), params=FAST_PARAMS
    )
    expensive = train_model(
        training_set, costs=CostModel(false_block=500.0, review=1.0), params=FAST_PARAMS
    )

    assert expensive.thresholds.block_threshold >= cheap.thresholds.block_threshold
    assert expensive.test_metrics.block_rate <= cheap.test_metrics.block_rate


def test_the_review_band_exists(trained) -> None:
    """Three decisions, not two - there has to be room between the thresholds."""
    assert trained.thresholds.review_threshold <= trained.thresholds.block_threshold


def test_the_model_scores_a_single_payment_the_same_way(trained, training_set) -> None:
    """The scoring API will pass one row; the answer must not depend on batching."""
    matrix = build_matrix(training_set.tail(20))

    batch_scores = trained.model.predict_proba(matrix)[:, 1]
    single_score = trained.model.predict_proba(matrix.head(1))[:, 1][0]

    assert single_score == pytest.approx(batch_scores[0], rel=1e-6)


def test_the_model_expects_exactly_the_declared_columns(trained) -> None:
    assert list(trained.model.feature_names_in_) == MODEL_COLUMNS


def test_a_window_with_no_data_is_rejected(training_set: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        train_model(training_set.head(3), params=FAST_PARAMS)


# --- Explanations -----------------------------------------------------------


def test_shap_explains_a_single_decision(trained, training_set: pd.DataFrame) -> None:
    """An analyst cannot act on "0.87"; they need the reasons behind it."""
    from training.explain import Explainer

    explainer = Explainer(trained.model)
    scores = trained.model.predict_proba(build_matrix(training_set))[:, 1]
    riskiest = build_matrix(training_set.iloc[[int(scores.argmax())]])

    reasons = explainer.top_reasons(riskiest, top_n=3)

    assert 0 < len(reasons) <= 3
    assert all(reason.contribution > 0 for reason in reasons)
    assert all(reason.feature in MODEL_COLUMNS for reason in reasons)
    assert "increases risk" in reasons[0].describe()


def test_reasons_are_ranked_by_strength(trained, training_set: pd.DataFrame) -> None:
    from training.explain import Explainer

    explainer = Explainer(trained.model)
    matrix = build_matrix(training_set.tail(1))

    reasons = explainer.top_reasons(matrix, top_n=5)

    contributions = [reason.contribution for reason in reasons]
    assert contributions == sorted(contributions, reverse=True)


def test_explaining_a_batch_is_rejected(trained, training_set: pd.DataFrame) -> None:
    from training.explain import Explainer

    with pytest.raises(ValueError, match="one payment"):
        Explainer(trained.model).top_reasons(build_matrix(training_set.head(5)))


def test_global_importance_names_real_features(trained) -> None:
    assert not trained.feature_importance.empty
    assert set(trained.feature_importance.feature) <= set(MODEL_COLUMNS)
    assert (trained.feature_importance.mean_abs_shap >= 0).all()


# --- MLflow -----------------------------------------------------------------


def test_the_run_is_logged_and_the_model_registered(trained, tmp_path) -> None:
    """A model nobody can find again is not a deliverable.

    SQLite rather than the file store because the model registry needs a
    database behind it - the same reason the Compose service runs with a
    SQLite backend store.
    """
    from mlflow.tracking import MlflowClient

    from training.train import log_to_mlflow

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    outcome = log_to_mlflow(
        trained, tracking_uri=tracking_uri, experiment="test", registered_model="fraud-test"
    )

    assert outcome["version"] is not None
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(outcome["run_id"])
    assert run.data.metrics["test_pr_auc"] == pytest.approx(trained.test_metrics.pr_auc)
    assert run.data.params["scale_pos_weight"]


def test_the_first_model_becomes_champion(trained, tmp_path) -> None:
    from mlflow.tracking import MlflowClient

    from training.train import log_to_mlflow

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    outcome = log_to_mlflow(
        trained, tracking_uri=tracking_uri, experiment="test", registered_model="fraud-test"
    )

    assert outcome["promoted"] is True
    client = MlflowClient(tracking_uri=tracking_uri)
    champion = client.get_model_version_by_alias("fraud-test", "champion")
    assert champion.version == outcome["version"]


def test_a_worse_challenger_does_not_replace_the_champion(trained, tmp_path) -> None:
    """The promotion rule, which phase 7 runs weekly from Airflow."""
    import copy

    from mlflow.tracking import MlflowClient

    from training.train import log_to_mlflow

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    champion = log_to_mlflow(
        trained, tracking_uri=tracking_uri, experiment="test", registered_model="fraud-test"
    )

    weaker = copy.copy(trained)
    weaker.test_metrics = copy.copy(trained.test_metrics)
    weaker.test_metrics.pr_auc = trained.test_metrics.pr_auc - 0.1
    challenger = log_to_mlflow(
        weaker, tracking_uri=tracking_uri, experiment="test", registered_model="fraud-test"
    )

    assert challenger["promoted"] is False
    client = MlflowClient(tracking_uri=tracking_uri)
    assert (
        client.get_model_version_by_alias("fraud-test", "champion").version == champion["version"]
    )
