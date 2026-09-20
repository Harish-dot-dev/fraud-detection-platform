"""Loading the model the API scores with.

The API does not read a file from disk that somebody remembered to copy there.
It asks the MLflow registry for whatever currently carries the ``champion``
alias, which means promoting a model is a registry operation rather than a
deployment - and the audit log can always say exactly which version made a
decision.

The thresholds travel with the model. They were tuned on that model's own
validation window (``training/train.py``), so pairing version 7's model with
version 4's thresholds would be an operating point nobody ever evaluated.

If the registry is unreachable, or there is no champion yet, the service starts
anyway in **degraded mode**: rules still run, the model does not. A fraud
platform that refuses to answer is worse than one that answers conservatively,
and the degradation is explicit in every response and every audit record rather
than hidden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger("serving.model")


@dataclass
class ModelBundle:
    """A scoring model plus everything needed to use it correctly."""

    model: Any
    version: str
    review_threshold: float
    block_threshold: float
    source: str
    explainer: Any = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, matrix: pd.DataFrame) -> float:
        """Fraud probability for a single prepared row."""
        return float(self.model.predict_proba(matrix)[:, 1][0])


def degraded_bundle(review_threshold: float, block_threshold: float) -> ModelBundle:
    """A bundle with no model: rules-only scoring."""
    return ModelBundle(
        model=None,
        version="none",
        review_threshold=review_threshold,
        block_threshold=block_threshold,
        source="degraded",
    )


def load_champion(
    tracking_uri: str,
    registered_model: str,
    alias: str = "champion",
    fallback_review_threshold: float = 0.3,
    fallback_block_threshold: float = 0.8,
    build_explainer: bool = True,
) -> ModelBundle:
    """Load the champion model from the MLflow registry.

    Never raises: a registry problem degrades the service rather than taking it
    down. The caller sees ``source == "degraded"`` and can alert on it.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        version = client.get_model_version_by_alias(registered_model, alias)
        model = mlflow.xgboost.load_model(f"models:/{registered_model}@{alias}")

        # Thresholds come from the run that produced this model version.
        metrics = client.get_run(version.run_id).data.metrics
        review = float(metrics.get("review_threshold", fallback_review_threshold))
        block = float(metrics.get("block_threshold", fallback_block_threshold))

        _limit_threads(model)

        bundle = ModelBundle(
            model=model,
            version=str(version.version),
            review_threshold=review,
            block_threshold=block,
            source=f"mlflow:{registered_model}@{alias}",
        )
        if build_explainer:
            bundle.explainer = _build_explainer(model)

        logger.info(
            "loaded %s version %s (review >= %.3f, block >= %.3f)",
            registered_model,
            version.version,
            review,
            block,
        )
        return bundle

    except Exception as error:  # noqa: BLE001 - any failure means degraded mode
        logger.warning(
            "could not load %s@%s from %s (%s); serving rules only",
            registered_model,
            alias,
            tracking_uri,
            error,
        )
        return degraded_bundle(fallback_review_threshold, fallback_block_threshold)


def _limit_threads(model: Any) -> None:
    """Score with one thread.

    A single-row prediction gains nothing from parallelism, and the default -
    one thread per core, per request - collapses under concurrency: eight
    in-flight requests on four cores produced 32 threads competing for them,
    and p50 latency went from 36 ms to 526 ms. Measured, not guessed.
    """
    try:
        model.set_params(n_jobs=1)
    except Exception:  # pragma: no cover - not every model object supports it
        logger.debug("could not pin the model to a single thread")


def _build_explainer(model: Any) -> Any:
    """SHAP explainer, or None if it cannot be built.

    Explanations are a nice-to-have on the scoring path: losing them should
    cost reasons in the audit log, not the ability to score payments.
    """
    try:
        from training.explain import Explainer

        return Explainer(model)
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "could not build the SHAP explainer (%s); decisions will have no reasons", error
        )
        return None
