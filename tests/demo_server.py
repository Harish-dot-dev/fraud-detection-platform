"""Run the scoring API without Docker.

Useful in two places: the tests, and measuring latency on a machine with no
Redis, no Kafka and no MLflow. It wires the real ``ScoringService`` to a
fake Redis and a model trained on the synthetic fixture:

    python -m tests.demo_server --port 8000

**A latency number measured against this is not the docker-compose number.**
There is no network hop to Redis and no broker behind the audit sink, so it
measures the Python scoring path and nothing else. Any report produced from it
has to say so.
"""

from __future__ import annotations

import argparse
import logging
import sys

from features.definitions import update_state
from features.store import FeatureStore
from serving.decisions import ListDecisionSink
from serving.model import ModelBundle
from serving.rules import RulesEngine
from serving.scoring import ScoringService
from tests.spark_helpers import events_from_fixture
from tests.training_helpers import synthetic_training_set
from training.train import train_model

logger = logging.getLogger("demo-server")

SALT = "demo" * 16


def build_demo_bundle(rows: int = 4000, seed: int = 11) -> ModelBundle:
    """Train a small model and wrap it the way the registry loader would."""
    result = train_model(
        synthetic_training_set(n_rows=rows, seed=seed),
        params={"n_estimators": 80, "max_depth": 4},
        early_stopping_rounds=20,
    )
    from serving.model import _limit_threads
    from training.explain import Explainer

    _limit_threads(result.model)
    return ModelBundle(
        model=result.model,
        version="demo",
        review_threshold=result.thresholds.review_threshold,
        block_threshold=result.thresholds.block_threshold,
        source="demo (trained in-process on synthetic data)",
        explainer=Explainer(result.model),
    )


def build_demo_service(
    bundle: ModelBundle | None = None, warm_cards: bool = True
) -> ScoringService:
    """A fully wired scoring service backed by fakeredis."""
    import fakeredis
    import pandas as pd

    from tests.conftest import IDENTITY_FIXTURE, TRANSACTION_FIXTURE

    store = FeatureStore(fakeredis.FakeStrictRedis(decode_responses=True), salt=SALT)

    if warm_cards:
        # Give the cards some history, so the velocity features are not all
        # "first payment on this card".
        transactions = pd.read_csv(TRANSACTION_FIXTURE)
        identity = pd.read_csv(IDENTITY_FIXTURE)
        states: dict = {}
        for event in events_from_fixture(transactions, identity):
            from features.definitions import CardState

            states[event.card_key] = update_state(states.get(event.card_key, CardState()), event)
        store.set_many(states.items())

    return ScoringService(
        store=store,
        rules=RulesEngine.from_file(),
        bundle=bundle or build_demo_bundle(),
        sink=ListDecisionSink(),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import uvicorn

    import serving.app as app_module

    logger.info("training the demo model...")
    service = build_demo_service()
    # Install the service directly: the app's lifespan would try to reach a
    # real Redis and a real registry.
    app_module._service = service
    logger.info(
        "demo model ready (review >= %.3f, block >= %.3f)",
        service.bundle.review_threshold,
        service.bundle.block_threshold,
    )

    uvicorn.run(app_module.app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
