"""Evaluating the analyst assistant.

An LLM feature without an evaluation is a demo. This measures the four things
that decide whether a fraud analyst can rely on the summaries:

**Factual accuracy** - does every number in the summary appear in the facts?
Checked automatically, by extracting the numbers from the generated text and
matching them against what the pipeline computed. This is the metric that
matters most: a summary that invents "3.2x the usual amount" is worse than no
summary, because it is wrong in a way that reads as authoritative.

**Schema validity** - what share of responses parse and pass the Pydantic
model? A small model asked for JSON will sometimes produce prose.

**Retrieval quality** - of the similar cases retrieved, what share had the same
confirmed outcome as the case being reviewed? If a fraudulent payment retrieves
four legitimate ones, the "similar cases" panel is actively misleading.

**Latency** - per summary. Generation happens while an analyst waits.

    make llm-eval                 # against Ollama
    make llm-eval ARGS="--dry-run"  # harness smoke test, not a measurement

Results go to ``reports/llm_eval.json``, and the report records which generator
and which embedder produced them - so a run made with the stub embedder can
never be mistaken for a measurement of the real thing.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common.config import REPO_ROOT, get_settings
from genai.case_store import CaseStore, connect, describe_case
from genai.embeddings import build_embedder
from genai.facts import CaseFacts
from genai.summarise import (
    ScriptedGenerator,
    SummaryGenerator,
    check_grounding,
    summarise_case,
)
from serving.decisions import Decision, now

logger = logging.getLogger("llm-eval")

DEFAULT_EXPORT_DIR = REPO_ROOT / "data" / "warehouse" / "export"
DEFAULT_REPORT = REPO_ROOT / "reports" / "llm_eval.json"
MINIMUM_CASES = 50


@dataclass
class EvalReport:
    """What the evaluation found, and under what conditions."""

    cases: int
    generator: str
    embedder: str
    schema_valid: int
    grounded: int
    usable: int
    schema_validity: float
    factual_accuracy: float
    retrieval_precision: float
    retrieval_cases_checked: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    repairs_needed: int
    unsupported_number_examples: list[float] = field(default_factory=list)
    invented_case_examples: list[str] = field(default_factory=list)
    note: str = ""

    def describe(self) -> str:
        return (
            f"{self.cases} flagged cases with {self.generator} / {self.embedder}\n"
            f"  schema validity    {self.schema_validity:.1%} "
            f"({self.schema_valid}/{self.cases})\n"
            f"  factual accuracy   {self.factual_accuracy:.1%} "
            f"(every number traced back to the facts)\n"
            f"  retrieval quality  {self.retrieval_precision:.1%} "
            f"of retrieved cases shared the outcome\n"
            f"  latency            p50 {self.latency_p50_ms:.0f} ms, "
            f"p95 {self.latency_p95_ms:.0f} ms\n"
            f"  repairs needed     {self.repairs_needed}"
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2))
        return destination


def load_flagged_cases(export_dir: Path, limit: int) -> list[tuple[Decision, bool]]:
    """Flagged payments whose true outcome is known, oldest first.

    Deterministic: the same export always produces the same evaluation set, so
    two runs are comparable.
    """
    import duckdb

    connection = duckdb.connect()
    rows = connection.execute(
        f"""
        WITH latest AS (
            SELECT *, row_number() OVER (
                PARTITION BY transaction_id ORDER BY decided_at DESC
            ) AS rank
            FROM read_parquet('{export_dir}/decisions/*.parquet')
        )
        SELECT d.transaction_id, d.card_token, d.decided_at, d.decision, d.score,
               d.triggered_by, d.reason, d.rule_name, d.model_version,
               d.review_threshold, d.block_threshold, d.top_reasons, d.features,
               c.is_fraud, s.product_cd, s.device_type
        FROM latest d
        JOIN read_parquet('{export_dir}/chargebacks/*.parquet') c USING (transaction_id)
        LEFT JOIN read_parquet('{export_dir}/silver/*.parquet') s USING (transaction_id)
        WHERE d.rank = 1
          AND d.decision IN ('review', 'block')
          AND c.label_available_at <= current_timestamp
        ORDER BY d.transaction_id
        LIMIT {limit}
        """
    ).fetchall()

    cases = []
    for row in rows:
        cases.append(
            (
                Decision(
                    transaction_id=int(row[0]),
                    card_token=row[1],
                    decided_at=row[2] or now(),
                    decision=row[3],
                    score=row[4],
                    triggered_by=row[5],
                    reason=row[6] or "",
                    rule_name=row[7],
                    model_version=row[8] or "unknown",
                    review_threshold=float(row[9] or 0.0),
                    block_threshold=float(row[10] or 1.0),
                    top_reasons=[dict(reason) for reason in (row[11] or [])],
                    features=dict(row[12] or {}),
                ),
                bool(row[13]),
            )
        )
    return cases


def _dry_run_generator(count: int) -> ScriptedGenerator:
    """A generator that exercises every branch of the harness.

    Two thirds of the responses are well-formed and grounded, one sixth is not
    JSON at all, and one sixth quotes a number that appears nowhere in the
    facts. That is enough to prove the metrics move - it is emphatically not a
    measurement of any model.
    """
    responses: list[str] = []
    for index in range(count * 2):  # x2, because a rejected response is retried
        if index % 6 == 3:
            responses.append("I think this payment is suspicious, but I am not sure.")
        elif index % 6 == 4:
            responses.append(
                json.dumps(
                    {
                        "summary": "This payment is 47.3 times the cardholder's usual amount "
                        "and was made on an unfamiliar device late at night.",
                        "key_risk_signals": ["unusual amount", "new device"],
                        "similar_cases": [],
                        "recommended_action": "investigate_further",
                        "confidence": 0.6,
                    }
                )
            )
        else:
            responses.append(
                json.dumps(
                    {
                        "summary": "The payment was flagged by the scoring model. The facts "
                        "show behaviour that differs from this card's normal pattern, so a "
                        "reviewer should confirm it with the cardholder.",
                        "key_risk_signals": ["score above the review threshold"],
                        "similar_cases": [],
                        "recommended_action": "investigate_further",
                        "confidence": 0.5,
                    }
                )
            )
    return ScriptedGenerator(responses)


def evaluate(
    cases: list[tuple[Decision, bool]],
    store: CaseStore | None,
    generator: SummaryGenerator,
    embedder_name: str,
    top_k: int = 5,
    note: str = "",
) -> EvalReport:
    """Run every case through the assistant and score the results."""
    latencies: list[float] = []
    schema_valid = grounded = usable = repairs = 0
    retrieved_total = retrieved_matching = 0
    unsupported_examples: list[float] = []
    invented_examples: list[str] = []

    for decision, is_fraud in cases:
        similar = []
        if store is not None:
            description = describe_case(
                amount=float(decision.features.get("amount", 0.0)),
                decision=decision.decision,
                features=decision.features,
            )
            similar = store.find_similar(
                description, top_k=top_k, exclude_transaction_id=decision.transaction_id
            )
            retrieved_total += len(similar)
            retrieved_matching += sum(case.is_fraud == is_fraud for case in similar)

        facts = CaseFacts(
            transaction_id=decision.transaction_id,
            card_token=decision.card_token,
            decided_at=decision.decided_at,
            decision=decision.decision,
            score=decision.score,
            triggered_by=decision.triggered_by,
            reason=decision.reason,
            rule_name=decision.rule_name,
            model_version=decision.model_version,
            review_threshold=decision.review_threshold,
            block_threshold=decision.block_threshold,
            amount=float(decision.features.get("amount", 0.0)),
            features=decision.features,
            top_reasons=decision.top_reasons,
            similar_cases=similar,
        )

        result = summarise_case(facts, generator)
        latencies.append(result.latency_ms)
        repairs += max(result.attempts - 1, 0)

        if result.summary is not None:
            schema_valid += 1
            numbers, invented = check_grounding(result.summary, facts)
            if not numbers and not invented:
                grounded += 1
            unsupported_examples.extend(numbers[:2])
            invented_examples.extend(invented[:2])
        if result.usable:
            usable += 1

    total = max(len(cases), 1)
    return EvalReport(
        cases=len(cases),
        generator=generator.name,
        embedder=embedder_name,
        schema_valid=schema_valid,
        grounded=grounded,
        usable=usable,
        schema_validity=schema_valid / total,
        # Of everything that parsed, how much was traceable to the facts.
        factual_accuracy=grounded / schema_valid if schema_valid else 0.0,
        retrieval_precision=retrieved_matching / retrieved_total if retrieved_total else 0.0,
        retrieval_cases_checked=retrieved_total,
        latency_p50_ms=statistics.median(latencies) if latencies else 0.0,
        latency_p95_ms=(statistics.quantiles(latencies, n=20)[-1] if len(latencies) > 1 else 0.0),
        latency_mean_ms=statistics.fmean(latencies) if latencies else 0.0,
        repairs_needed=repairs,
        unsupported_number_examples=sorted(set(unsupported_examples))[:10],
        invented_case_examples=sorted(set(invented_examples))[:10],
        note=note,
    )


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--cases", type=int, default=MINIMUM_CASES)
    parser.add_argument("--top-k", type=int, default=settings.rag_top_k)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise the harness with scripted responses instead of a model",
    )
    parser.add_argument("--allow-stub-embedder", action="store_true")
    parser.add_argument("--no-retrieval", action="store_true", help="skip the case store")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    settings = get_settings()

    cases = load_flagged_cases(Path(args.export_dir), args.cases)
    if len(cases) < MINIMUM_CASES:
        logger.warning(
            "only %s flagged cases with confirmed labels (the evaluation set is meant to be "
            "at least %s) - run more payments through the pipeline",
            len(cases),
            MINIMUM_CASES,
        )
    if not cases:
        logger.error("no flagged cases found in %s", args.export_dir)
        return 1

    store = None
    embedder_name = "none"
    if not args.no_retrieval:
        embedder = build_embedder(settings.embedding_model, allow_stub=args.allow_stub_embedder)
        embedder_name = embedder.name
        store = CaseStore(
            connect(
                settings.pgvector_host,
                settings.pgvector_port,
                settings.pgvector_db,
                settings.pgvector_user,
                settings.pgvector_password,
            ),
            embedder,
        )

    note = ""
    if args.dry_run:
        generator: SummaryGenerator = _dry_run_generator(len(cases))
        note = (
            "DRY RUN: responses are scripted to exercise the harness. These numbers "
            "measure the evaluation code, not any model."
        )
    else:
        from genai.summarise import OllamaGenerator

        generator = OllamaGenerator(settings.ollama_base_url, settings.ollama_model)

    if embedder_name == "hashing-stub":
        note = (note + " " if note else "") + (
            "Retrieval used the deterministic stub embedder, so retrieval quality "
            "here is not a measurement of semantic search."
        )

    report = evaluate(cases, store, generator, embedder_name, args.top_k, note)
    print(report.describe())
    if report.note:
        print(f"\n  NOTE: {report.note}")
    print(f"\n  report: {report.save(args.report)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
