#!/usr/bin/env python3
"""Measure /score latency and throughput.

    make load-test
    make load-test ARGS="--requests 5000 --concurrency 16"

Sends payments to a running API and reports p50, p95, p99 and throughput, then
writes the result to ``reports/latency.json``. The README quotes that file and
nothing else.

Percentiles, not averages: a mean hides the tail, and the tail is what a
customer waiting at a checkout actually experiences. p99 is the number worth
arguing about.

By default it sweeps several concurrency levels, because a single latency
figure is close to meaningless on its own. At concurrency 1 you measure how
long the service takes; above saturation you are measuring the queue in front
of it, and the two get quoted interchangeably far too often.

Every run records *what* was measured - the URL, the concurrency, the decision
mix - because a latency figure without its conditions is not a measurement.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.events import payment_event_from_row  # noqa: E402
from common.timeline import DEFAULT_EPOCH  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "train_transaction_sample.csv"


@dataclass
class LatencyReport:
    """What a load test measured, and under what conditions."""

    label: str
    api_url: str
    requests: int
    concurrency: int
    successes: int
    failures: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    throughput_rps: float
    duration_seconds: float
    decision_mix: dict[str, int] = field(default_factory=dict)
    note: str = ""

    def describe(self) -> str:
        return (
            f"{self.successes}/{self.requests} scored in {self.duration_seconds:.1f}s "
            f"({self.throughput_rps:.0f} req/s, concurrency {self.concurrency})\n"
            f"  p50 {self.p50_ms:.1f} ms | p95 {self.p95_ms:.1f} ms | "
            f"p99 {self.p99_ms:.1f} ms | max {self.max_ms:.1f} ms"
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2))
        return destination


def load_payloads(count: int, source: Path = FIXTURE) -> list[str]:
    """Build request bodies from the fixture, cycling if more are needed."""
    frame = pd.read_csv(source)
    payloads = [
        payment_event_from_row(row, epoch=DEFAULT_EPOCH).model_dump_json()
        for row in frame.to_dict(orient="records")
    ]
    if not payloads:
        raise ValueError(f"no payments in {source}")
    return [payloads[i % len(payloads)] for i in range(count)]


def run_load_test(
    api_url: str,
    payloads: list[str],
    concurrency: int = 8,
    label: str = "",
    note: str = "",
    timeout: float = 10.0,
) -> LatencyReport:
    """Fire the payloads at the API and measure each round trip."""
    latencies: list[float] = []
    decisions: Counter[str] = Counter()
    failures = 0

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    headers = {"content-type": "application/json"}

    with httpx.Client(timeout=timeout, limits=limits) as client:

        def send(payload: str) -> None:
            nonlocal failures
            started = time.perf_counter()
            try:
                response = client.post(f"{api_url}/score", content=payload, headers=headers)
                response.raise_for_status()
                # Measured client-side: this is the round trip a caller sees,
                # not the server's own view of its work.
                latencies.append((time.perf_counter() - started) * 1000.0)
                decisions[response.json()["decision"]] += 1
            except (httpx.HTTPError, KeyError, ValueError):
                failures += 1

        started_at = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(send, payloads))
        duration = time.perf_counter() - started_at

    if not latencies:
        raise RuntimeError(f"every request to {api_url} failed - is the API running?")

    values = np.array(latencies)
    return LatencyReport(
        label=label or "load test",
        api_url=api_url,
        requests=len(payloads),
        concurrency=concurrency,
        successes=len(latencies),
        failures=failures,
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        mean_ms=float(statistics.fmean(latencies)),
        max_ms=float(values.max()),
        throughput_rps=len(latencies) / duration if duration else 0.0,
        duration_seconds=duration,
        decision_mix=dict(decisions),
        note=note,
    )


@dataclass
class SweepReport:
    """A set of runs at different concurrency levels."""

    label: str
    api_url: str
    note: str
    runs: list[LatencyReport]

    @property
    def service_latency(self) -> LatencyReport:
        """The lowest-concurrency run: the service's own latency, unqueued."""
        return min(self.runs, key=lambda run: run.concurrency)

    @property
    def peak_throughput_rps(self) -> float:
        return max(run.throughput_rps for run in self.runs)

    def describe(self) -> str:
        lines = [f"{self.label} ({self.api_url})", ""]
        lines.append(f"{'concurrency':>12} {'p50':>9} {'p95':>9} {'p99':>9} {'req/s':>9}")
        for run in sorted(self.runs, key=lambda r: r.concurrency):
            lines.append(
                f"{run.concurrency:>12} {run.p50_ms:>8.1f}ms {run.p95_ms:>8.1f}ms "
                f"{run.p99_ms:>8.1f}ms {run.throughput_rps:>9.0f}"
            )
        service = self.service_latency
        lines.append("")
        lines.append(
            f"service latency (concurrency {service.concurrency}): "
            f"p50 {service.p50_ms:.1f} ms, p95 {service.p95_ms:.1f} ms, p99 {service.p99_ms:.1f} ms"
        )
        lines.append(f"peak throughput: {self.peak_throughput_rps:.0f} req/s")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": self.label,
            "api_url": self.api_url,
            "note": self.note,
            "service_latency": asdict(self.service_latency),
            "peak_throughput_rps": self.peak_throughput_rps,
            "runs": [asdict(run) for run in sorted(self.runs, key=lambda r: r.concurrency)],
        }
        destination.write_text(json.dumps(payload, indent=2))
        return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8",
        help="comma-separated levels to sweep, or a single number",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="unmeasured requests first: the first call pays for lazy imports",
    )
    parser.add_argument("--label", default="", help="what this run measured")
    parser.add_argument("--note", default="", help="conditions worth recording alongside it")
    parser.add_argument("--report", default=str(REPO_ROOT / "reports" / "latency.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    levels = [int(level) for level in str(args.concurrency).split(",") if level.strip()]

    if args.warmup:
        print(f"warming up with {args.warmup} requests...")
        run_load_test(args.api_url, load_payloads(args.warmup), concurrency=max(levels))

    payloads = load_payloads(args.requests)
    runs = []
    for level in levels:
        print(f"measuring at concurrency {level}...")
        run = run_load_test(
            args.api_url, payloads, concurrency=level, label=args.label, note=args.note
        )
        if run.failures:
            print(f"  WARNING: {run.failures} requests failed")
        runs.append(run)

    sweep = SweepReport(
        label=args.label or "load test", api_url=args.api_url, note=args.note, runs=runs
    )
    print()
    print(sweep.describe())
    print(f"\nreport: {sweep.save(args.report)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
