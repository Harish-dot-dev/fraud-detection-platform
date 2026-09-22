"""Check that the configured LLM and embedding providers actually answer.

    make llm-check

Exists because the expensive way to discover a wrong deployment name is forty
minutes into `make llm-eval`, and the confusing way is a 404 from inside the
analyst app. This makes the smallest possible request to each configured
provider and reports what came back.

Deliberately cheap: a two-word prompt and a two-word embedding. On Azure that
is a handful of tokens, so running it before every real run costs nothing
worth counting.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from common.config import get_settings

logger = logging.getLogger("llm-check")


def check_generator(settings) -> bool:
    """One tiny chat request through whichever generator is configured."""
    from genai.summarise import build_generator

    generator = build_generator(settings)
    print(f"LLM        provider={settings.llm_provider} model={generator.name}")

    started = time.perf_counter()
    try:
        # Asks for JSON because that is what the real prompts do - a deployment
        # whose API version predates response_format fails here rather than
        # halfway through an evaluation run.
        reply = generator.generate(
            [
                {"role": "system", "content": "Reply with a single JSON object."},
                {"role": "user", "content": 'Reply with exactly {"ok": true} as JSON.'},
            ]
        )
    except Exception as error:  # noqa: BLE001 - the whole point is to report it
        print(f"           FAILED: {error}")
        return False

    elapsed = (time.perf_counter() - started) * 1000.0
    print(f"           OK in {elapsed:.0f} ms -> {reply.strip()[:80]}")
    return True


def check_embedder(settings) -> bool:
    """One tiny embedding request, checking the width pgvector expects."""
    from genai.embeddings import EMBEDDING_DIMENSIONS, build_embedder

    embedder = build_embedder(settings)
    print(f"Embedder   provider={settings.embedding_provider} model={embedder.name}")

    started = time.perf_counter()
    try:
        vectors = embedder.embed(["a card used at a new merchant"])
    except Exception as error:  # noqa: BLE001
        print(f"           FAILED: {error}")
        return False

    elapsed = (time.perf_counter() - started) * 1000.0
    width = vectors.shape[1]
    if width != EMBEDDING_DIMENSIONS:
        print(
            f"           FAILED: {width} dimensions, but the pgvector "
            f"column is {EMBEDDING_DIMENSIONS}"
        )
        return False

    print(f"           OK in {elapsed:.0f} ms -> {width} dimensions")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-embedder", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if settings.uses_paid_provider:
        print("NOTE: a paid provider is configured. This check makes two small requests.\n")

    results = []
    if not args.skip_llm:
        results.append(check_generator(settings))
    if not args.skip_embedder:
        results.append(check_embedder(settings))

    ok = all(results)
    print("\nall providers responded" if ok else "\nat least one provider failed - see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
