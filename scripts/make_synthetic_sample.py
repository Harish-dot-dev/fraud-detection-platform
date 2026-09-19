#!/usr/bin/env python3
"""Write the synthetic IEEE-CIS-shaped fixture used by CI.

    python scripts/make_synthetic_sample.py --rows 1000 --out tests/fixtures

The output is committed to the repository so the test suite runs on a clean
clone without a Kaggle account. Generation is seeded, so re-running this with
the same arguments produces an identical file and an empty git diff.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.synthetic import generate_transactions  # noqa: E402

TRANSACTION_FILENAME = "train_transaction_sample.csv"
IDENTITY_FILENAME = "train_identity_sample.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000, help="number of transactions")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.035,
        help="share of fraudulent transactions (real data is ~3.5%%)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures",
        help="output directory",
    )
    args = parser.parse_args()

    transactions, identity = generate_transactions(
        n_rows=args.rows, seed=args.seed, fraud_rate=args.fraud_rate
    )

    args.out.mkdir(parents=True, exist_ok=True)
    transaction_path = args.out / TRANSACTION_FILENAME
    identity_path = args.out / IDENTITY_FILENAME
    transactions.to_csv(transaction_path, index=False)
    identity.to_csv(identity_path, index=False)

    fraud_count = int(transactions["isFraud"].sum())
    print(f"{transaction_path}: {len(transactions)} rows, {len(transactions.columns)} columns")
    print(f"  fraud: {fraud_count} ({fraud_count / len(transactions):.2%})")
    print(f"  cards: {transactions['card1'].nunique()} distinct card1 values")
    print(f"{identity_path}: {len(identity)} rows, {len(identity.columns)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
