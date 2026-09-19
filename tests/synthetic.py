"""Synthetic stand-in for the IEEE-CIS Fraud Detection dataset.

Why this exists
---------------
The real dataset is competition data: it cannot be committed (licence) and it
needs a Kaggle token to download. CI has neither. So every test, and every
smoke run of the pipeline, uses data generated here instead.

The generator reproduces the *schema* of ``train_transaction.csv`` and
``train_identity.csv`` exactly - same column names, same order, same mix of
numeric / categorical / mostly-missing columns - so that code written against
the fixture also works against the real files.

It is NOT a statistical clone of the real data. It has:
  * a realistic class imbalance (~3.5% fraud, as in the real data),
  * card-level structure, so per-card velocity features have something to find,
  * fraud injected as *bursts* on a compromised card with a new device,

which is enough to exercise the feature, training and scoring code. Any model
metric produced from this data is meaningless as a measure of real performance
and must never be published as one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- Schema -----------------------------------------------------------------
# Column names and order taken from the real train_transaction.csv /
# train_identity.csv headers.

_BASE_TRANSACTION_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
]
_C_COLUMNS = [f"C{i}" for i in range(1, 15)]
_D_COLUMNS = [f"D{i}" for i in range(1, 16)]
_M_COLUMNS = [f"M{i}" for i in range(1, 10)]
_V_COLUMNS = [f"V{i}" for i in range(1, 340)]

TRANSACTION_COLUMNS: list[str] = (
    _BASE_TRANSACTION_COLUMNS + _C_COLUMNS + _D_COLUMNS + _M_COLUMNS + _V_COLUMNS
)

_ID_NUMERIC_COLUMNS = [f"id_{i:02d}" for i in range(1, 12)]
_ID_CATEGORICAL_COLUMNS = [f"id_{i:02d}" for i in range(12, 39)]
IDENTITY_COLUMNS: list[str] = (
    ["TransactionID"] + _ID_NUMERIC_COLUMNS + _ID_CATEGORICAL_COLUMNS + ["DeviceType", "DeviceInfo"]
)

# The handful of V columns the generator actually populates. In the real data
# the V block is ~90% missing in places, so leaving the rest empty is faithful.
# V258 / V294 / V317 are among the columns that consistently rank highly in
# public solutions, so those are the ones given a fraud signal here.
_POPULATED_V_COLUMNS = ["V1", "V2", "V3", "V4", "V5", "V12", "V13", "V30", "V62", "V83"]
_SIGNAL_V_COLUMNS = ["V258", "V294", "V317"]

_CARD_BRANDS = ["visa", "mastercard", "american express", "discover"]
_CARD_TYPES = ["debit", "credit"]
_PRODUCT_CODES = ["W", "C", "R", "H", "S"]
_EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "anonymous.com",
    "aol.com",
    "outlook.com",
    "comcast.net",
    "icloud.com",
]
_DEVICE_INFO = [
    "Windows",
    "iOS Device",
    "MacOS",
    "Trident/7.0",
    "SM-G930V Build/NRD90M",
    "rv:57.0",
    "SAMSUNG SM-G950F Build/R16NW",
]
_BROWSERS = ["chrome 63.0", "mobile safari 11.0", "ie 11.0 for desktop", "firefox 57.0", "safari"]
_OPERATING_SYSTEMS = ["Windows 10", "iOS 11.1.2", "Mac OS X 10_13_1", "Android 7.0", "Windows 7"]
_SCREEN_RESOLUTIONS = ["1920x1080", "1366x768", "2208x1242", "1334x750", "1280x800"]

# Real transactions start one day into the synthetic clock and the dataset spans
# roughly six months. TransactionDT is a seconds offset from an unknown origin.
_FIRST_TRANSACTION_DT = 86_400


@dataclass(frozen=True)
class _Card:
    """One synthetic card identity.

    The real dataset has no card ID, so the pipeline builds a proxy from
    card1 + addr1 + P_emaildomain (see docs/design_decisions.md). Generating
    cards as real entities first, and only then flattening them into rows, is
    what makes that proxy meaningful in the fixture.
    """

    card1: int
    card2: float
    card3: float
    card4: str
    card5: float
    card6: str
    addr1: float
    addr2: float
    email: str
    device_info: str
    device_type: str
    typical_amount: float


def _make_cards(rng: np.random.Generator, n_cards: int) -> list[_Card]:
    return [
        _Card(
            card1=int(rng.integers(1000, 18500)),
            card2=float(rng.integers(100, 600)),
            card3=150.0,
            card4=str(rng.choice(_CARD_BRANDS, p=[0.65, 0.28, 0.04, 0.03])),
            card5=float(rng.integers(100, 240)),
            card6=str(rng.choice(_CARD_TYPES, p=[0.75, 0.25])),
            addr1=float(rng.integers(100, 540)),
            addr2=87.0,
            email=str(rng.choice(_EMAIL_DOMAINS)),
            device_info=str(rng.choice(_DEVICE_INFO)),
            device_type=str(rng.choice(["desktop", "mobile"], p=[0.6, 0.4])),
            # Log-normal spend: most cards small, a few big spenders.
            typical_amount=float(np.exp(rng.normal(4.0, 0.7))),
        )
        for _ in range(n_cards)
    ]


def generate_transactions(
    n_rows: int = 1000,
    seed: int = 42,
    fraud_rate: float = 0.035,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic (transactions, identity) pair.

    Args:
        n_rows: number of transactions to generate.
        seed: RNG seed - the same seed always produces byte-identical output.
        fraud_rate: share of transactions marked ``isFraud=1``. The real
            dataset sits at about 3.5%.

    Returns:
        ``(transactions, identity)``. The identity frame covers only a subset
        of transactions, as in the real data where roughly a quarter of rows
        have a matching identity record.
    """
    rng = np.random.default_rng(seed)

    # Roughly 20 transactions per card, with a floor so small samples still
    # produce repeat activity for the velocity features to pick up.
    n_cards = max(20, n_rows // 20)
    cards = _make_cards(rng, n_cards)

    # Card popularity is heavily skewed: a few cards transact constantly.
    popularity = rng.dirichlet(np.full(n_cards, 0.7))

    n_fraud = max(1, int(round(n_rows * fraud_rate)))
    # Fraud arrives in bursts on a handful of compromised cards rather than
    # being sprinkled uniformly - that is what makes velocity features work.
    n_compromised = max(1, n_fraud // 4)
    compromised_idx = rng.choice(n_cards, size=min(n_compromised, n_cards), replace=False)

    # Inter-arrival times: mean ~40 seconds of dataset time between payments.
    gaps = rng.exponential(40.0, size=n_rows).astype(int) + 1
    transaction_dt = _FIRST_TRANSACTION_DT + np.cumsum(gaps)

    is_fraud = np.zeros(n_rows, dtype=int)
    fraud_positions = rng.choice(n_rows, size=n_fraud, replace=False)
    is_fraud[fraud_positions] = 1

    card_idx = rng.choice(n_cards, size=n_rows, p=popularity)
    # Fraudulent rows are reassigned to a compromised card.
    card_idx[fraud_positions] = rng.choice(compromised_idx, size=n_fraud)

    rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []

    for i in range(n_rows):
        card = cards[card_idx[i]]
        fraud = bool(is_fraud[i])

        # Fraudulent amounts skew high relative to the card's own baseline,
        # which is the signal the "amount vs card average" feature exists for.
        multiplier = (
            float(np.exp(rng.normal(1.1, 0.5))) if fraud else float(np.exp(rng.normal(0.0, 0.4)))
        )
        amount = round(card.typical_amount * multiplier, 3)

        row: dict[str, object] = {
            "TransactionID": 2_987_000 + i,
            "isFraud": int(fraud),
            "TransactionDT": int(transaction_dt[i]),
            "TransactionAmt": amount,
            "ProductCD": str(rng.choice(_PRODUCT_CODES, p=[0.74, 0.12, 0.06, 0.05, 0.03])),
            "card1": card.card1,
            "card2": card.card2,
            "card3": card.card3,
            "card4": card.card4,
            "card5": card.card5,
            "card6": card.card6,
            "addr1": card.addr1,
            "addr2": card.addr2,
            # dist1/dist2 are mostly missing in the real data.
            "dist1": float(rng.integers(0, 400)) if rng.random() < 0.40 else np.nan,
            "dist2": float(rng.integers(0, 400)) if rng.random() < 0.07 else np.nan,
            "P_emaildomain": card.email,
            # Fraud often shows a mismatched recipient domain.
            "R_emaildomain": (
                str(rng.choice(_EMAIL_DOMAINS))
                if fraud or rng.random() < 0.25
                else (card.email if rng.random() < 0.5 else np.nan)
            ),
        }

        # C1-C14: counting features (how many addresses/phones/emails are
        # associated with the card). Higher counts correlate with fraud.
        count_scale = 6.0 if fraud else 1.5
        for c in _C_COLUMNS:
            row[c] = float(rng.poisson(count_scale))

        # D1-D15: "days since" features. D1 is days since the card first
        # appeared; the rest are frequently missing.
        row["D1"] = float(rng.integers(0, 640))
        for d in _D_COLUMNS[1:]:
            row[d] = float(rng.integers(0, 640)) if rng.random() < 0.45 else np.nan

        # M1-M9: match flags (name on card matches, address matches, ...).
        # A mismatch is a genuine fraud signal, so fraud rows fail more often.
        for m in _M_COLUMNS:
            if rng.random() < 0.45:
                row[m] = np.nan
            elif m == "M4":
                row[m] = str(rng.choice(["M0", "M1", "M2"]))
            else:
                row[m] = "F" if rng.random() < (0.55 if fraud else 0.15) else "T"

        # V1-V339: Vesta's engineered features. Left empty except for a small
        # populated block, mirroring how sparse this part of the real file is.
        for v in _V_COLUMNS:
            row[v] = np.nan
        for v in _POPULATED_V_COLUMNS:
            row[v] = float(rng.integers(0, 3))
        for v in _SIGNAL_V_COLUMNS:
            row[v] = round(float(rng.normal(3.0 if fraud else 0.0, 1.0)), 4)

        rows.append(row)

        # Identity is present for roughly a quarter of transactions, and more
        # often for fraud (fraudsters use channels that carry device data).
        if rng.random() < (0.55 if fraud else 0.22):
            identity_rows.append(
                _make_identity_row(rng, transaction_id=row["TransactionID"], card=card, fraud=fraud)
            )

    transactions = pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)
    identity = pd.DataFrame(identity_rows, columns=IDENTITY_COLUMNS)
    return transactions, identity


def _make_identity_row(
    rng: np.random.Generator, transaction_id: int, card: _Card, fraud: bool
) -> dict[str, object]:
    """Build one train_identity.csv row for a transaction."""
    row: dict[str, object] = {"TransactionID": transaction_id}

    # id_01-id_11 are numeric risk scores; id_01 and id_02 are almost always
    # present, the rest much less so.
    row["id_01"] = float(-5 * rng.integers(0, 20))
    row["id_02"] = float(rng.integers(1000, 600000))
    for col in _ID_NUMERIC_COLUMNS[2:]:
        row[col] = round(float(rng.normal(50, 25)), 2) if rng.random() < 0.25 else np.nan

    for col in _ID_CATEGORICAL_COLUMNS:
        row[col] = np.nan
    row["id_12"] = str(rng.choice(["Found", "NotFound"], p=[0.35, 0.65]))
    row["id_15"] = str(rng.choice(["New", "Found", "Unknown"], p=[0.35, 0.5, 0.15]))
    row["id_16"] = str(rng.choice(["Found", "NotFound"], p=[0.7, 0.3]))
    row["id_28"] = str(rng.choice(["New", "Found"], p=[0.4, 0.6]))
    row["id_29"] = str(rng.choice(["Found", "NotFound"], p=[0.7, 0.3]))
    row["id_30"] = str(rng.choice(_OPERATING_SYSTEMS))
    row["id_31"] = str(rng.choice(_BROWSERS))
    row["id_33"] = str(rng.choice(_SCREEN_RESOLUTIONS))
    row["id_34"] = f"match_status:{rng.integers(0, 3)}"
    for col in ["id_35", "id_36", "id_37", "id_38"]:
        row[col] = "T" if rng.random() < 0.6 else "F"

    # A device the card has never been seen on is one of the strongest signals
    # available at scoring time, so fraud rows usually carry an unfamiliar one.
    if fraud and rng.random() < 0.8:
        row["DeviceType"] = str(rng.choice(["desktop", "mobile"]))
        row["DeviceInfo"] = str(rng.choice(_DEVICE_INFO))
    else:
        row["DeviceType"] = card.device_type
        row["DeviceInfo"] = card.device_info
    return row
