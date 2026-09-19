"""Mapping the dataset's ``TransactionDT`` offset onto a real calendar.

``TransactionDT`` in the IEEE-CIS data is **not** a timestamp: it is a number of
seconds since an origin Vesta never published. The column is useful (the gaps
between transactions are real) but you cannot window over it, split on a date,
or simulate a chargeback arriving "three weeks later" without a calendar.

So the whole platform anchors the offsets to a fixed reference date:

    event_time = SYNTHETIC_EPOCH + TransactionDT seconds

The absolute dates are therefore fictional; every interval between them is real.
Because the mapping is a pure function of a configured constant, it is the same
in the producer, the streaming job, the training set builder and the chargeback
simulation - which is what stops those four pieces from quietly disagreeing
about what "day 40" means.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

# Matches SYNTHETIC_EPOCH in .env.example.
DEFAULT_EPOCH = datetime(2023, 1, 1, tzinfo=UTC)

# The smallest TransactionDT in the real dataset. Transactions therefore begin
# one day after the epoch, which is worth knowing when reading a date off a
# dashboard and wondering where day zero went.
FIRST_TRANSACTION_DT = 86_400


def parse_epoch(value: str | datetime) -> datetime:
    """Parse the configured epoch into a timezone-aware UTC datetime.

    Accepts ISO-8601 strings, including the ``Z`` suffix that
    ``datetime.fromisoformat`` only learned to handle in Python 3.11.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def to_event_time(transaction_dt: int | float, epoch: datetime = DEFAULT_EPOCH) -> datetime:
    """Convert a ``TransactionDT`` offset to a wall-clock timestamp."""
    return epoch + timedelta(seconds=float(transaction_dt))


def to_transaction_dt(event_time: datetime, epoch: datetime = DEFAULT_EPOCH) -> int:
    """Inverse of :func:`to_event_time`, rounded to whole seconds.

    Needed by the chargeback simulation, which works in calendar days and has
    to map back onto the dataset's own clock.
    """
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=UTC)
    return int(round((event_time - epoch).total_seconds()))


def dataset_span_days(first_dt: int | float, last_dt: int | float) -> float:
    """Number of days covered by a range of ``TransactionDT`` values."""
    return (float(last_dt) - float(first_dt)) / 86_400.0
