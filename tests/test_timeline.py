"""Tests for the synthetic timeline.

Everything time-related in the platform - feature windows, the train/test
split, chargeback arrival - is built on this mapping, so it is worth nailing
down precisely.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from common.timeline import (
    DEFAULT_EPOCH,
    FIRST_TRANSACTION_DT,
    dataset_span_days,
    parse_epoch,
    to_event_time,
    to_transaction_dt,
)


def test_parse_epoch_handles_the_z_suffix() -> None:
    assert parse_epoch("2023-01-01T00:00:00Z") == DEFAULT_EPOCH
    assert parse_epoch("2023-01-01T00:00:00+00:00") == DEFAULT_EPOCH


def test_parse_epoch_assumes_utc_when_no_offset_is_given() -> None:
    assert parse_epoch("2023-01-01T00:00:00") == DEFAULT_EPOCH
    assert parse_epoch(datetime(2023, 1, 1)) == DEFAULT_EPOCH


def test_first_transaction_lands_one_day_after_the_epoch() -> None:
    """A surprise worth documenting: the dataset's clock starts at 86400."""
    assert to_event_time(FIRST_TRANSACTION_DT) == DEFAULT_EPOCH + timedelta(days=1)


def test_offsets_map_to_timestamps_in_order() -> None:
    earlier = to_event_time(100_000)
    later = to_event_time(100_060)

    assert later - earlier == timedelta(seconds=60)
    assert earlier.tzinfo is UTC


def test_round_trip_is_lossless_to_the_second() -> None:
    for offset in (86_400, 123_456, 15_811_131):
        assert to_transaction_dt(to_event_time(offset)) == offset


def test_round_trip_accepts_naive_datetimes() -> None:
    naive = datetime(2023, 1, 2)

    assert to_transaction_dt(naive) == 86_400


def test_a_different_epoch_shifts_every_timestamp_equally() -> None:
    other = datetime(2017, 12, 1, tzinfo=UTC)

    shifted = to_event_time(86_400, epoch=other)
    assert shifted == datetime(2017, 12, 2, tzinfo=UTC)
    assert to_transaction_dt(shifted, epoch=other) == 86_400


def test_dataset_span_days() -> None:
    assert dataset_span_days(86_400, 86_400 + 86_400 * 182) == 182.0
