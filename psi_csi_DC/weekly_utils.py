"""
Weekly window helpers for PSI/CSI.

Weeks run Sunday (start) -> Saturday (end), inclusive on both ends.
Only *completed* weeks are returned by default, so a drift value is never
computed on a partially-populated week (which would look like a volume drop
rather than a distribution shift).
"""

from datetime import date, timedelta

import pandas as pd

# Weeks run Sunday -> Saturday. These two constants are the only place the week
# boundary is defined; change both together if the business definition ever moves
# (e.g. 'W-MON' / 0 for Monday -> Sunday).
# Note the pandas alias names the weekday the generated dates land on, so
# 'W-SUN' yields the Sundays that start each week.
WEEK_START_FREQ = 'W-SUN'
WEEK_START_WEEKDAY = 6  # Sunday, matching date.weekday() (Mon=0 ... Sun=6)


def _to_date(value):
    """Accepts date / datetime / 'YYYY-MM-DD' and returns a datetime.date."""
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.to_datetime(value).date()


def snap_to_week_start(value):
    """Rolls a date back to the Sunday of the week it falls in."""
    d = _to_date(value)
    offset = (d.weekday() - WEEK_START_WEEKDAY) % 7
    return d - timedelta(days=offset)


def get_last_completed_week(reference_date=None):
    """
    Returns (week_start, week_end) as dates for the most recent week that has
    fully elapsed relative to reference_date (defaults to today).

    Example: run on Tue 2026-09-15 -> (2026-09-06, 2026-09-12)
    """
    ref = _to_date(reference_date) if reference_date is not None else date.today()
    current_week_start = snap_to_week_start(ref)
    week_end = current_week_start - timedelta(days=1)   # previous Saturday
    week_start = week_end - timedelta(days=6)           # previous Sunday
    return week_start, week_end


def get_recent_completed_weeks(n_weeks=1, reference_date=None):
    """
    Returns (prod_start_date, prod_end_date) as 'YYYY-MM-DD' strings spanning
    the last n_weeks completed weeks. n_weeks=1 is the normal weekly run;
    n_weeks>1 is a backfill.
    """
    if n_weeks < 1:
        raise ValueError("n_weeks must be >= 1")

    latest_start, latest_end = get_last_completed_week(reference_date)
    earliest_start = latest_start - timedelta(weeks=n_weeks - 1)
    return earliest_start.strftime("%Y-%m-%d"), latest_end.strftime("%Y-%m-%d")


def get_week_windows(prod_start_date, prod_end_date, include_partial_weeks=False):
    """
    Splits [prod_start_date, prod_end_date] into weekly windows.

    The monthly job does:
        prod_months = pd.date_range(start, end, freq='MS')
        end_date    = start_date + pd.DateOffset(months=1) - timedelta(days=1)

    This is the weekly equivalent. The start is snapped back to its Sunday so a
    hardcoded override like '2026-08-05' still produces whole weeks.

    Returns a list of (week_label, start_ts, end_ts) where:
        week_label -> 'YYYY-MM-DD' of the Sunday, used as run_date
        start_ts   -> Sunday   00:00 Timestamp
        end_ts     -> Saturday 00:00 Timestamp (publish_time is a DATE, so an
                      inclusive .between() on midnight timestamps is correct)
    """
    range_start = snap_to_week_start(prod_start_date)
    range_end = _to_date(prod_end_date)

    if range_end < range_start:
        raise ValueError(
            f"prod_end_date ({range_end}) is before the week start ({range_start})"
        )

    week_starts = pd.date_range(range_start, range_end, freq=WEEK_START_FREQ)

    data_start = _to_date(prod_start_date)

    windows = []
    for start_ts in week_starts:
        end_ts = start_ts + timedelta(days=6)

        if not include_partial_weeks:
            # A week only counts if the whole Sun-Sat span is inside the data
            # range. Otherwise the drift value reflects missing days, not drift.
            if start_ts.date() < data_start:
                print(
                    f"Skipping partial week starting {start_ts.date()} "
                    f"(data only starts {data_start})"
                )
                continue
            if end_ts.date() > range_end:
                print(
                    f"Skipping partial week starting {start_ts.date()} "
                    f"(ends {end_ts.date()}, data only available to {range_end})"
                )
                continue

        windows.append((start_ts.strftime("%Y-%m-%d"), start_ts, end_ts))

    return windows


def get_query_range(windows):
    """
    Returns ('YYYY-MM-DD', 'YYYY-MM-DD') covering every window, so the prod SQL
    pulls exactly the days that will be scored.
    """
    if not windows:
        raise ValueError("no complete weeks to score")

    return (
        windows[0][1].strftime("%Y-%m-%d"),
        windows[-1][2].strftime("%Y-%m-%d"),
    )
