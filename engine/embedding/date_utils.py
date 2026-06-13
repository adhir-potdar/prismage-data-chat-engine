"""
Generic date utilities for embedding-based search.

Pure functions — no domain knowledge, no external dependencies.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, Tuple


def validate_yyyymmdd(date_str: str) -> bool:
    """Return True if date_str is a valid YYYYMMDD string."""
    if not date_str or len(date_str) != 8 or not date_str.isdigit():
        return False
    try:
        datetime.strptime(date_str, '%Y%m%d')
        return True
    except ValueError:
        return False


def format_date_readable(date_int: int) -> str:
    """
    Format integer date (YYYYMMDD) to human-readable string.

    Example: 20250105 -> "Jan 05, 2025"
    """
    try:
        return datetime.strptime(str(date_int), '%Y%m%d').strftime('%b %d, %Y')
    except (ValueError, TypeError):
        return f"Date {date_int}"


def extract_dates_from_collection_name(
    collection_name: str,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Extract period dates from a collection name.

    Expected format:
        {dimension}_{granularity}_{start1}_{end1}_vs_{start2}_{end2}[_date_{analysis_date}]

    Returns:
        (period1_start, period1_end, period2_start, period2_end) as ints, or None on parse failure.
    """
    try:
        if '_vs_' not in collection_name:
            return None, None, None, None

        before_vs, after_vs_full = collection_name.split('_vs_', 1)
        after_vs = after_vs_full.split('_date_')[0]  # strip optional _date_ suffix

        before_parts = before_vs.split('_')
        after_parts = after_vs.split('_')

        if len(before_parts) < 2 or len(after_parts) < 2:
            return None, None, None, None

        return (
            int(before_parts[-2]),
            int(before_parts[-1]),
            int(after_parts[0]),
            int(after_parts[1]),
        )
    except (ValueError, IndexError):
        return None, None, None, None


def collection_overlaps_date_range(
    collection_name: str,
    start_date: int,
    end_date: int,
) -> bool:
    """
    Return True if either period in the collection overlaps with [start_date, end_date].

    Overlap condition: period_start <= end_date AND period_end >= start_date.
    If dates cannot be parsed, returns True (include by default).
    """
    p1_start, p1_end, p2_start, p2_end = extract_dates_from_collection_name(collection_name)

    if p1_start is None:
        return True  # Can't determine — include collection

    period1_overlaps = (p1_start <= end_date) and (p1_end >= start_date)
    period2_overlaps = (p2_start <= end_date) and (p2_end >= start_date)
    return period1_overlaps or period2_overlaps


def parse_and_validate_date_range(
    date_range_info: dict,
    current_date_str: str,
) -> Tuple[Optional[int], Optional[int], str, Optional[str]]:
    """
    Parse and validate an extracted date range dict.

    Args:
        date_range_info: Dict with keys: specified, start_date, end_date, confidence
        current_date_str: Today's date as YYYYMMDD string

    Returns:
        (start_date, end_date, status, error_message)
        status: 'valid' | 'not_specified' | 'invalid_format' | 'invalid_range' | 'future_date'
    """
    if not date_range_info or not date_range_info.get('specified', False):
        return None, None, 'not_specified', None

    start_str = str(date_range_info.get('start_date', ''))
    end_str = str(date_range_info.get('end_date', ''))

    if not validate_yyyymmdd(start_str) or not validate_yyyymmdd(end_str):
        return None, None, 'invalid_format', f"Invalid date format: start={start_str}, end={end_str}"

    start_date = int(start_str)
    end_date = int(end_str)

    if end_date < start_date:
        return None, None, 'invalid_range', f"End date {end_date} is before start date {start_date}"

    try:
        current_date_int = int(current_date_str)
        if start_date > current_date_int and end_date > current_date_int:
            return None, None, 'future_date', f"Date range {start_date}-{end_date} is entirely in the future"
    except (ValueError, TypeError):
        pass

    return start_date, end_date, 'valid', None
