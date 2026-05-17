"""
EngineCapabilities — base class defining overridable engine behaviours.

Plugins can subclass this in their own capabilities.py to override specific
behaviours without touching generic engine code.  The loader detects a
capabilities.py in the plugin directory, imports the first subclass it finds,
instantiates it, and injects it into the engine pipeline.

Currently overridable:
    build_date_filter   — WHERE clause when an explicit date range is given
    build_snapshot_filter — WHERE clause when no date range is given but the
                           table is configured with date_mode = "snapshot"

Add new methods here (with sensible defaults) whenever a new engine behaviour
needs to be plugin-configurable.  Plugins only override the methods they need;
all others fall through to the defaults below.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.intent import DateRange
    from models.metadata import Table


class EngineCapabilities:
    """Default engine capabilities. Suitable for most plugins."""

    def build_date_filter(
        self,
        table: str,
        date_col: str,
        date_range: "DateRange",
        table_meta: "Table | None" = None,
    ) -> str:
        """
        Return the WHERE condition for an explicit date range.

        Default: BETWEEN covering every row in the range.
        Suitable for time-series tables where all rows in a period are needed.
        """
        return f"{date_col} BETWEEN '{date_range.start}' AND '{date_range.end}'"

    def build_snapshot_filter(
        self,
        table: str,
        date_col: str,
        table_meta: "Table | None" = None,
    ) -> str:
        """
        Return the WHERE condition when no date range is given but the table
        uses date_mode = "snapshot".

        Default: latest available row globally.
        """
        return f"{date_col} = (SELECT MAX({date_col}) FROM {table})"
