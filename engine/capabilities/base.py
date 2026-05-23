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

    def get_metric_suffix(self, table_name: str) -> str:
        """
        Return a suffix to append to each metric column alias in SELECT.

        Default: no suffix (e.g. cymtd → cymtd).
        Override in plugins that run multiple sub-table variants for the same channel
        so the ResultMerger can join them without column collisions.
        Example: a plugin with _val and _vol table variants might return "_val" or "_vol"
        so cymtd becomes cymtd_val / cymtd_vol in the merged result.
        """
        return ""

    def get_default_metrics(self) -> list[str]:
        """
        Metrics to use when the LLM extracts none from the question.

        Default: [] (no fallback — leave metrics empty and let the query builder
        decide or return an error). Override in plugins to provide a sensible
        business default (e.g. ["fy_25"] for annual sales data).
        """
        return []

    def get_embedding_threshold(self) -> float:
        """
        Minimum similarity ratio relative to the top-scoring table.

        Tables whose cosine similarity score is below (top_score * threshold)
        are excluded by the embedding router. Default 0.0 = no filtering.

        Override in plugins with multiple table entity types (e.g. product vs
        sales_rep tables that share dimensions) to prevent over-routing.
        Example: return 0.90 to keep only tables within 10% of the top score.
        """
        return 0.0

    def get_default_tables(self) -> list[str]:
        """
        Tables to use when no dimension is specified in the question.

        Default: [] (no restriction). Override in plugins to designate a compact
        summary table set for dimension-free aggregate queries so routing does not
        fan out to all table types simultaneously.
        """
        return []

    def get_default_table_variant(self) -> str | None:
        """
        Table variant to apply as default when no explicit variant was requested
        and no dimension is being grouped.

        Default: None (no restriction). Override in plugins where multiple table
        variants exist (e.g. value vs volume) to avoid merging incompatible
        variants within the same channel when there are no join keys.
        """
        return None
