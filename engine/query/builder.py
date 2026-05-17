"""
QueryBuilder — orchestrates all sub-builders to produce SQL queries
from an enriched ParsedIntent. No LLM involved — fully deterministic.
"""
from __future__ import annotations
import logging
from models.intent import ParsedIntent
from models.query import BuiltQuery
from engine.metadata.registry import MetadataRegistry
from engine.query.router import TableRouter
from engine.query.formula_engine import FormulaEngine, QueryContext
from engine.query.having_engine import HavingEngine

logger = logging.getLogger(__name__)


class QueryBuilder:
    """
    Builds one SQL query per resolved table.

    Pipeline per table:
      SELECT  ← dimensions + absolute/avg/cumulative metrics + formula expansions
      FROM    ← table name
      WHERE   ← dimension filters + date range
      GROUP BY← all non-aggregated SELECT columns (dimensions)
      HAVING  ← from HavingEngine (metric_comparison / vs_average / gap_to_target)
      ORDER BY← from intent.sort or default
      LIMIT   ← from intent.limit or default (100)
    """

    DEFAULT_LIMIT = 100

    def __init__(
        self,
        registry: MetadataRegistry,
        router: TableRouter,
        formula_engine: FormulaEngine,
        having_engine: HavingEngine,
        context: QueryContext | None = None,
    ):
        self.registry = registry
        self.router = router
        self.formula_engine = formula_engine
        self.having_engine = having_engine
        self.context = context or QueryContext()

    def build(self, intent: ParsedIntent) -> list[BuiltQuery]:
        table_groups = self.router.resolve(intent)
        if not table_groups:
            logger.warning("No tables resolved for intent.")
            return []

        queries = []
        for tg in table_groups:
            sql = self._build_one(intent, tg.table)
            if sql:
                queries.append(BuiltQuery(sql=sql, table=tg.table, channel=tg.channel))
        return queries

    # ── Core SQL assembly ────────────────────────────────────────────────────

    def _build_one(self, intent: ParsedIntent, table: str) -> str | None:
        select_parts = self._build_select(intent, table)
        if not select_parts:
            return None

        where = self._build_where(intent, table)
        group_by = self._build_group_by(intent)
        having = self.having_engine.build(intent.having, table) if intent.having else ""
        order_by = self._build_order_by(intent)
        limit = f"LIMIT {intent.limit}" if intent.limit else f"LIMIT {self.DEFAULT_LIMIT}"

        parts = [
            f"SELECT {', '.join(select_parts)}",
            f"FROM {table}",
        ]
        if where:
            parts.append(where)
        if group_by:
            parts.append(group_by)
        if having:
            parts.append(having)
        if order_by and "ORDER BY" not in having:
            parts.append(order_by)
        parts.append(limit)

        return "\n".join(parts)

    # ── SELECT ───────────────────────────────────────────────────────────────

    def _build_select(self, intent: ParsedIntent, table: str) -> list[str]:
        parts = []

        # Dimensions first (no aggregation)
        for dim in intent.dimensions:
            col = self.registry.get_db_column(dim)
            if col and self.registry.table_has_dimension(table, dim):
                parts.append(col)

        # Absolute, average, and cumulative metrics
        for m_name in intent.metrics:
            if not self.registry.table_has_metric(table, m_name):
                continue
            col = self.registry.get_metric_column(m_name)
            agg = self.registry.get_aggregate_fn(m_name)
            if col and agg:
                parts.append(f"{agg}({col}) AS {m_name}")

        # Percentage and formula metrics
        for f_name in intent.formula_metrics:
            expr = self.formula_engine.expand(f_name, self.context)
            label = self.formula_engine.get_display_label(f_name)
            if expr:
                alias = f_name.lower().replace(" ", "_")
                parts.append(f"({expr}) AS {alias}")

        return parts

    # ── WHERE ────────────────────────────────────────────────────────────────

    def _build_where(self, intent: ParsedIntent, table: str) -> str:
        conditions = []

        for dim_name, value in intent.filters.items():
            col = self.registry.get_db_column(dim_name)
            dim = self.registry.get_dimension(dim_name)
            if col:
                safe_value = value.replace("'", "''")
                if dim and dim.filter_mode == "ilike":
                    conditions.append(f"{col} ILIKE '%{safe_value}%'")
                else:
                    conditions.append(f"{col} = '{safe_value}'")

        date_col = self.registry.get_date_column(table)
        if intent.date_range and date_col:
            conditions.append(
                f"{date_col} BETWEEN '{intent.date_range.start}' AND '{intent.date_range.end}'"
            )
        elif date_col:
            # Apply snapshot mode if configured and no explicit date range given
            table_meta = self.registry.get_table(table)
            if table_meta and table_meta.date_mode == "snapshot":
                conditions.append(
                    f"{date_col} = (SELECT MAX({date_col}) FROM {table})"
                )

        return ("WHERE " + " AND ".join(conditions)) if conditions else ""

    # ── GROUP BY ─────────────────────────────────────────────────────────────

    def _build_group_by(self, intent: ParsedIntent) -> str:
        cols = []
        for dim in intent.dimensions:
            col = self.registry.get_db_column(dim)
            if col:
                cols.append(col)
        return ("GROUP BY " + ", ".join(cols)) if cols else ""

    # ── ORDER BY ─────────────────────────────────────────────────────────────

    def _build_order_by(self, intent: ParsedIntent) -> str:
        if not intent.sort:
            return ""
        col = self.registry.get_metric_column(intent.sort.metric)
        agg = self.registry.get_aggregate_fn(intent.sort.metric)
        if col and agg:
            return f"ORDER BY {agg}({col}) {intent.sort.direction}"
        return ""
