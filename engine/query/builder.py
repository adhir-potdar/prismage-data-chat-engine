"""
QueryBuilder — orchestrates all sub-builders to produce SQL queries
from an enriched ParsedIntent. No LLM involved — fully deterministic.
"""
from __future__ import annotations
import logging
from models.intent import ParsedIntent
from models.query import BuiltQuery
from engine.capabilities.base import EngineCapabilities
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
        capabilities: EngineCapabilities | None = None,
    ):
        self.registry = registry
        self.router = router
        self.formula_engine = formula_engine
        self.having_engine = having_engine
        self.context = context or QueryContext()
        self.capabilities = capabilities or EngineCapabilities()

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

        # Skip table if metrics were requested but none are available on it.
        # Use explicit registry lookups instead of fragile string heuristics.
        available_metrics = sum(
            1 for m in intent.metrics if self.registry.table_has_metric(table, m)
        )
        available_formula_metrics = sum(
            1 for f in intent.formula_metrics if self.registry.table_has_metric(table, f)
        )
        if (intent.metrics or intent.formula_metrics) and (available_metrics + available_formula_metrics) == 0:
            logger.debug(f"Skipping table '{table}': no requested metrics available.")
            return None

        # Skip table if dimensions were requested but none are available on it.
        # A query with aggregated metrics but no GROUP BY returns one grand-total
        # row that cannot be meaningfully merged with grouped results from other tables.
        available_dimensions = sum(
            1 for d in intent.dimensions if self.registry.table_has_dimension(table, d)
        )
        if intent.dimensions and available_dimensions == 0:
            logger.debug(f"Skipping table '{table}': no requested dimensions available.")
            return None

        where = self._build_where(intent, table)
        group_by = self._build_group_by(intent, table)
        having = self.having_engine.build(intent.having, table, intent.metrics) if intent.having else ""
        order_by = self._build_order_by(intent, table)
        # Always fetch DEFAULT_LIMIT rows in SQL; user's intent.limit is applied
        # post-merge in the NL responder to avoid premature truncation when
        # results from multiple tables are merged before the final rank/slice.
        limit = f"LIMIT {self.DEFAULT_LIMIT}"

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

        suffix = self.capabilities.get_metric_suffix(table)

        # Absolute, average, and cumulative metrics
        for m_name in intent.metrics:
            if not self.registry.table_has_metric(table, m_name):
                continue
            col = self.registry.get_metric_column(m_name)
            agg = self.registry.get_aggregate_fn(m_name)
            if col and agg:
                parts.append(f"{agg}({col}) AS {m_name}{suffix}")
            else:
                # Metric has no db_column — may be a formula metric misrouted into
                # intent.metrics by the LLM. Expand via formula_ref if available.
                m_meta = self.registry.get_metric(m_name)
                if m_meta and m_meta.formula_ref:
                    expr = self.formula_engine.expand(m_meta.formula_ref, self.context, table=table)
                    if expr:
                        parts.append(f"({expr}) AS {m_name}{suffix}")

        # Percentage and formula metrics
        for f_name in intent.formula_metrics:
            if not self.registry.table_has_metric(table, f_name):
                continue
            if not self.registry.get_formula(f_name):
                # Regular metric mistakenly placed in formula_metrics by the LLM
                # — already handled by the metrics loop above; skip silently.
                continue
            expr = self.formula_engine.expand(f_name, self.context, table=table)
            if expr:
                alias = f_name.lower().replace(" ", "_")
                parts.append(f"({expr}) AS {alias}{suffix}")

        # Always include data_as_of so the response knows the data date
        date_col = self.registry.get_date_column(table)
        if date_col:
            parts.append(f"MAX({date_col}) AS data_as_of")

        return parts

    # ── WHERE ────────────────────────────────────────────────────────────────

    # Placeholder values treated as missing data — excluded from GROUP BY queries
    _NULL_PLACEHOLDERS = ("'-'", "'N.A.'", "'NA'", "'N/A'")

    def _build_where(self, intent: ParsedIntent, table: str) -> str:
        conditions = []

        for dim_name, value in intent.filters.items():
            if not self.registry.table_has_dimension(table, dim_name):
                continue
            col = self.registry.get_db_column(dim_name)
            dim = self.registry.get_dimension(dim_name)
            safe_value = str(value).replace("'", "''")
            if col:
                if dim and dim.filter_mode == "ilike":
                    conditions.append(f"{col} ILIKE '%{safe_value}%'")
                else:
                    conditions.append(f"{col} = '{safe_value}'")
            elif dim and dim.hierarchy_name:
                # Virtual dimension (db_column=None) — OR-expand across all real
                # dimensions in the same hierarchy that exist in this table.
                hier_cols = [
                    self.registry.get_db_column(d)
                    for d in self.registry.get_dimensions_by_hierarchy(dim.hierarchy_name)
                    if self.registry.table_has_dimension(table, d)
                ]
                if hier_cols:
                    or_parts = [f"{c} ILIKE '%{safe_value}%'" for c in hier_cols]
                    conditions.append(f"({' OR '.join(or_parts)})")

        # Exclude null/placeholder rows for all grouped dimensions
        for dim_name in intent.dimensions:
            if not self.registry.table_has_dimension(table, dim_name):
                continue
            col = self.registry.get_db_column(dim_name)
            if col:
                placeholders = ", ".join(self._NULL_PLACEHOLDERS)
                conditions.append(
                    f"({col} IS NOT NULL AND {col} NOT IN ({placeholders}))"
                )

        date_col = self.registry.get_date_column(table)
        table_meta = self.registry.get_table(table)
        if intent.date_range and date_col:
            conditions.append(
                self.capabilities.build_date_filter(table, date_col, intent.date_range, table_meta)
            )
        elif date_col and table_meta and table_meta.date_mode == "snapshot":
            conditions.append(
                self.capabilities.build_snapshot_filter(table, date_col, table_meta)
            )

        return ("WHERE " + " AND ".join(conditions)) if conditions else ""

    # ── GROUP BY ─────────────────────────────────────────────────────────────

    def _build_group_by(self, intent: ParsedIntent, table: str) -> str:
        cols = []
        for dim in intent.dimensions:
            col = self.registry.get_db_column(dim)
            if col and self.registry.table_has_dimension(table, dim):
                cols.append(col)
        return ("GROUP BY " + ", ".join(cols)) if cols else ""

    # ── ORDER BY ─────────────────────────────────────────────────────────────

    def _build_order_by(self, intent: ParsedIntent, table: str) -> str:
        suffix = self.capabilities.get_metric_suffix(table)

        if intent.sort and intent.sort.metric:
            metric = intent.sort.metric
            # Only add ORDER BY if this table actually has the sort metric
            if not self.registry.table_has_metric(table, metric):
                return ""
            col = self.registry.get_metric_column(metric)
            agg = self.registry.get_aggregate_fn(metric)
            if col and agg:
                # Use the SELECT alias rather than repeating the aggregation expression.
                # Re-wrapping the alias in the aggregate (e.g. SUM(SUM(col))) is
                # illegal in ClickHouse and redundant in PostgreSQL — both support
                # ORDER BY alias defined in SELECT.
                return f"ORDER BY {metric}{suffix} {intent.sort.direction}"
            # Formula / percentage metric — use SELECT alias
            return f"ORDER BY {metric}{suffix} {intent.sort.direction}"

        # Default: sort by all requested metrics using direction from intent.sort (if set) or DESC
        direction = intent.sort.direction if intent.sort else "DESC"
        order_cols = []
        for m_name in intent.metrics:
            if not self.registry.table_has_metric(table, m_name):
                continue
            col = self.registry.get_metric_column(m_name)
            m_meta = self.registry.get_metric(m_name)
            if col or (m_meta and m_meta.formula_ref):
                order_cols.append(f"{m_name}{suffix} {direction}")
        for f_name in intent.formula_metrics:
            if not self.registry.table_has_metric(table, f_name):
                continue
            order_cols.append(f"{f_name}{suffix} {direction}")
        return f"ORDER BY {', '.join(order_cols)}" if order_cols else ""
