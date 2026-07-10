"""
HavingEngine — builds SQL HAVING clauses from HavingConfig using
pattern templates defined in business_rules.json.
"""
from __future__ import annotations
import logging
from models.intent import HavingConfig
from models.metadata import HavingPatternDef
from engine.metadata.registry import MetadataRegistry

logger = logging.getLogger(__name__)


class HavingEngine:
    """
    Translates a structured HavingConfig into a SQL HAVING clause string.

    Pattern templates are loaded from business_rules.json having_patterns.
    Column names and aggregate functions are resolved from the metadata registry.
    """

    def __init__(self, registry: MetadataRegistry, patterns: list[HavingPatternDef]):
        self.registry = registry
        self.patterns: dict[str, HavingPatternDef] = {p.type: p for p in patterns}

    def build(self, having: HavingConfig, table: str, metrics: list[str] | None = None) -> str:
        if having.type == "vs_average":
            return self._build_vs_average(having, table, metrics)
        elif having.type == "gap_to_target":
            return self._build_gap_to_target(having, table)
        elif having.type == "metric_comparison":
            return self._build_metric_comparison(having, table)
        else:
            logger.warning(f"Unknown having type: {having.type}")
            return ""

    # ── Pattern builders ─────────────────────────────────────────────────────

    def _build_vs_average(
        self, having: HavingConfig, table: str, metrics: list[str] | None = None
    ) -> str:
        if having.conditions:
            cond = having.conditions[0]
            metric_name = cond.metric1
            op = cond.operator or ("<" if having.polarity == "negative" else ">")
        elif metrics:
            # LLM omitted explicit conditions — infer from intent metrics
            metric_name = next(
                (m for m in metrics if self.registry.get_metric_column(m)), None
            )
            if not metric_name:
                return ""
            op = "<" if having.polarity == "negative" else ">"
        else:
            return ""
        col = self.registry.get_metric_column(metric_name)
        agg = self.registry.get_aggregate_fn(metric_name)
        if not col:
            return ""
        # Use table-qualified column to prevent ClickHouse from confusing the
        # column name with a SELECT alias of the same name (avoids ILLEGAL_AGGREGATION).
        return f"HAVING {agg}({table}.{col}) {op} (SELECT AVG({col}) FROM {table})"

    def _build_gap_to_target(self, having: HavingConfig, table: str) -> str:
        if not having.conditions:
            return ""
        cond = having.conditions[0]
        metric_col = self.registry.get_metric_column(cond.metric1)
        tgt_col = self.registry.get_metric_column(cond.metric2)
        agg = self.registry.get_aggregate_fn(cond.metric1)
        if not metric_col or not tgt_col:
            return ""
        return (
            f"HAVING ABS(SUM({table}.{tgt_col}) - {agg}({table}.{metric_col})) > 0 "
            f"ORDER BY ABS(SUM({table}.{tgt_col}) - {agg}({table}.{metric_col})) ASC"
        )

    def _build_metric_comparison(self, having: HavingConfig, table: str) -> str:
        clauses = []
        for cond in having.conditions:
            col1 = self.registry.get_metric_column(cond.metric1)
            col2 = self.registry.get_metric_column(cond.metric2)
            agg1 = self.registry.get_aggregate_fn(cond.metric1)
            agg2 = self.registry.get_aggregate_fn(cond.metric2)
            op = cond.operator
            if col1 and col2 and op in {"<", ">", "<=", ">="}:
                # Use table-qualified column names to prevent ClickHouse from
                # resolving bare column names as SELECT aliases (ILLEGAL_AGGREGATION).
                clauses.append(f"{agg1}({table}.{col1}) {op} {agg2}({table}.{col2})")

        if not clauses:
            return ""

        # Prefer the condition_join from the parsed intent; fall back to the
        # pattern-level default defined in business_rules.json.
        join = having.condition_join if having.condition_join else "AND"
        if not join:
            pattern = self.patterns.get("metric_comparison")
            join = pattern.multi_condition_join if pattern else "AND"

        return "HAVING " + f" {join} ".join(clauses)
