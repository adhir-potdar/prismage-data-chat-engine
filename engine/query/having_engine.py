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

    def build(self, having: HavingConfig, table: str) -> str:
        handler = {
            "vs_average": self._build_vs_average,
            "gap_to_target": self._build_gap_to_target,
            "metric_comparison": self._build_metric_comparison,
        }.get(having.type)

        if not handler:
            logger.warning(f"Unknown having type: {having.type}")
            return ""

        return handler(having, table)

    # ── Pattern builders ─────────────────────────────────────────────────────

    def _build_vs_average(self, having: HavingConfig, table: str) -> str:
        if not having.conditions:
            return ""
        cond = having.conditions[0]
        col = self.registry.get_metric_column(cond.metric1)
        agg = self.registry.get_aggregate_fn(cond.metric1)
        op = cond.operator or ("<" if having.polarity == "negative" else ">")
        if not col:
            return ""
        return f"HAVING {agg}({col}) {op} (SELECT AVG({col}) FROM {table})"

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
            f"HAVING ABS(SUM({tgt_col}) - {agg}({metric_col})) > 0 "
            f"ORDER BY ABS(SUM({tgt_col}) - {agg}({metric_col})) ASC"
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
                clauses.append(f"{agg1}({col1}) {op} {agg2}({col2})")

        if not clauses:
            return ""

        # Prefer the condition_join from the parsed intent; fall back to the
        # pattern-level default defined in business_rules.json.
        join = having.condition_join if having.condition_join else "AND"
        if not join:
            pattern = self.patterns.get("metric_comparison")
            join = pattern.multi_condition_join if pattern else "AND"

        return "HAVING " + f" {join} ".join(clauses)
