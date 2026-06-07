"""
FormulaEngine — expands formula templates into SQL expressions.
Substitutes component db_columns and runtime variables at query build time.
"""
from __future__ import annotations
import logging
from models.metadata import Formula
from engine.metadata.registry import MetadataRegistry

logger = logging.getLogger(__name__)


class QueryContext:
    """Runtime variables injected into formulas that need date-based calculations."""

    def __init__(self, days_elapsed: int = 0, total_days: int = 30, days_remaining: int = 0):
        self.days_elapsed = days_elapsed
        self.total_days = total_days
        self.days_remaining = days_remaining or max(0, total_days - days_elapsed)

    def as_dict(self) -> dict:
        return {
            "days_elapsed": self.days_elapsed,
            "total_days": self.total_days,
            "days_remaining": self.days_remaining,
        }


class FormulaEngine:
    """
    Expands a formula definition into a concrete SQL expression.

    Steps:
    1. Look up the Formula by name from the registry.
    2. Resolve each component metric name → db_column.
    3. Substitute all {placeholder} tokens in the expression template.
    4. Inject runtime_vars (days_elapsed, total_days, days_remaining) from QueryContext.
    5. Return the final SQL expression string (used directly in SELECT).
    """

    def __init__(self, registry: MetadataRegistry):
        self.registry = registry

    def expand(self, formula_name: str, context: QueryContext | None = None, table: str | None = None) -> str | None:
        formula = self.registry.get_formula(formula_name)
        if not formula:
            logger.warning(f"Formula not found: {formula_name}")
            return None

        substitutions = self._build_substitutions(formula, context, table)
        try:
            expr = formula.expression.format(**substitutions)
        except KeyError as e:
            logger.error(f"Formula {formula_name} missing substitution key: {e}")
            return None

        # Window formulas wrap the inner expression with SUM(...) OVER () to
        # compute a grand total alongside per-row aggregates.
        if formula.window:
            expr = f"SUM({expr}) OVER ()"

        return expr

    def get_display_label(self, formula_name: str) -> str:
        formula = self.registry.get_formula(formula_name)
        return formula.display if formula else formula_name

    def get_component_columns(self, formula_name: str) -> list[str]:
        """Return the db_column names for all components of a formula."""
        formula = self.registry.get_formula(formula_name)
        if not formula:
            return []
        cols = []
        for component in formula.components:
            col = self.registry.get_metric_column(component)
            if col:
                cols.append(col)
        return cols

    def is_window_formula(self, formula_name: str) -> bool:
        formula = self.registry.get_formula(formula_name)
        return formula.window if formula else False

    # ── Private ──────────────────────────────────────────────────────────────

    def _build_substitutions(self, formula: Formula, context: QueryContext | None, table: str | None = None) -> dict:
        subs = {}

        # Substitute component metric names → db_columns
        for component in formula.components:
            col = self.registry.get_metric_column(component)
            if col:
                subs[col] = col       # {cymtd} → cymtd (column name as-is)
                subs[component] = col  # {cymtd} if name differs from column

        # Inject table name (used by SQL-based runtime expressions like CRR/RRR)
        if table:
            subs["table"] = table

        # Substitute runtime variables from QueryContext
        if formula.runtime_vars and context:
            subs.update(context.as_dict())
        elif formula.runtime_vars:
            # Fallback safe values if no context provided
            subs.update({"days_elapsed": 1, "total_days": 30, "days_remaining": 29})

        return subs
