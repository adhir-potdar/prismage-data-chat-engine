"""
MetadataRegistry — runtime lookup service over loaded metadata.
Provides alias resolution, table routing helpers, and formula access.
"""
from __future__ import annotations
from models.metadata import Dimension, Metric, Formula, Table, MetricCategory
from engine.metadata.loader import MetadataConfig


class MetadataRegistry:
    """
    Built from a MetadataConfig at engine startup.
    All lookups are O(1) via pre-built index dicts.
    """

    def __init__(self, config: MetadataConfig):
        self._dimensions: dict[str, Dimension] = {d.name: d for d in config.dimensions}
        self._metrics: dict[str, Metric] = {m.name: m for m in config.metrics}
        self._formulas: dict[str, Formula] = {f.name: f for f in config.formulas}
        self._tables: dict[str, Table] = {t.name: t for t in config.tables}

        # Alias → canonical name indexes
        self._dim_alias: dict[str, str] = {}
        for d in config.dimensions:
            for alias in d.aliases:
                self._dim_alias[alias.lower()] = d.name

        self._metric_alias: dict[str, str] = {}
        for m in config.metrics:
            for alias in m.aliases:
                self._metric_alias[alias.lower()] = m.name

    # ── Dimension lookups ────────────────────────────────────────────────────

    def has_dimension(self, name: str) -> bool:
        return name in self._dimensions

    def get_dimension(self, name: str) -> Dimension | None:
        return self._dimensions.get(name)

    def resolve_dimension_alias(self, phrase: str) -> str | None:
        return self._dim_alias.get(phrase.lower())

    def get_db_column(self, dimension_name: str) -> str | None:
        d = self._dimensions.get(dimension_name)
        return d.db_column if d else None

    def get_dimensions_by_hierarchy(self, hierarchy_name: str) -> list[str]:
        """Return names of all dimensions that belong to the given hierarchy and have a real db_column."""
        return [
            d.name for d in self._dimensions.values()
            if d.hierarchy_name == hierarchy_name and d.db_column
        ]

    # ── Metric lookups ───────────────────────────────────────────────────────

    def has_metric(self, name: str) -> bool:
        return name in self._metrics

    def has_formula(self, name: str) -> bool:
        return name in self._formulas

    def get_metric(self, name: str) -> Metric | None:
        return self._metrics.get(name)

    def get_formula(self, name: str) -> Formula | None:
        return self._formulas.get(name)

    def resolve_metric_alias(self, phrase: str) -> str | None:
        return self._metric_alias.get(phrase.lower())

    def get_metric_column(self, name: str) -> str | None:
        m = self._metrics.get(name)
        return m.db_column if m else None

    def get_aggregate_fn(self, name: str) -> str:
        m = self._metrics.get(name)
        if m and m.aggregate_fn:
            return m.aggregate_fn.value
        return "SUM"

    def get_metric_category(self, name: str) -> MetricCategory | None:
        m = self._metrics.get(name)
        return m.category if m else None

    def register_formula(self, name: str, expression: str,
                          components: list[str], label: str) -> None:
        """Register an inline formula created by a business rule (session-scoped)."""
        self._formulas[name] = Formula(
            name=name, display=label, expression=expression,
            components=components, runtime_vars=[], window=False,
        )

    def override_formula(self, name: str, expression: str,
                          components: list[str], label: str) -> None:
        """Override an existing formula expression for the current request."""
        if name in self._formulas:
            original = self._formulas[name]
            self._formulas[f"__override_{name}"] = original  # keep backup
        self._formulas[name] = Formula(
            name=name, display=label, expression=expression,
            components=components, runtime_vars=[], window=False,
        )

    def restore_formula_overrides(self) -> None:
        """Restore any overridden formulas to their original definitions."""
        to_restore = [k for k in self._formulas if k.startswith("__override_")]
        for key in to_restore:
            original_name = key[len("__override_"):]
            self._formulas[original_name] = self._formulas.pop(key)

    # ── Table lookups ────────────────────────────────────────────────────────

    def get_table(self, name: str) -> Table | None:
        return self._tables.get(name)

    def get_tables_for_channel(self, channel: str) -> list[Table]:
        return [t for t in self._tables.values() if t.channel == channel]

    def get_date_column(self, table_name: str) -> str | None:
        t = self._tables.get(table_name)
        return t.date_column if t else None

    def table_has_dimension(self, table_name: str, dimension: str) -> bool:
        t = self._tables.get(table_name)
        return dimension in t.dimensions if t else False

    def table_has_metric(self, table_name: str, metric: str) -> bool:
        t = self._tables.get(table_name)
        return metric in t.metrics if t else False

    # ── Rendering helpers (for prompt injection) ─────────────────────────────

    def render_dimensions(self) -> str:
        lines = []
        for d in self._dimensions.values():
            aliases = ", ".join(d.aliases[:5])
            if d.db_column:
                col_info = f"column: {d.db_column}"
            elif d.hierarchy_name:
                col_info = f"virtual — groups by full {d.hierarchy_name} hierarchy"
            else:
                col_info = "virtual"
            lines.append(f"  - {d.name} (also: {aliases}) → {col_info}")
        return "\n".join(lines)

    def render_metrics(self) -> str:
        lines = []
        for m in self._metrics.values():
            aliases = ", ".join(m.aliases[:5])
            lines.append(f"  - {m.name} [{m.category.value}] (also: {aliases})")
        return "\n".join(lines)

    def render_formulas(self) -> str:
        lines = []
        for f in self._formulas.values():
            lines.append(f"  - {f.name}: {f.display}")
        return "\n".join(lines)
