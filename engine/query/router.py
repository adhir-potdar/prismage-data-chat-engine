"""
TableRouter — determines which database tables to query based on the enriched
ParsedIntent. Uses table_affinity from dimensions.json and metrics.json.
"""
from __future__ import annotations
from dataclasses import dataclass
from engine.metadata.registry import MetadataRegistry
from models.intent import ParsedIntent


@dataclass
class TableGroup:
    table: str
    channel: str
    label: str


class TableRouter:
    """
    Resolves which tables satisfy ALL requested dimensions and metrics.

    Logic:
    1. Collect table_affinity sets for every dimension and metric in the intent.
    2. Intersect all sets — tables that appear in ALL affinity sets can serve the query.
    3. Filter by requested channels (from business rules or default).
    4. Return one TableGroup per matching table.
    """

    def __init__(self, registry: MetadataRegistry, capabilities=None):
        self.registry = registry
        self._capabilities = capabilities

    def resolve(self, intent: ParsedIntent) -> list[TableGroup]:
        all_metrics = intent.metrics + intent.formula_metrics
        candidate_tables = self._intersect_affinities(intent.dimensions, all_metrics)

        if not candidate_tables:
            return []

        # When no dimension is specified, narrow table candidates:
        # - If filter dimensions (e.g. sales_person) have specific table affinities,
        #   prefer those tables (they carry the filter column and will apply the WHERE).
        # - Otherwise fall back to plugin-configured default tables.
        no_real_dims = not intent.dimensions
        if no_real_dims:
            filter_affinities: set[str] = set()
            for dim_name in intent.filters:
                d = self.registry.get_dimension(dim_name)
                if d and d.table_affinity:
                    filter_affinities.update(d.table_affinity)
            if filter_affinities:
                filtered = [t for t in candidate_tables if t in filter_affinities]
                candidate_tables = filtered or candidate_tables
            elif self._capabilities:
                defaults = self._capabilities.get_default_tables()
                if defaults:
                    default_set = set(defaults)
                    filtered = [t for t in candidate_tables if t in default_set]
                    candidate_tables = filtered or candidate_tables

        # Filter by channel: intent.channel_filter (from question parser) takes
        # precedence; intent.channels (from business rules) is a secondary filter.
        channel_scope = intent.channels[:]
        if intent.channel_filter and intent.channel_filter not in channel_scope:
            channel_scope = [intent.channel_filter]
        if channel_scope:
            filtered = [t for t in candidate_tables
                        if self.registry.get_table(t) and
                           self.registry.get_table(t).channel in channel_scope]
            candidate_tables = filtered or candidate_tables

        # Filter by table variant (e.g. "value" or "volume") when explicitly
        # specified — keeps only tables whose variant field matches.
        # When no variant requested and no dimension grouped, apply the plugin
        # default variant to avoid merging incompatible types without join keys.
        if intent.metric_variant:
            filtered = [t for t in candidate_tables
                        if self.registry.get_table(t) and
                           self.registry.get_table(t).variant == intent.metric_variant]
            candidate_tables = filtered or candidate_tables
        elif no_real_dims and self._capabilities:  # no dimensions at all
            default_variant = self._capabilities.get_default_table_variant()
            if default_variant:
                filtered = [t for t in candidate_tables
                            if self.registry.get_table(t) and
                               self.registry.get_table(t).variant == default_variant]
                candidate_tables = filtered or candidate_tables

        return [
            TableGroup(
                table=t,
                channel=self.registry.get_table(t).channel if self.registry.get_table(t) else "unknown",
                label=t,
            )
            for t in candidate_tables
        ]

    def can_resolve(self, intent: ParsedIntent) -> bool:
        return len(self.resolve(intent)) > 0

    # ── Private ──────────────────────────────────────────────────────────────

    def _intersect_affinities(self, dimensions: list[str], metrics: list[str]) -> list[str]:
        # First pass: try primary_tables for dimensions (preferred entity tables).
        # If the primary_tables intersection is non-empty, use it.
        # This ensures "top products" routes to product tables, not sales_rep tables.
        primary_sets = []
        for dim in dimensions:
            d = self.registry.get_dimension(dim)
            if d and d.primary_tables:
                primary_sets.append(set(d.primary_tables))

        if primary_sets:
            primary_common = primary_sets[0]
            for s in primary_sets[1:]:
                primary_common = primary_common & s
            if primary_common:
                # Also intersect with metric affinities to ensure tables have the metrics
                metric_sets = self._metric_affinity_sets(metrics)
                if metric_sets:
                    for s in metric_sets:
                        primary_common = primary_common & s
                # Also enforce table_affinity of dims that have no primary_tables —
                # e.g. "category" has no primary_tables but only lives in sales_rep tables.
                # Without this, product_name's primary_tables wins and category is silently dropped.
                for dim in dimensions:
                    d = self.registry.get_dimension(dim)
                    if d and not d.primary_tables and d.table_affinity:
                        primary_common = primary_common & set(d.table_affinity)
                if primary_common:
                    return sorted(primary_common)

        # Fall back to full table_affinity intersection
        affinity_sets = []
        for dim in dimensions:
            d = self.registry.get_dimension(dim)
            if d and d.table_affinity:
                affinity_sets.append(set(d.table_affinity))
        affinity_sets.extend(self._metric_affinity_sets(metrics))

        if not affinity_sets:
            return []

        common = affinity_sets[0]
        for s in affinity_sets[1:]:
            common = common & s

        return sorted(common)

    def _metric_affinity_sets(self, metrics: list[str]) -> list[set]:
        sets = []
        for metric_name in metrics:
            m = self.registry.get_metric(metric_name)
            if m and m.table_affinity:
                sets.append(set(m.table_affinity))
            f = self.registry.get_formula(metric_name)
            if f:
                for component in f.components:
                    cm = self.registry.get_metric(component)
                    if cm and cm.table_affinity:
                        sets.append(set(cm.table_affinity))
        return sets
