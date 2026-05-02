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

    def __init__(self, registry: MetadataRegistry):
        self.registry = registry

    def resolve(self, intent: ParsedIntent) -> list[TableGroup]:
        all_metrics = intent.metrics + intent.formula_metrics
        candidate_tables = self._intersect_affinities(intent.dimensions, all_metrics)

        if not candidate_tables:
            return []

        # Filter by channels if specified by business rules
        if intent.channels:
            filtered = []
            for table_name in candidate_tables:
                table = self.registry.get_table(table_name)
                if table and table.channel in intent.channels:
                    filtered.append(table_name)
            candidate_tables = filtered or candidate_tables  # fallback to all if filter removes everything

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
        affinity_sets = []

        for dim in dimensions:
            d = self.registry.get_dimension(dim)
            if d and d.table_affinity:
                affinity_sets.append(set(d.table_affinity))

        for metric_name in metrics:
            m = self.registry.get_metric(metric_name)
            if m and m.table_affinity:
                affinity_sets.append(set(m.table_affinity))
            # For formula metrics, check component affinities
            f = self.registry.get_formula(metric_name)
            if f:
                for component in f.components:
                    cm = self.registry.get_metric(component)
                    if cm and cm.table_affinity:
                        affinity_sets.append(set(cm.table_affinity))

        if not affinity_sets:
            return []

        common = affinity_sets[0]
        for s in affinity_sets[1:]:
            common = common & s

        return sorted(common)
