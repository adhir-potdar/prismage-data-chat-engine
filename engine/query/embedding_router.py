"""
EmbeddingTableRouter — semantic table routing via MetadataEmbeddingStore.

Drop-in alternative to TableRouter. Uses cosine similarity over embedded table
documents instead of table_affinity set intersection, making it more robust for
questions with unfamiliar phrasing or when table_affinity lists are incomplete.

The affinity-based TableRouter remains the default; this router is opted-in via
router_mode="embedding" in build_engine().
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from engine.metadata.embedding_store import MetadataEmbeddingStore
from engine.metadata.registry import MetadataRegistry
from engine.query.router import TableGroup
from models.intent import ParsedIntent

if TYPE_CHECKING:
    from engine.capabilities.base import EngineCapabilities


class EmbeddingTableRouter:
    """
    Resolves tables by semantic similarity between a query derived from the
    ParsedIntent and the embedded table metadata documents.

    The query is constructed from intent dimensions + metrics + formula_metrics.
    Channel filtering (from business rules) is applied after similarity ranking.
    An optional similarity threshold (from capabilities) drops tables that score
    significantly below the top result, preventing over-routing to irrelevant tables.
    """

    def __init__(
        self,
        store: MetadataEmbeddingStore,
        registry: MetadataRegistry,
        capabilities: "EngineCapabilities | None" = None,
    ):
        self._store = store
        self._registry = registry
        self._capabilities = capabilities

    def resolve(self, intent: ParsedIntent) -> list[TableGroup]:
        query = self._build_query(intent)
        scored = self._store.find_tables_scored(query)

        if not scored:
            return []

        candidates = [t for t, _ in scored]

        # ── Step 1: Channel filter ────────────────────────────────────────────
        # Restrict to channels specified by business rules (e.g. primary+secondary).
        if intent.channels:
            filtered = [
                t for t in candidates
                if (tbl := self._registry.get_table(t)) and tbl.channel in intent.channels
            ]
            candidates = filtered or candidates

        # ── Step 2: Dimension/metric relevance filter ─────────────────────────
        # When BOTH dimensions and metrics are requested, require the table to
        # support at least one of each (AND logic). This prevents tables that share
        # metrics but lack the requested grouping dimension from being included
        # (e.g. dashboard tables passing on metrics alone for a ZSM-grouped query).
        # When only one of dims/metrics is specified, use OR (permissive) logic.
        requested_dims = set(intent.dimensions)
        requested_metrics = set(intent.metrics + intent.formula_metrics)
        if requested_dims and requested_metrics:
            relevant = [
                t for t in candidates
                if (
                    any(self._registry.table_has_dimension(t, d) for d in requested_dims)
                    and any(self._registry.table_has_metric(t, m) for m in requested_metrics)
                )
            ]
            candidates = relevant or candidates
        elif requested_dims or requested_metrics:
            relevant = [
                t for t in candidates
                if (
                    any(self._registry.table_has_dimension(t, d) for d in requested_dims)
                    or any(self._registry.table_has_metric(t, m) for m in requested_metrics)
                )
            ]
            candidates = relevant or candidates

        # ── Step 3: No-dimension default-table narrowing ─────────────────────
        # When no (real) dimension is in the intent, all metric-bearing tables
        # pass the relevance check above.  Restrict to the plugin-configured
        # default table set (e.g. summary/dashboard tables) so a bare aggregate
        # question does not fan out to all table types simultaneously.
        # Note: virtual dimensions (db_column=None) are expanded to concrete
        # hierarchy siblings in QueryBuilderStage before routing, so by the time
        # this router runs, dimensions[] only contains real dimensions or is empty.
        # Exception: when filter dimensions have specific table affinities (e.g.
        # sales_person filters route to sales_rep tables), prefer those tables.
        no_real_dims = not intent.dimensions
        if no_real_dims:
            filter_affinities: set[str] = set()
            for dim_name in intent.filters:
                d = self._registry.get_dimension(dim_name)
                if d and d.table_affinity:
                    filter_affinities.update(d.table_affinity)
            if filter_affinities:
                filtered = [t for t in candidates if t in filter_affinities]
                candidates = filtered or candidates
            elif self._capabilities:
                defaults = self._capabilities.get_default_tables()
                if defaults:
                    default_set = set(defaults)
                    filtered = [t for t in candidates if t in default_set]
                    candidates = filtered or candidates

        # ── Step 4: Filter-dimension narrowing ───────────────────────────────
        # If the intent has WHERE filters on specific dimensions (e.g. asm=Anuj),
        # prefer tables that actually carry those columns. Tables missing ALL filter
        # dimensions would silently drop the filter, producing unfiltered results.
        if intent.filters:
            filter_dims = set(intent.filters.keys())
            with_filter_dims = [
                t for t in candidates
                if any(self._registry.table_has_dimension(t, d) for d in filter_dims)
            ]
            if with_filter_dims:
                candidates = with_filter_dims
            # If no table has any filter dimension, leave candidates unchanged
            # (graceful fallback — filter will be silently ignored).

        # ── Step 5: Metric variant filter ────────────────────────────────────
        # If an explicit variant was requested, keep only matching tables.
        # If no variant was requested and no effective join-key dimensions exist
        # (empty or all-virtual), apply the plugin default variant to avoid merging
        # incompatible table types within the same channel without join keys.
        if intent.metric_variant:
            filtered = [
                t for t in candidates
                if (tbl := self._registry.get_table(t)) and tbl.variant == intent.metric_variant
            ]
            candidates = filtered or candidates
        elif no_real_dims and self._capabilities:  # no dimensions at all
            default_variant = self._capabilities.get_default_table_variant()
            if default_variant:
                filtered = [
                    t for t in candidates
                    if (tbl := self._registry.get_table(t)) and tbl.variant == default_variant
                ]
                candidates = filtered or candidates

        # ── Step 6: Similarity threshold ─────────────────────────────────────
        # Drop tables scoring significantly below the best remaining candidate.
        # Threshold is relative (e.g. 0.90 = keep tables within 10% of top score).
        # Applied last so earlier structural filters determine the reference table,
        # preventing high-scoring but structurally wrong tables from raising the bar.
        threshold = self._capabilities.get_embedding_threshold() if self._capabilities else 0.0
        if threshold > 0.0 and candidates:
            score_map = dict(scored)
            top_score = max(score_map.get(t, 0.0) for t in candidates)
            min_score = top_score * threshold
            filtered_by_score = [t for t in candidates if score_map.get(t, 0.0) >= min_score]
            candidates = filtered_by_score or candidates

        return [
            TableGroup(
                table=t,
                channel=self._registry.get_table(t).channel
                if self._registry.get_table(t) else "unknown",
                label=t,
            )
            for t in candidates
        ]

    def can_resolve(self, intent: ParsedIntent) -> bool:
        return len(self.resolve(intent)) > 0

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_query(self, intent: ParsedIntent) -> str:
        terms = intent.dimensions + intent.metrics + intent.formula_metrics
        return " ".join(terms) if terms else "general data query"
