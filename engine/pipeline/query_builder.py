"""
Stage 2 — Query Builder
Converts an enriched ParsedIntent into SQL queries.
Pure Python, metadata-driven — no LLM involved.
"""
from __future__ import annotations
import logging
from models.intent import ParsedIntent
from models.query import BuiltQuery
from engine.query.builder import QueryBuilder
from engine.rules.engine import BusinessRulesEngine

logger = logging.getLogger(__name__)


class QueryBuilderStage:
    """
    Stage 2: ParsedIntent (enriched) → list[BuiltQuery]

    Flow:
        enriched intent
          → BusinessRulesEngine.enrich() applies column/formula/sort rules
          → QueryBuilder.build() routes to tables and assembles SQL
          → Returns list of BuiltQuery (one per resolved table)
    """

    def __init__(self, rules_engine: BusinessRulesEngine, query_builder: QueryBuilder):
        self.rules_engine = rules_engine
        self.query_builder = query_builder

    def build(self, intent: ParsedIntent, question: str) -> list[BuiltQuery]:
        # Snapshot question-requested metrics before business rules add context columns
        question_metrics = list(intent.metrics) + list(intent.formula_metrics)
        enriched = self.rules_engine.enrich(intent, question)
        if not enriched.display_metrics:
            enriched.display_metrics = question_metrics
        logger.debug(f"Rules applied: {enriched.applied_rules}")

        # Expand virtual hierarchy filters (e.g. sales_person="Anuj" → search
        # across all concrete hierarchy levels: zsm, rsm, asm, so).
        # Only applied when there are NO concrete group-by dimensions — when
        # dimensions are present, _build_where already OR-expands virtual filters
        # across hierarchy columns within each table, which is sufficient.
        filter_expanded = self._expand_hierarchy_filters(enriched) if not enriched.dimensions else []
        if filter_expanded:
            all_queries: list[BuiltQuery] = []
            for level_name, level_intent in filter_expanded:
                for q in self.query_builder.build(level_intent):
                    q.label = level_name
                    all_queries.append(q)
            if not all_queries:
                logger.warning("Stage 2: no queries built after hierarchy filter expansion.")
            return all_queries

        # Expand virtual hierarchy dimensions into separate query groups.
        # A virtual dimension (db_column=None) with a hierarchy_name (e.g.
        # sales_person → sales hierarchy) is replaced by each concrete sibling
        # in that hierarchy (zsm, rsm, asm, so), generating one labeled query
        # group per level so the result is presented grouped by hierarchy level.
        expanded = self._expand_hierarchy_dimensions(enriched)
        if expanded:
            all_queries: list[BuiltQuery] = []
            for level_name, level_intent in expanded:
                for q in self.query_builder.build(level_intent):
                    q.label = level_name
                    all_queries.append(q)
            if not all_queries:
                logger.warning("Stage 2: no queries built after hierarchy expansion.")
            return all_queries

        queries = self.query_builder.build(enriched)
        if not queries:
            logger.warning("Stage 2: no queries built from intent.")
        return queries

    # ── Private ──────────────────────────────────────────────────────────────

    def _expand_hierarchy_dimensions(
        self, intent: ParsedIntent
    ) -> list[tuple[str, ParsedIntent]]:
        """
        For each virtual dimension (db_column=None, hierarchy_name set) in
        intent.dimensions, return one (level_name, modified_intent) pair per
        concrete dimension in that hierarchy.  Non-virtual dimensions are left
        untouched.  Returns [] when no expansion is needed.
        """
        registry = self.query_builder.registry
        virtual_dims = [
            d for d in intent.dimensions
            if registry.get_db_column(d) is None
            and registry.get_dimension(d) is not None
            and registry.get_dimension(d).hierarchy_name
        ]
        if not virtual_dims:
            return []

        result: list[tuple[str, ParsedIntent]] = []
        for vdim in virtual_dims:
            hierarchy = registry.get_dimension(vdim).hierarchy_name
            concrete = registry.get_dimensions_by_hierarchy(hierarchy)
            for cdim in concrete:
                new_dims = [cdim if d == vdim else d for d in intent.dimensions]
                expanded_intent = intent.model_copy(update={"dimensions": new_dims})
                result.append((cdim, expanded_intent))
        return result

    def _expand_hierarchy_filters(
        self, intent: ParsedIntent
    ) -> list[tuple[str, ParsedIntent]]:
        """
        For each virtual dimension used as a filter (e.g. sales_person="Anuj"),
        return one (level_name, modified_intent) pair per concrete hierarchy level,
        replacing the virtual filter key with the concrete dimension key.

        Example: filters={sales_person: "Anuj"} → 4 pairs:
          ("zsm", intent with filters={zsm: "Anuj"}),
          ("rsm", intent with filters={rsm: "Anuj"}),
          ("asm", intent with filters={asm: "Anuj"}),
          ("so",  intent with filters={so:  "Anuj"})
        Returns [] when no virtual filter keys are found.
        """
        registry = self.query_builder.registry
        virtual_filter_keys = [
            dim_name for dim_name in intent.filters
            if registry.get_db_column(dim_name) is None
            and registry.get_dimension(dim_name) is not None
            and registry.get_dimension(dim_name).hierarchy_name
        ]
        if not virtual_filter_keys:
            return []

        result: list[tuple[str, ParsedIntent]] = []
        for vdim in virtual_filter_keys:
            hierarchy = registry.get_dimension(vdim).hierarchy_name
            concrete = registry.get_dimensions_by_hierarchy(hierarchy)
            value = intent.filters[vdim]
            for cdim in concrete:
                new_filters = {cdim if k == vdim else k: v for k, v in intent.filters.items()}
                expanded_intent = intent.model_copy(update={"filters": new_filters})
                result.append((cdim, expanded_intent))
        return result
