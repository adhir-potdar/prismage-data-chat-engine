"""
Generic collection finder for embedding-based search.

Supports two modes:
  - Fast: uses CollectionMetadataService (metadata table in DB)
  - Slow: parses collection names directly from the vector store

All dimension/fallback rules are injected from plugin schema_config.
No domain knowledge hardcoded here.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Set, Tuple

from engine.embedding.date_utils import collection_overlaps_date_range

logger = logging.getLogger(__name__)


class CollectionFinder:
    """
    Finds collections matching dimension + granularity + date range.

    schema_config expected keys:
        dimensions.hierarchy  (list[str]) — most specific to least, e.g. ["property_geo_device", ..., "overall"]
        dimensions.fallback   (dict[str, str]) — e.g. {"property_geo": "property_geo_device"}
        time_granularities    (list[str]) — e.g. ["qoq", "qtd", "mom", ...]
        search.max_collections (int)
    """

    def __init__(self, schema_config: Dict):
        dim_cfg = schema_config.get('dimensions', {})
        self.hierarchy: List[str] = dim_cfg.get('hierarchy', ['overall'])
        self.fallback_rules: Dict[str, str] = dim_cfg.get('fallback', {})
        self.time_granularities: List[str] = [
            g.lower() for g in schema_config.get('time_granularities', [])
        ]
        search_cfg = schema_config.get('search', {})
        self.max_collections: int = search_cfg.get('max_collections', 10)

    # ── Dimension selection ────────────────────────────────────────────────

    def select_dimension(
        self,
        dimension_combination: Optional[str],
        dimension_values: Dict[str, List[str]],
    ) -> str:
        """
        Return most specific dimension string based on LLM-extracted info.

        If dimension_combination was explicitly identified by the LLM, use it.
        Otherwise, infer from which dimension_values are present.
        """
        if dimension_combination and dimension_combination not in ('null', 'None', ''):
            return dimension_combination

        # Infer from which keys have values
        has = {k for k, v in dimension_values.items() if v}

        # Walk hierarchy from most specific to least; pick first that matches
        for dim in self.hierarchy:
            parts = set(dim.split('_'))
            # "overall" never matches partial — only if nothing was mentioned
            if dim == 'overall':
                continue
            if parts.issubset(has):
                return dim

        return 'overall'

    def apply_fallback(
        self,
        preferred: str,
        available: Set[str],
    ) -> Tuple[str, Optional[str]]:
        """
        Return (selected_dimension, explanation_or_None).

        Applies hardcoded-rule fallback first, then similarity-based fallback.
        """
        if preferred in available:
            return preferred, None

        # Try explicit fallback map
        if preferred in self.fallback_rules:
            target = self.fallback_rules[preferred]
            if target in available:
                msg = f"{preferred} not available — using {target} as fallback"
                return target, msg

        # Similarity-based fallback
        if available:
            best, msg = self._similarity_fallback(preferred, available)
            return best, msg

        return preferred, None  # will fail later with helpful error

    def _similarity_fallback(
        self,
        preferred: str,
        available: Set[str],
    ) -> Tuple[str, str]:
        pref_parts = set(preferred.split('_'))
        scored = []
        for dim in available:
            parts = set(dim.split('_'))
            common = len(pref_parts & parts)
            scored.append((dim, common, len(parts)))

        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        best = scored[0][0]
        msg = f"{preferred} not available — using {best} (closest match)"
        return best, msg

    # ── Collection finding ─────────────────────────────────────────────────

    def find_fast(
        self,
        metadata_service,
        dimension: str,
        granularities: Optional[List[str]],
        start_date: Optional[int],
        end_date: Optional[int],
        dimension_values: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Find collections using metadata table (fast path).

        Returns dict mapping granularity -> sorted list of collection names.
        Total collections capped at max_collections with smart distribution.
        """
        if not granularities:
            granularities = metadata_service.get_all_granularities() or self.time_granularities

        all_by_gran: Dict[str, List[str]] = {}
        for gran in granularities:
            matching = metadata_service.find_collections(
                dimension=dimension,
                time_granularity=gran,
                start_date=start_date,
                end_date=end_date,
                dimension_values=dimension_values,
            )
            if matching:
                all_by_gran[gran.lower()] = sorted(m['collection_name'] for m in matching)

        return self._distribute(all_by_gran)

    def find_slow(
        self,
        pipeline,
        dimension: str,
        granularities: Optional[List[str]],
        start_date: Optional[int],
        end_date: Optional[int],
    ) -> Dict[str, List[str]]:
        """
        Find collections by parsing collection names (slow/fallback path).

        Returns dict mapping granularity -> sorted list of collection names.
        """
        search_grans = [g.lower() for g in granularities] if granularities else self.time_granularities

        # Get all available collection names
        try:
            all_collections = pipeline.list_collections() if hasattr(pipeline, 'list_collections') \
                else self._list_via_vector_store(pipeline)
        except Exception as exc:
            logger.error("Could not list collections: %s", exc)
            return {}

        # Filter by dimension prefix + granularity suffix
        all_by_gran: Dict[str, List[str]] = {}
        for coll in all_collections:
            if not coll.startswith(f"{dimension}_"):
                continue
            suffix = coll[len(dimension) + 1:]
            for gran in self.time_granularities:
                if suffix.startswith(f"{gran}_"):
                    if gran not in search_grans:
                        break
                    if start_date and end_date:
                        if not collection_overlaps_date_range(coll, start_date, end_date):
                            break
                    all_by_gran.setdefault(gran, []).append(coll)
                    break

        for gran in all_by_gran:
            all_by_gran[gran] = sorted(all_by_gran[gran])

        return self._distribute(all_by_gran)

    def _list_via_vector_store(self, pipeline) -> List[str]:
        """Extract collection names from vector store via pipeline internals."""
        db = pipeline.db_connection
        vs = pipeline.vector_store
        model = vs.table_factory.get_or_create_model(pipeline.namespace)
        with db.get_session() as session:
            rows = session.query(model.collection_name).distinct().all()
        return [r[0] for r in rows]

    def get_available_dimensions(
        self,
        metadata_service=None,
        pipeline=None,
    ) -> Set[str]:
        """Return set of available dimensions from metadata or collection names."""
        dims: Set[str] = set()
        if metadata_service:
            try:
                dims = set(metadata_service.get_all_dimensions())
                return dims
            except Exception:
                pass

        if pipeline:
            try:
                collections = self._list_via_vector_store(pipeline)
                for coll in collections:
                    for gran in self.time_granularities:
                        idx = coll.find(f'_{gran}_')
                        if idx >= 0:
                            dims.add(coll[:idx])
                            break
            except Exception:
                pass

        return dims

    # ── Smart distribution algorithm ──────────────────────────────────────

    def _distribute(self, all_by_gran: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Cap total collections to max_collections with smart distribution.

        Algorithm (deterministic — all iterations sorted):
          Phase 1: Allocate at least 1 per granularity
          Phase 2: Distribute remaining slots evenly
          Phase 3: Redistribute any leftover slots
        """
        if not all_by_gran:
            return {}

        total_available = sum(len(v) for v in all_by_gran.values())
        if total_available <= self.max_collections:
            return dict(all_by_gran)

        result: Dict[str, List[str]] = {}
        num_grans = len(all_by_gran)
        allocated = 0

        # Phase 1
        for gran, colls in sorted(all_by_gran.items()):
            if allocated >= self.max_collections:
                break
            take = min(1, len(colls))
            result[gran] = colls[-take:] if take else []
            allocated += take

        # Phase 2
        remaining = self.max_collections - allocated
        if remaining > 0:
            per_gran = remaining // num_grans
            leftover = remaining % num_grans
            for idx, (gran, colls) in enumerate(sorted(all_by_gran.items())):
                if remaining <= 0:
                    break
                already = len(result.get(gran, []))
                avail_more = len(colls) - already
                extra = per_gran + (1 if idx < leftover else 0)
                take_extra = min(extra, avail_more, remaining)
                if take_extra > 0:
                    total_take = already + take_extra
                    result[gran] = colls[-total_take:]
                    remaining -= take_extra

        # Phase 3
        remaining = self.max_collections - sum(len(v) for v in result.values())
        while remaining > 0:
            redistributed = False
            for gran, colls in sorted(all_by_gran.items()):
                if remaining <= 0:
                    break
                already = len(result.get(gran, []))
                avail_more = len(colls) - already
                if avail_more > 0:
                    take = min(1, avail_more, remaining)
                    result[gran] = colls[-(already + take):]
                    remaining -= take
                    redistributed = True
            if not redistributed:
                break

        return result

    def extract_dimension_values_for_search(
        self,
        dimension: str,
        extracted_info: Dict,
    ) -> Optional[List[str]]:
        """
        Pull the relevant dimension values from LLM-extracted info for the given dimension.

        Collects all non-empty values from every key in dimension_values — this makes
        the method fully generic across any dimension type (property/geo/device for
        revenue management, or adunit/campaignname/lineitem etc. for campaign management).
        Returns None if nothing was extracted (search all).
        """
        all_dv = extracted_info.get('dimension_values', {})
        values = []
        for v in all_dv.values():
            if isinstance(v, list):
                values.extend(v)
        return values if values else None
