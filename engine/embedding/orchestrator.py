"""
3-phase async orchestrator for embedding-based search and analysis.

Phase 1: Parallel vector search across all granularities
Phase 2: Global top-K filtering + granularity distribution
Phase 3: Parallel batch analysis → optional per-granularity synthesis → collective synthesis

All configuration injected; no domain knowledge here.
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

from engine.embedding.searcher import Searcher, SearchResult, AnalysisResult
from engine.embedding.analyzer import Analyzer
from engine.embedding.synthesizer import Synthesizer

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Wires Searcher → Analyzer → Synthesizer into a 3-phase async pipeline.

    search_config keys used here:
        max_results_per_batch          (int, default 2)
        top_k_global                   (int, default 5)
        max_granularities_for_top_results (int, default 3)
        enable_hybrid_parallelization  (bool, default True)
        enable_per_granularity_synthesis (bool, default False)
    """

    def __init__(
        self,
        searcher: Searcher,
        analyzer: Analyzer,
        synthesizer: Synthesizer,
        search_config: Dict,
        granularity_order: List[str],
    ):
        self.searcher = searcher
        self.analyzer = analyzer
        self.synthesizer = synthesizer
        self.max_results_per_batch: int = search_config.get('max_results_per_batch', 2)
        self.top_k: int = search_config.get('top_k_global', 5)
        self.max_grans_for_top: int = search_config.get('max_granularities_for_top_results', 3)
        self.hybrid: bool = search_config.get('enable_hybrid_parallelization', True)
        self.per_gran_synthesis: bool = search_config.get('enable_per_granularity_synthesis', False)
        self.granularity_order = [g.lower() for g in granularity_order]

    async def run(
        self,
        pipeline,
        llm_service,
        collections_by_granularity: Dict[str, List[str]],
        question: str,
        dimension: str,
        requested_metrics: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        target_granularities: Optional[List[str]] = None,
        specified_dates: Optional[Dict] = None,
    ) -> Tuple[str, Optional[Dict]]:
        """
        Execute full pipeline. Returns (answer_text, filtered_collections).
        filtered_collections is the subset of collections that yielded results.
        """
        # Apply granularity filter
        if target_granularities:
            tg_lower = {g.lower() for g in target_granularities}
            working = {k: v for k, v in collections_by_granularity.items()
                       if k.lower() in tg_lower}
            if not working:
                logger.warning("No collections matched target granularities — using all")
                working = collections_by_granularity
        else:
            working = collections_by_granularity

        # ── Phase 1: Parallel search ──────────────────────────────────────
        search_semaphore = self.searcher.make_semaphore()
        analysis_semaphore = asyncio.Semaphore(5)

        search_tasks = [
            asyncio.create_task(
                self.searcher.search_granularity(
                    pipeline, gran, colls, question, search_semaphore, filters
                )
            )
            for gran in self.granularity_order
            if gran in working
            for colls in [working[gran]]
        ]

        if not search_tasks:
            return "No relevant information found.", None

        t_search = time.time()
        search_results_raw = await asyncio.gather(*search_tasks, return_exceptions=True)
        logger.info("All searches completed in %.2fs", time.time() - t_search)

        # Flatten results with metadata
        all_results = []
        for sr in search_results_raw:
            if isinstance(sr, Exception) or (isinstance(sr, SearchResult) and sr.error):
                continue
            if isinstance(sr, SearchResult) and sr.results:
                for record, similarity in sr.results:
                    all_results.append({
                        'granularity': sr.granularity,
                        'record': record,
                        'similarity': similarity,
                    })

        if not all_results:
            return "No relevant information found to analyze.", None

        # ── Phase 2: Global top-K with granularity distribution ───────────
        top_results = self._distribute_top_k(all_results)

        # Group back by granularity
        results_by_gran: Dict[str, List] = {}
        for r in top_results:
            results_by_gran.setdefault(r['granularity'], []).append(
                (r['record'], r['similarity'])
            )

        filtered_collections = {
            gran: working[gran] for gran in results_by_gran if gran in working
        }

        # ── Phase 3: Analysis + Synthesis ────────────────────────────────
        if self.hybrid:
            analysis_results = await self._hybrid_analysis(
                llm_service, results_by_gran, question, dimension,
                requested_metrics, analysis_semaphore
            )
        else:
            analysis_results = await self._standard_analysis(
                llm_service, results_by_gran, working, question, dimension,
                requested_metrics, analysis_semaphore
            )

        successful = [r for r in analysis_results
                      if isinstance(r, AnalysisResult) and r.insights and not r.error]
        if not successful:
            return "Failed to extract insights from any time granularities.", None

        answer = await self.synthesizer.synthesize_collective(
            llm_service, successful, question, dimension, requested_metrics,
            filtered_collections=filtered_collections,
            specified_dates=specified_dates,
        )
        return answer, filtered_collections

    # ── Phase 2 helpers ───────────────────────────────────────────────────

    def _distribute_top_k(self, all_results: List[Dict]) -> List[Dict]:
        """Select top-K results distributed across available granularities."""
        by_gran: Dict[str, List[Dict]] = {}
        for r in all_results:
            by_gran.setdefault(r['granularity'], []).append(r)

        for g in by_gran:
            by_gran[g].sort(key=lambda x: x['similarity'], reverse=True)

        # Rank granularities by (count DESC, max_similarity DESC, name ASC for determinism)
        priority = sorted(
            by_gran.items(),
            key=lambda kv: (-len(kv[1]), -max(r['similarity'] for r in kv[1]), kv[0]),
        )

        top: List[Dict] = []
        target_grans = min(self.max_grans_for_top, len(priority))

        # Phase 1: 1 from each target granularity
        if target_grans >= 2:
            for gran, results in priority[:target_grans]:
                if results:
                    top.append(results[0])
            # Phase 2: fill remaining quota
            quota = self.top_k - len(top)
            allocated_ids = {r['record'].chunk_id for r in top}
            for gran, results in priority:
                if quota <= 0:
                    break
                for r in results:
                    if quota <= 0:
                        break
                    if r['record'].chunk_id not in allocated_ids:
                        top.append(r)
                        allocated_ids.add(r['record'].chunk_id)
                        quota -= 1
        else:
            all_results_sorted = sorted(all_results, key=lambda x: x['similarity'], reverse=True)
            top = all_results_sorted[:self.top_k]

        return top

    # ── Phase 3 helpers ───────────────────────────────────────────────────

    async def _hybrid_analysis(
        self,
        llm_service,
        results_by_gran: Dict,
        question: str,
        dimension: str,
        requested_metrics: Optional[List[str]],
        semaphore: asyncio.Semaphore,
    ) -> List[AnalysisResult]:
        """Split each granularity's results into max_results_per_batch batches, analyze in parallel."""
        batch_tasks = []
        batch_gran_map: List[str] = []

        for gran, results in results_by_gran.items():
            n = len(results)
            num_batches = (n + self.max_results_per_batch - 1) // self.max_results_per_batch
            for b_idx in range(num_batches):
                batch = results[b_idx * self.max_results_per_batch:(b_idx + 1) * self.max_results_per_batch]
                batch_tasks.append(asyncio.create_task(
                    self.analyzer.analyze_batch(
                        llm_service, gran, batch, b_idx + 1,
                        question, dimension, requested_metrics, semaphore
                    )
                ))
                batch_gran_map.append(gran)

        if not batch_tasks:
            return []

        raw_batches = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Group by granularity
        batches_by_gran: Dict[str, List[Dict]] = {}
        for batch_result in raw_batches:
            if isinstance(batch_result, Exception):
                continue
            batches_by_gran.setdefault(batch_result.get('granularity'), []).append(batch_result)

        if self.per_gran_synthesis:
            # Synthesize each granularity's batches
            syn_tasks = [
                asyncio.create_task(
                    self.synthesizer.synthesize_batches(
                        llm_service, gran, batches, question, dimension, requested_metrics
                    )
                )
                for gran, batches in batches_by_gran.items()
                if any(b.get('insights') for b in batches)
            ]
            return list(await asyncio.gather(*syn_tasks, return_exceptions=True))
        else:
            # Combine batches directly into AnalysisResult per granularity
            results = []
            for gran, batches in batches_by_gran.items():
                successful = [b for b in batches if b.get('insights')]
                if successful:
                    combined = "\n\n".join(b['insights'] for b in successful)
                    gran_name = self.analyzer.granularity_names.get(gran, gran.upper())
                    results.append(AnalysisResult(
                        granularity=gran,
                        granularity_name=gran_name,
                        insights=combined,
                        analysis_time=sum(b.get('analysis_time', 0) for b in successful),
                    ))
            return results

    async def _standard_analysis(
        self,
        llm_service,
        results_by_gran: Dict,
        working: Dict,
        question: str,
        dimension: str,
        requested_metrics: Optional[List[str]],
        semaphore: asyncio.Semaphore,
    ) -> List[AnalysisResult]:
        """Original approach: analyze all results per granularity together in one LLM call."""
        tasks = []
        for gran, results in results_by_gran.items():
            sr = SearchResult(
                granularity=gran,
                collections=working.get(gran, []),
                results=results,
                search_time=0.0,
            )
            # Use a single-result batch (batch_index=1) to reuse analyze_batch
            tasks.append(asyncio.create_task(
                self.analyzer.analyze_batch(
                    llm_service, gran, results, 1,
                    question, dimension, requested_metrics, semaphore
                )
            ))
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        analysis_results = []
        for r in raw:
            if isinstance(r, Exception):
                continue
            gran = r.get('granularity', '')
            gran_name = self.analyzer.granularity_names.get(gran, gran.upper())
            analysis_results.append(AnalysisResult(
                granularity=gran,
                granularity_name=gran_name,
                insights=r.get('insights', ''),
                analysis_time=r.get('analysis_time', 0),
                error=r.get('error'),
            ))
        return analysis_results
