"""
Generic async vector similarity searcher.

Runs semaphore-limited parallel searches across collections using
any EmbeddingPipeline that exposes search_similar().
No domain knowledge — all config injected at construction.
"""
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Container for search results from one granularity."""
    granularity: str
    collections: List[str]
    results: List[Any]          # list of (record, similarity) tuples
    search_time: float
    error: Optional[str] = None


@dataclass
class AnalysisResult:
    """Container for analysis results from one granularity."""
    granularity: str
    granularity_name: str
    insights: str
    analysis_time: float
    error: Optional[str] = None


async def _run_sync_in_executor(func, *args):
    """Run a synchronous function in the default thread pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


class Searcher:
    """
    Performs async vector similarity search across collections.

    Config keys (search_config dict):
        max_concurrent_searches (int, default 5)
        max_results_per_collection (int, default 5)
        similarity_threshold (float, default 0.3)
    """

    def __init__(self, search_config: Dict[str, Any]):
        self.max_concurrent = search_config.get('max_concurrent_searches', 5)
        self.limit = search_config.get('max_results_per_collection', 5)
        self.threshold = search_config.get('similarity_threshold', 0.3)

    async def search_granularity(
        self,
        pipeline,
        granularity: str,
        collections: List[str],
        question: str,
        semaphore: asyncio.Semaphore,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Search all collections for one granularity under a shared semaphore."""
        async with semaphore:
            start = asyncio.get_event_loop().time()
            logger.info("[%s] Searching %d collection(s)", granularity.upper(), len(collections))

            try:
                all_results = []
                for collection in collections:
                    info = await _run_sync_in_executor(
                        pipeline.search_similar,
                        question,
                        collection,
                        self.limit,
                        self.threshold,
                        filters,
                    )
                    if 'error' not in info and info.get('results'):
                        all_results.extend(info['results'])

                elapsed = asyncio.get_event_loop().time() - start
                logger.info("[%s] Found %d result(s) in %.2fs", granularity.upper(), len(all_results), elapsed)
                return SearchResult(granularity=granularity, collections=collections,
                                    results=all_results, search_time=elapsed)

            except Exception as exc:
                elapsed = asyncio.get_event_loop().time() - start
                logger.error("[%s] Search error: %s", granularity.upper(), exc)
                return SearchResult(granularity=granularity, collections=collections,
                                    results=[], search_time=elapsed, error=str(exc))

    def make_semaphore(self) -> asyncio.Semaphore:
        return asyncio.Semaphore(self.max_concurrent)
