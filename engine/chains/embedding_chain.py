"""
EmbeddingChain — generic embedding-based question answering chain.

Implements the same .answer(question, verbose=False) -> ChatResponse interface
as ChatbotChain, enabling both SQL and embedding plugins to be used identically.

All domain knowledge is injected via plugin config dicts at construction time.
"""
from __future__ import annotations
import asyncio
import concurrent.futures
import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.query import ChatResponse

logger = logging.getLogger(__name__)

# Root of the prismage-data-chat-engine repo (two levels up from this file)
_REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
_DYNAMIC_EMBEDDINGS_SRC = _REPO_ROOT / 'dynamic-embeddings' / 'src'
_DYNAMIC_EMBEDDINGS_ENV = _REPO_ROOT / 'dynamic-embeddings' / '.env'


def _ensure_dynamic_embeddings_on_path() -> None:
    """Add dynamic-embeddings/src to sys.path if not already present."""
    src = str(_DYNAMIC_EMBEDDINGS_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def _load_env_file(env_path: Path) -> None:
    """Load key=value pairs from an .env file into os.environ (no-op if missing)."""
    if not env_path.exists():
        return
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())


class EmbeddingChain:
    """
    Embedding-based question answering chain.

    Constructed by PluginLoader from plugin manifest + config files.
    Uses engine/embedding/ components (Searcher, Analyzer, Synthesizer, etc.)
    wired together with plugin-injected configuration.
    """

    def __init__(
        self,
        namespace: str,
        plugin_dir: str,
        schema_config: Dict,
        prompts_config: Dict,
        llm_model: str = 'gpt-4o-mini',
        enable_charts: bool = False,
    ):
        self.namespace = namespace
        self.plugin_dir = Path(plugin_dir).resolve()
        self.schema_config = schema_config
        self.prompts_config = prompts_config
        self.llm_model = llm_model
        self.enable_charts = enable_charts

        # Bootstrap environment + imports
        _load_env_file(_DYNAMIC_EMBEDDINGS_ENV)
        _ensure_dynamic_embeddings_on_path()

        # Load metric definitions from plugin config
        self._metrics: Dict[str, Dict] = self._load_metrics()
        self._metric_names: List[str] = sorted(self._metrics.keys())

        # Build granularity names map from schema
        self._granularity_names: Dict[str, str] = schema_config.get('granularity_names', {
            'qoq': 'Quarter Over Quarter', 'qtd': 'Quarter To Date',
            'mom': 'Month Over Month',     'mtd': 'Month To Date',
            'wow': 'Week Over Week',       'wtd': 'Week To Date',
            'dod': 'Day Over Day',
        })

        # Build engine components (lazy import to keep top-level import clean)
        self._build_components()

        logger.info(
            "EmbeddingChain ready: namespace=%s plugin=%s metrics=%d",
            namespace, self.plugin_dir.name, len(self._metric_names),
        )

    # ── Public interface ──────────────────────────────────────────────────

    def answer(self, question: str, verbose: bool = False) -> ChatResponse:
        """
        Answer a question using embedding-based search.
        Returns ChatResponse with answer/summary/detail populated.
        sql_queries and query_results are always empty.
        """
        t_start = time.time()
        try:
            result = self._run_async(question, verbose)
            elapsed = time.time() - t_start
            answer_text, filtered_collections, dimension = result

            summary, detail = self._split_answer(answer_text)
            return ChatResponse(
                question=question,
                answer=answer_text,
                summary=summary,
                detail=detail,
                success=True,
                step_timings={'total_s': round(elapsed, 2)},
            )

        except Exception as exc:
            logger.exception("EmbeddingChain error: %s", exc)
            return ChatResponse(
                question=question,
                answer="An internal error occurred. Please try again.",
                success=False,
                error=str(exc),
            )

    # ── Internal pipeline ─────────────────────────────────────────────────

    def _run_async(self, question: str, verbose: bool):
        """Run the async pipeline, handling both sync and async call contexts."""
        coro = self._pipeline(question, verbose)
        try:
            loop = asyncio.get_running_loop()
            # Already inside an event loop — run in thread pool
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)

    async def _pipeline(self, question: str, verbose: bool):
        """Full 5-step pipeline: parse → validate → find → search+analyze → synthesize."""
        from dynamic_embeddings.pipelines.embedding_pipeline import EmbeddingPipeline
        from dynamic_embeddings.database.connection import DatabaseConnection
        from dynamic_embeddings.services.llm_service import LLMService

        current_date = datetime.now().strftime('%Y%m%d')

        # Step 1: Parse question
        parsed = self._question_parser.parse(question, self._llm_service)
        if not parsed['success']:
            return parsed.get('error', 'Failed to analyze question'), {}, ''

        # Step 2: Validate + normalize metrics
        validation = self._question_parser.validate(
            parsed, normalize_fn=self._normalize_metric
        )
        if not validation['valid']:
            feedback = self._build_rejection_feedback(validation)
            return feedback, {}, ''

        requested_metrics = validation['metrics']['normalized']

        # Step 3: Extract date range
        date_info = self._question_parser.get_date_range(parsed, current_date)
        start_date = date_info['start_date']
        end_date = date_info['end_date']

        # Step 4: Select dimension
        dimension = self._collection_finder.select_dimension(
            parsed.get('dimension_combination'),
            validation['dimensions']['values'],
        )

        # Step 4b: Dimension availability + fallback
        target_grans = validation['granularities']['normalized'] or None

        db_connection = DatabaseConnection()
        embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        pipeline = EmbeddingPipeline(
            database_connection=db_connection,
            namespace=self.namespace,
            embedding_model=embedding_model,
        )

        try:
            metadata_service = self._get_metadata_service(pipeline)
        except Exception:
            metadata_service = None

        available_dims = self._collection_finder.get_available_dimensions(
            metadata_service, pipeline
        )
        dimension, dim_note = self._collection_finder.apply_fallback(dimension, available_dims)
        if dim_note and verbose:
            logger.info(dim_note)

        # Step 5: Find collections
        dim_values = self._collection_finder.extract_dimension_values_for_search(
            dimension, parsed
        )

        if metadata_service:
            collections_by_gran = self._collection_finder.find_fast(
                metadata_service, dimension,
                [g.lower() for g in target_grans] if target_grans else None,
                start_date, end_date, dim_values,
            )
        else:
            collections_by_gran = self._collection_finder.find_slow(
                pipeline, dimension,
                [g.lower() for g in target_grans] if target_grans else None,
                start_date, end_date,
            )

        if not collections_by_gran:
            return (
                f"No collections found for dimension '{dimension}'. "
                f"Available: {', '.join(sorted(available_dims)) or 'none'}",
                {}, dimension,
            )

        # Step 6: Search + Analyze + Synthesize (orchestrator)
        filters = {'dimension_value': dim_values} if dim_values else None
        specified_dates = None
        if date_info.get('status') == 'valid':
            raw_text = parsed.get('date_range', {}).get('raw_text', '')
            specified_dates = {
                'raw_text': raw_text,
                'start_date': start_date,
                'end_date': end_date,
            }
        answer, filtered = await self._orchestrator.run(
            pipeline=pipeline,
            llm_service=self._llm_service,
            collections_by_granularity=collections_by_gran,
            question=question,
            dimension=dimension,
            requested_metrics=requested_metrics,
            filters=filters,
            target_granularities=target_grans,
            specified_dates=specified_dates,
        )

        try:
            pipeline.close()
        except Exception:
            pass

        return answer, filtered or {}, dimension

    # ── Component wiring ──────────────────────────────────────────────────

    def _build_components(self) -> None:
        from engine.embedding.question_parser import QuestionParser
        from engine.embedding.collection_finder import CollectionFinder
        from engine.embedding.searcher import Searcher
        from engine.embedding.analyzer import Analyzer
        from engine.embedding.synthesizer import Synthesizer
        from engine.embedding.orchestrator import Orchestrator

        _ensure_dynamic_embeddings_on_path()
        from dynamic_embeddings.services.llm_service import LLMService

        self._llm_service = LLMService(model=self.llm_model)

        self._question_parser = QuestionParser(self.schema_config, self.prompts_config)
        self._question_parser.set_metrics(self._metric_names)

        self._collection_finder = CollectionFinder(self.schema_config)

        search_cfg = self.schema_config.get('search', {})
        self._searcher = Searcher(search_cfg)
        self._analyzer = Analyzer(self.prompts_config, self._granularity_names)
        self._synthesizer = Synthesizer(self.prompts_config, self._granularity_names)

        gran_order = self.schema_config.get('time_granularities', [])
        self._orchestrator = Orchestrator(
            searcher=self._searcher,
            analyzer=self._analyzer,
            synthesizer=self._synthesizer,
            search_config=search_cfg,
            granularity_order=gran_order,
        )

    # ── Metric helpers ────────────────────────────────────────────────────

    def _load_metrics(self) -> Dict[str, Dict]:
        """Load metric definitions from plugin's kpi_metrics.csv."""
        metrics_file = self.schema_config.get('metrics_file', 'config/kpi_metrics.csv')
        csv_path = self.plugin_dir / metrics_file
        if not csv_path.exists():
            logger.warning("kpi_metrics.csv not found at %s", csv_path)
            return {}
        metrics = {}
        try:
            with open(csv_path, newline='') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    name = row.get('Metric_Name', '').strip()
                    mtype = row.get('Metric_Type', '').strip()
                    formula = row.get('Metric_Formula', '').strip()
                    if name and mtype:
                        metrics[name] = {'type': mtype, 'formula': formula}
        except Exception as exc:
            logger.error("Error loading kpi_metrics.csv: %s", exc)
        return metrics

    def _normalize_metric(self, metric_name: str) -> Optional[str]:
        """Normalize a metric name to canonical form (case-insensitive, partial match)."""
        lower = metric_name.lower()
        for known in self._metrics:
            if lower == known.lower():
                return known
        for known in self._metrics:
            if lower in known.lower() or known.lower() in lower:
                return known
        return None

    # ── Metadata service ──────────────────────────────────────────────────

    def _get_metadata_service(self, pipeline):
        from dynamic_embeddings.services.collection_metadata_service import CollectionMetadataService
        svc = CollectionMetadataService(pipeline.db_connection, namespace=self.namespace)
        # Quick existence check
        with pipeline.db_connection.suppress_errors():
            with pipeline.db_connection.get_session() as session:
                session.query(svc.MetadataModel).limit(1).first()
        return svc

    # ── Rejection feedback ────────────────────────────────────────────────

    def _build_rejection_feedback(self, validation: Dict) -> str:
        issues = []
        if validation['metrics']['missing']:
            issues.append("No metrics found in your question. Please specify at least one metric.")
        elif validation['metrics'].get('invalid'):
            invalid = ', '.join(validation['metrics']['invalid'])
            issues.append(f"Unrecognized metric(s): {invalid}")
            if self._metric_names:
                sample = ', '.join(self._metric_names[:10])
                issues.append(f"Supported metrics include: {sample}{'...' if len(self._metric_names) > 10 else ''}")
        return "\n".join(issues) if issues else "Unable to process your question."

    # ── Answer formatting ─────────────────────────────────────────────────

    @staticmethod
    def _split_answer(answer: str):
        """Split answer into (summary, detail) — first paragraph vs rest."""
        paragraphs = [p.strip() for p in answer.strip().split('\n\n') if p.strip()]
        if not paragraphs:
            return answer, answer
        summary = paragraphs[0]
        detail = '\n\n'.join(paragraphs[1:]) if len(paragraphs) > 1 else answer
        return summary, detail
