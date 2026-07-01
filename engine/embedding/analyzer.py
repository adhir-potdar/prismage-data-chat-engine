"""
Generic LLM-based batch result analyzer for embedding search.

Analyzes 1-N search results (records + similarity scores) for a single
granularity batch and returns key insights as text.

The analysis prompt template is fully injected from the plugin's prompts_config,
keeping this module free of any domain knowledge.
"""
from __future__ import annotations
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from engine.embedding.date_utils import extract_dates_from_collection_name, format_date_readable

logger = logging.getLogger(__name__)


async def _run_sync(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


class Analyzer:
    """
    Analyzes a batch of embedding search results using LLM.

    prompts_config expected keys:
        batch_analysis  (str) — prompt template with placeholders:
            {granularity_name}, {granularity}, {dimension},
            {date_instruction}, {metrics_instruction}, {context}, {question}
    """

    def __init__(self, prompts_config: Dict, granularity_names: Dict[str, str]):
        self.prompt_template: str = prompts_config.get('batch_analysis', '')
        self.granularity_names = granularity_names  # e.g. {"qoq": "Quarter Over Quarter", ...}

    async def analyze_batch(
        self,
        llm_service,
        granularity: str,
        batch_results: List[Any],   # list of (record, similarity) tuples
        batch_index: int,
        question: str,
        dimension: str,
        requested_metrics: Optional[List[str]],
        semaphore: asyncio.Semaphore,
    ) -> Dict:
        """
        Analyze one batch of results. Returns:
            {granularity, batch_index, insights, analysis_time, error}
        """
        async with semaphore:
            start = asyncio.get_event_loop().time()
            try:
                if not batch_results:
                    return self._empty(granularity, batch_index, 'No results in batch')

                # Build context from records
                context_parts = []
                collection_name = None
                for record, similarity in batch_results:
                    context_parts.append(
                        f"From {record.document_id} (similarity: {similarity:.3f}): {record.text}"
                    )
                    if collection_name is None and hasattr(record, 'collection_name'):
                        collection_name = record.collection_name
                context = "\n\n".join(context_parts)
                context = self._fix_divzero_in_context(context)
                # Prepend unit annotation so LLM knows values are already in display units
                context = (
                    "[UNIT NOTE: All numeric values below are in their final display units. "
                    "Percentage metrics (CTR, Fill Rate, etc.) already have ×100 applied: "
                    "a stored value of 0.49 means 0.49%, NOT 49%. "
                    "Do NOT re-multiply by 100.]\n\n"
                ) + context

                # Build date instruction from collection name
                date_instruction = "period 1 value, period 2 value"
                if collection_name:
                    p1s, p1e, p2s, p2e = extract_dates_from_collection_name(collection_name)
                    if p1e and p2e:
                        date_instruction = (
                            f"{format_date_readable(p1e)} value, "
                            f"{format_date_readable(p2e)} value"
                        )

                granularity_name = self.granularity_names.get(granularity, granularity.upper())
                metrics_instruction = self._metrics_instruction(requested_metrics)

                prompt = self.prompt_template.format(
                    granularity_name=granularity_name,
                    granularity=granularity.upper(),
                    dimension=dimension,
                    date_instruction=date_instruction,
                    metrics_instruction=metrics_instruction,
                )

                result = await _run_sync(
                    llm_service.generate_answer,
                    prompt,
                    context,
                    question,
                    0.1,
                    250,
                )

                elapsed = asyncio.get_event_loop().time() - start
                if result['success']:
                    logger.info("[%s-B%d] Batch complete in %.2fs", granularity.upper(), batch_index, elapsed)
                    return {
                        'granularity': granularity, 'batch_index': batch_index,
                        'insights': result['answer'], 'analysis_time': elapsed, 'error': None,
                    }
                else:
                    logger.warning("[%s-B%d] Batch failed: %s", granularity.upper(), batch_index, result.get('error'))
                    return self._empty(granularity, batch_index, result.get('error', 'Analysis failed'),
                                      elapsed)

            except Exception as exc:
                elapsed = asyncio.get_event_loop().time() - start
                logger.error("[%s-B%d] Exception: %s", granularity.upper(), batch_index, exc)
                return self._empty(granularity, batch_index, str(exc), elapsed)

    @staticmethod
    def _fix_divzero_in_context(context: str) -> str:
        """Fix #DIV/0! change_percentage stored as 0.0 when period1_value is zero.

        When the source CSV has change_percentage = #DIV/0! (period1 was zero),
        safe_float() converts it to 0.0. This misleads the LLM into computing
        infinity itself. We replace it with an explicit null marker so the LLM
        uses change_absolute as the reported change value instead.
        """
        NULL_MARKER = (
            'null (base period was zero; report the change_absolute value '
            'as the change with no % symbol — never output infinity or ∞)'
        )

        def fix_section(text: str) -> str:
            # Only fix when period1_value = 0 AND change_absolute is non-zero
            if not re.search(r'period1_value(?:\s+with\s+value\s*|:\s*)0(?:\.0)?\b', text):
                return text
            m = re.search(r'change_absolute(?:\s+with\s+value\s*|:\s*)(\d+(?:\.\d+)?)', text)
            if not m or float(m.group(1)) == 0.0:
                return text
            # Replace the misleading zero change_percentage with the null marker
            text = re.sub(
                r'(change_percentage(?:\s+with\s+value\s*|:\s*))0(?:\.0)?\b',
                r'\g<1>' + NULL_MARKER,
                text,
            )
            return text

        # Split by "In section" markers to fix metric-level chunks individually
        parts = re.split(r'(?=\bIn section\b)', context)
        return ''.join(fix_section(p) for p in parts)

    @staticmethod
    def _metrics_instruction(requested_metrics: Optional[List[str]]) -> str:
        if not requested_metrics:
            return ''
        metrics_list = ', '.join(requested_metrics)
        return (
            f"\n\n**FOCUS ONLY ON THESE METRICS**: {metrics_list}\n"
            f"**CRITICAL**: Do NOT report on other metrics even if present in the data."
        )

    @staticmethod
    def _empty(granularity: str, batch_index: int, error: str, elapsed: float = 0.0) -> Dict:
        return {
            'granularity': granularity, 'batch_index': batch_index,
            'insights': '', 'analysis_time': elapsed, 'error': error,
        }
