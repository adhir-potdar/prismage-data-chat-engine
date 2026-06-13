"""
Generic LLM-based synthesizer for embedding search results.

Two synthesis functions:
  1. synthesize_granularity_batches() — merge multiple batches for one granularity
  2. synthesize_collective()          — final synthesis across all granularities

Both prompt templates are injected from the plugin's prompts_config.
"""
from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Optional

from engine.embedding.searcher import AnalysisResult
from engine.embedding.date_utils import extract_dates_from_collection_name, format_date_readable

logger = logging.getLogger(__name__)


async def _run_sync(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


class Synthesizer:
    """
    Synthesizes batch and cross-granularity analyses using LLM.

    prompts_config expected keys:
        granularity_synthesis  (str) — template for per-granularity batch merging
            placeholders: {granularity_name}, {granularity}, {combined_insights},
                          {metrics_instruction}
        synthesis              (str) — template for final collective synthesis
            placeholders: {dimension}, {available_granularities}, {all_insights_text},
                          {question}, {collective_metrics_instruction}
    """

    def __init__(self, prompts_config: Dict, granularity_names: Dict[str, str]):
        self.gran_synthesis_template: str = prompts_config.get('granularity_synthesis', '')
        self.collective_template: str = prompts_config.get('synthesis', '')
        self.granularity_names = granularity_names

    async def synthesize_batches(
        self,
        llm_service,
        granularity: str,
        batch_analyses: List[Dict],
        question: str,
        dimension: str,
        requested_metrics: Optional[List[str]],
    ) -> AnalysisResult:
        """
        Merge multiple batch analyses for one granularity into a single AnalysisResult.
        If only one batch, returns it directly (no LLM call needed).
        """
        gran_name = self.granularity_names.get(granularity, granularity.upper())
        start = asyncio.get_event_loop().time()

        successful = [b for b in batch_analyses if b.get('insights')]
        if not successful:
            return AnalysisResult(granularity=granularity, granularity_name=gran_name,
                                  insights='', analysis_time=0, error='All batches failed')

        if len(successful) == 1:
            b = successful[0]
            return AnalysisResult(granularity=granularity, granularity_name=gran_name,
                                  insights=b['insights'], analysis_time=b['analysis_time'])

        # Multiple batches — synthesize
        combined = "\n\n".join(f"Batch {b['batch_index']}: {b['insights']}" for b in successful)
        max_time = max(b.get('analysis_time', 0) for b in successful)
        metrics_instruction = self._metrics_note(requested_metrics)

        prompt = self.gran_synthesis_template.format(
            granularity_name=gran_name,
            granularity=granularity.upper(),
            combined_insights=combined,
            metrics_instruction=metrics_instruction,
        )

        try:
            result = await _run_sync(llm_service.generate_answer, prompt, '', question, 0.1, 300)
            synthesis_time = asyncio.get_event_loop().time() - start
            if result['success']:
                return AnalysisResult(granularity=granularity, granularity_name=gran_name,
                                      insights=result['answer'],
                                      analysis_time=max_time + synthesis_time)
        except Exception as exc:
            logger.error("[%s] Batch synthesis error: %s", granularity.upper(), exc)

        # Fallback: concatenate
        return AnalysisResult(granularity=granularity, granularity_name=gran_name,
                              insights=combined, analysis_time=max_time)

    async def synthesize_collective(
        self,
        llm_service,
        analysis_results: List[AnalysisResult],
        question: str,
        dimension: str,
        requested_metrics: Optional[List[str]],
        filtered_collections: Optional[Dict] = None,
        specified_dates: Optional[Dict] = None,
    ) -> str:
        """
        Final synthesis: combine all granularity analyses into one comprehensive answer.
        Returns the formatted answer string.
        """
        successful = [r for r in analysis_results if r.insights and not r.error]
        if not successful:
            return "No relevant information found to analyze."

        available_grans = ', '.join(r.granularity.upper() for r in successful)

        all_insights = ''
        for r in successful:
            all_insights += f"\n{r.granularity_name} ({r.granularity.upper()}) Insights:\n"
            all_insights += r.insights + "\n"

        collective_metrics_instruction = ''
        if requested_metrics:
            metrics_list = ', '.join(requested_metrics)
            collective_metrics_instruction = (
                f"\n\n**FOCUS ONLY ON THESE METRICS**: {metrics_list}\n"
                f"**CRITICAL**: Do NOT include information about other metrics."
            )

        prompt = self.collective_template.format(
            dimension=dimension,
            available_granularities=available_grans,
            all_insights_text=all_insights,
            question=question,
            collective_metrics_instruction=collective_metrics_instruction,
        )

        try:
            result = await _run_sync(
                llm_service.generate_answer,
                prompt,
                all_insights,
                question,
                0.1,
                1500,
            )
            if result['success']:
                # Post-process: remove incomplete-data placeholder lines
                lines = [
                    line for line in result['answer'].split('\n')
                    if not any(
                        phrase in line.lower()
                        for phrase in ['data not available', 'data not provided', 'not mentioned', 'n/a']
                    )
                ]
                clean = '\n'.join(lines).strip()

                # Extract data period from selected collection names
                data_period_str = ""
                if filtered_collections:
                    all_dates = []
                    for colls in filtered_collections.values():
                        for coll in colls:
                            p1s, _, _, p2e = extract_dates_from_collection_name(coll)
                            if p1s is not None and p2e is not None:
                                all_dates.extend([p1s, p2e])
                    if all_dates:
                        actual_period = (
                            f"{format_date_readable(min(all_dates))}"
                            f" to {format_date_readable(max(all_dates))}"
                        )
                        if specified_dates and specified_dates.get('raw_text'):
                            data_period_str = (
                                f"📅 Requested: {specified_dates['raw_text']}"
                                f" | Actual Data: {actual_period}\n"
                            )
                        else:
                            data_period_str = f"📅 Data Period: {actual_period}\n"

                header = (
                    f"📈 COMPREHENSIVE MULTI-GRANULARITY ANALYSIS\n"
                    f"📦 Dimension: {dimension.upper()}\n"
                    f"🕒 Time Granularities Analyzed: {available_grans}\n"
                    + data_period_str
                    + "=" * 80 + "\n\n"
                )
                footer = (
                    "\n\n" + "=" * 80 + "\n"
                    f"💡 Analysis based on data from {len(successful)} time granularity/ies"
                )

                # Split LLM output: first paragraph = quick summary, rest = detail
                parts = clean.split('\n\n', 1)
                quick_summary = parts[0].strip()
                detail_body = parts[1].strip() if len(parts) > 1 else clean

                # Structure: quick_summary \n\n header + detail_body + footer
                # _split_answer() will pick quick_summary as summary and the rest as detail
                return quick_summary + '\n\n' + header + detail_body + footer

        except Exception as exc:
            logger.error("Collective synthesis error: %s", exc)

        return "Failed to generate collective analysis."

    @staticmethod
    def _metrics_note(requested_metrics: Optional[List[str]]) -> str:
        if not requested_metrics:
            return ''
        return f"\n**FOCUS ONLY ON**: {', '.join(requested_metrics)}"
