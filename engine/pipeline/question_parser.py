"""
Stage 1 — Question Parser
Converts a natural language question into a structured ParsedIntent.
Uses LangChain ChatPromptTemplate + LLM + PydanticOutputParser.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from models.intent import ParsedIntent
from engine.prompts.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from engine.capabilities.base import EngineCapabilities

logger = logging.getLogger(__name__)


class QuestionParser:
    """
    Stage 1: Natural Language → ParsedIntent

    Flow:
        question
          → PromptBuilder injects metadata (dimensions, metrics, domain_context)
          → LLM returns JSON
          → PydanticOutputParser validates into ParsedIntent
          → confidence checked; low confidence flagged for fallback
    """

    CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        llm: BaseChatModel,
        prompt_builder: PromptBuilder,
        capabilities: "EngineCapabilities | None" = None,
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.capabilities = capabilities
        self.output_parser = PydanticOutputParser(pydantic_object=ParsedIntent)

    def parse(self, question: str) -> tuple[ParsedIntent, bool]:
        """
        Returns (ParsedIntent, use_fallback).
        use_fallback=True when confidence < threshold or unknown terms detected.
        """
        try:
            fmt = self.output_parser.get_format_instructions()
            prompt = self.prompt_builder.build_parser_prompt(question, format_instructions=fmt)
            chain = prompt | self.llm | self.output_parser
            intent: ParsedIntent = chain.invoke({"question": question})

            # Default metric fallback: if LLM extracted no metrics, apply the
            # plugin-configured default (mirrors old engine Python-level fallback).
            if not intent.metrics and not intent.formula_metrics and self.capabilities:
                defaults = self.capabilities.get_default_metrics()
                if defaults:
                    logger.info("No metrics extracted — defaulting to %s", defaults)
                    intent.metrics = list(defaults)

            # Ensure the sort metric is included in metrics (so it appears in
            # SELECT and ORDER BY). If the LLM sets sort but not the metric itself.
            if intent.sort and intent.sort.metric:
                sm = intent.sort.metric
                from models.intent import SortConfig as _SC
                if sm not in intent.metrics and sm not in intent.formula_metrics:
                    intent.metrics = [sm] + [m for m in intent.metrics if m != sm]

            # Default sort: when limit is set but sort is null, sort DESC by the
            # first metric (so "top N" returns the highest-ranked rows).
            if intent.limit and not intent.sort and intent.metrics:
                from models.intent import SortConfig as _SC
                intent.sort = _SC(metric=intent.metrics[0], direction="DESC")

            # Translate channel_filter → intent.channels so TableRouter can filter tables.
            # Only set if business rules haven't already populated intent.channels.
            if intent.channel_filter and not intent.channels:
                intent.channels = [intent.channel_filter]

            use_fallback = intent.confidence < self.CONFIDENCE_THRESHOLD
            if use_fallback:
                logger.info(f"Low confidence ({intent.confidence:.2f}), routing to fallback.")
            return intent, use_fallback
        except Exception as e:
            logger.error(f"Stage 1 parse error: {e}")
            return ParsedIntent(confidence=0.0), True
