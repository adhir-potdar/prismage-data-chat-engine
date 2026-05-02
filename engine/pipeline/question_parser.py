"""
Stage 1 — Question Parser
Converts a natural language question into a structured ParsedIntent.
Uses LangChain ChatPromptTemplate + LLM + PydanticOutputParser.
"""
from __future__ import annotations
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from models.intent import ParsedIntent
from engine.prompts.prompt_builder import PromptBuilder

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

    def __init__(self, llm: BaseChatModel, prompt_builder: PromptBuilder):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.output_parser = PydanticOutputParser(pydantic_object=ParsedIntent)

    def parse(self, question: str) -> tuple[ParsedIntent, bool]:
        """
        Returns (ParsedIntent, use_fallback).
        use_fallback=True when confidence < threshold or unknown terms detected.
        """
        try:
            prompt = self.prompt_builder.build_parser_prompt(question)
            chain = prompt | self.llm | self.output_parser
            intent: ParsedIntent = chain.invoke({"question": question})
            use_fallback = intent.confidence < self.CONFIDENCE_THRESHOLD
            if use_fallback:
                logger.info(f"Low confidence ({intent.confidence:.2f}), routing to fallback.")
            return intent, use_fallback
        except Exception as e:
            logger.error(f"Stage 1 parse error: {e}")
            return ParsedIntent(confidence=0.0), True
