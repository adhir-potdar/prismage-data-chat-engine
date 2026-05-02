"""
Stage 4 — NL Responder
Generates a natural language answer from query execution results.
Uses programmatic enumeration as LLM input — never raw markdown tables.
"""
from __future__ import annotations
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from models.intent import ParsedIntent
from models.query import ExecutionResult
from engine.prompts.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class NLResponder:
    """
    Stage 4: ExecutionResult + ParsedIntent → natural language answer string

    Key design:
    - programmatic_enumeration is always used as data_context (not raw table)
    - task_description is overridden to prevent LLM re-filtering pre-validated data
    - output_hints from business rules control table vs narrative format
    """

    def __init__(self, llm: BaseChatModel, prompt_builder: PromptBuilder):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.output_parser = StrOutputParser()

    def respond(self, execution_result: ExecutionResult, intent: ParsedIntent, question: str) -> str:
        if not execution_result.programmatic_enumeration or \
                execution_result.programmatic_enumeration == "No data returned.":
            return "No data was found for your query. Please refine your question or check the filters."

        total_rows = execution_result.total_rows
        data_context = execution_result.programmatic_enumeration

        task_description = self._build_task_description(total_rows, intent)
        context_line = self._build_context_line(question, intent)

        prompt = self.prompt_builder.build_response_prompt(
            data_context=data_context,
            task_description=task_description,
            context_line=context_line,
        )

        try:
            chain = prompt | self.llm | self.output_parser
            return chain.invoke({})
        except Exception as e:
            logger.error(f"Stage 4 LLM error: {e}")
            return data_context   # graceful fallback: return enumeration directly

    # ── Private ──────────────────────────────────────────────────────────────

    def _build_task_description(self, total_rows: int, intent: ParsedIntent) -> str:
        if total_rows == 0:
            return "No results were returned. Inform the user clearly."

        # Programmatic mode: override task to prevent LLM re-filtering
        return (
            f"The {total_rows} entries listed above are the COMPLETE, PRE-VALIDATED results "
            f"already filtered by SQL. Provide business insights about ALL of them. "
            f"DO NOT re-evaluate, re-filter, or question whether entries qualify. "
            f"Describe patterns, highlight notable entries, and summarise overall findings."
        )

    def _build_context_line(self, question: str, intent: ParsedIntent) -> str:
        # Suppress the original question in the prompt to prevent LLM
        # re-applying its own filter semantics from the question text.
        if intent.applied_rules:
            return "Context: The following are pre-filtered sales data entries. Provide business insights and analysis."
        return f"User Question: {question}"
