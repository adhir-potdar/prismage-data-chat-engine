"""
ChatbotChain — LCEL pipeline composing all four stages.
Supports metadata-driven path and LangChain fallback path.
All stages are traced automatically via LangSmith when LANGSMITH_API_KEY is set.
"""
from __future__ import annotations
import logging
from langsmith import traceable
from models.intent import ParsedIntent
from models.query import ChatResponse, ExecutionResult
from engine.pipeline.question_parser import QuestionParser
from engine.pipeline.query_builder import QueryBuilderStage
from engine.pipeline.query_executor import QueryExecutor
from engine.pipeline.nl_responder import NLResponder

logger = logging.getLogger(__name__)


class ChatbotChain:
    """
    Orchestrates the full pipeline:

        Question
          │
          ▼ QuestionParser (Stage 1 — LLM + PydanticOutputParser)
          │
          ├─ confidence ≥ threshold
          │       ▼ QueryBuilderStage (Stage 2 — metadata-driven, no LLM)
          │
          └─ confidence < threshold
                  ▼ LangChain SQL Chain fallback
          │
          ▼ QueryExecutor (Stage 3 — LangChain QuerySQLDataBaseTool)
          │
          ▼ NLResponder (Stage 4 — LLM + programmatic enumeration)
          │
          ▼ ChatResponse
    """

    def __init__(
        self,
        parser: QuestionParser,
        query_builder: QueryBuilderStage,
        executor: QueryExecutor,
        responder: NLResponder,
        fallback_chain=None,   # LangChain create_sql_query_chain instance
    ):
        self.parser = parser
        self.query_builder = query_builder
        self.executor = executor
        self.responder = responder
        self.fallback_chain = fallback_chain

    @traceable(name="prismage_answer")
    def answer(self, question: str) -> ChatResponse:
        try:
            # Stage 1: Parse question → intent
            intent, use_fallback = self.parser.parse(question)

            # Stage 2: Build SQL queries
            if use_fallback and self.fallback_chain:
                queries, used_fallback = self._run_fallback(question, intent)
            else:
                queries = self.query_builder.build(intent, question)
                used_fallback = False

            if not queries:
                return ChatResponse(
                    question=question,
                    answer="I could not build a query for your question. Please try rephrasing.",
                    success=False,
                )

            # Stage 3: Execute queries
            execution_result: ExecutionResult = self.executor.execute(queries)
            execution_result.used_fallback = used_fallback
            execution_result.applied_rules = intent.applied_rules

            # Stage 4: Generate NL response
            answer, summary, detail = self.responder.respond(
                execution_result, intent, question
            )

            return ChatResponse(
                question=question,
                answer=answer,
                summary=summary,
                detail=detail,
                tabular=execution_result.programmatic_enumeration,
                success=True,
                sql_queries=[{"sql": q.sql, "table": q.table, "channel": q.channel} for q in queries],
                total_rows=execution_result.total_rows,
                used_fallback=used_fallback,
                applied_rules=intent.applied_rules,
            )

        except Exception as e:
            logger.exception(f"ChatbotChain error: {e}")
            return ChatResponse(
                question=question,
                answer="An internal error occurred. Please try again.",
                success=False,
                error=str(e),
            )

    # ── Fallback ─────────────────────────────────────────────────────────────

    @traceable(name="prismage_fallback_sql")
    def _run_fallback(self, question: str, intent: ParsedIntent):
        """Run LangChain SQL chain as fallback and wrap output into BuiltQuery list."""
        from models.query import BuiltQuery
        logger.info("Using LangChain SQL chain fallback.")
        try:
            sql = self.fallback_chain.invoke({"question": question})
            return [BuiltQuery(sql=sql, table="fallback", channel="fallback")], True
        except Exception as e:
            logger.error(f"Fallback chain error: {e}")
            return [], True
