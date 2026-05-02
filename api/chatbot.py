"""
Prismage Data Chat Engine — main entry point.
Wires together all components and exposes a simple answer() interface.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase

from adapters.database import create_database
from adapters.llm import create_llm
from engine.metadata.loader import MetadataLoader
from engine.metadata.registry import MetadataRegistry
from engine.metadata.validator import MetadataValidator
from engine.prompts.prompt_library import PromptLibrary
from engine.prompts.prompt_builder import PromptBuilder
from engine.rules.engine import BusinessRulesEngine
from engine.query.router import TableRouter
from engine.query.formula_engine import FormulaEngine, QueryContext
from engine.query.having_engine import HavingEngine
from engine.query.builder import QueryBuilder
from engine.pipeline.question_parser import QuestionParser
from engine.pipeline.query_builder import QueryBuilderStage
from engine.pipeline.query_executor import QueryExecutor
from engine.pipeline.nl_responder import NLResponder
from engine.chains.chatbot_chain import ChatbotChain
from models.query import ChatResponse


def build_engine(
    config_dir: str = "config/metadata",
    prompts_dir: str = "config/prompts",
    connection_string: str | None = None,
    llm_provider: str = "openai",
    llm_model: str | None = None,
    enable_fallback: bool = True,
) -> ChatbotChain:
    """
    Build and wire the full Prismage engine from config.

    Args:
        config_dir:         path to metadata JSON directory
        prompts_dir:        path to prompt JSON directory
        connection_string:  SQLAlchemy DB URL (falls back to DATABASE_URL env var)
        llm_provider:       "openai" or "anthropic"
        llm_model:          override default model name
        enable_fallback:    enable LangChain SQL chain fallback for low-confidence queries

    Returns:
        ChatbotChain — call .answer(question) to query the engine.
    """
    load_dotenv()

    # ── Metadata ─────────────────────────────────────────────────────────────
    loader = MetadataLoader(config_dir)
    config = loader.load()
    MetadataValidator(config).validate()
    registry = MetadataRegistry(config)

    # ── Prompts ───────────────────────────────────────────────────────────────
    library = PromptLibrary(prompts_dir)
    builder = PromptBuilder(library, registry, config)

    # ── Database ──────────────────────────────────────────────────────────────
    conn_str = connection_string or os.getenv("DATABASE_URL")
    if not conn_str:
        raise ValueError("DATABASE_URL env var or connection_string argument is required.")
    db = create_database(conn_str)

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm = create_llm(provider=llm_provider, model=llm_model)

    # ── Query sub-components ─────────────────────────────────────────────────
    context = QueryContext()
    router = TableRouter(registry)
    formula_engine = FormulaEngine(registry)
    having_engine = HavingEngine(registry, config.having_patterns)
    query_builder = QueryBuilder(registry, router, formula_engine, having_engine, context)

    # ── Pipeline stages ──────────────────────────────────────────────────────
    parser = QuestionParser(llm, builder)
    rules_engine = BusinessRulesEngine(config.rules, registry)
    builder_stage = QueryBuilderStage(rules_engine, query_builder)
    executor = QueryExecutor(db)
    responder = NLResponder(llm, builder)

    # ── Fallback (LangChain SQL chain) ────────────────────────────────────────
    fallback = None
    if enable_fallback:
        from langchain.chains import create_sql_query_chain
        fallback = create_sql_query_chain(llm, db)

    return ChatbotChain(parser, builder_stage, executor, responder, fallback)


def main():
    """Interactive CLI mode."""
    engine = build_engine()
    print("Prismage Data Chat Engine — type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        response: ChatResponse = engine.answer(question)
        print(f"\nAnswer:\n{response.answer}")
        if response.used_fallback:
            print("  [used LangChain fallback SQL chain]")
        if response.applied_rules:
            print(f"  [rules applied: {', '.join(response.applied_rules)}]")
        print()


if __name__ == "__main__":
    main()
