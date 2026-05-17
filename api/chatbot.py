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
    config_dir: str | None = None,
    prompts_dir: str | None = None,
    connection_string: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    enable_fallback: bool | None = None,
    router_mode: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_cache_path: str | None = None,
    embedding_top_k: int | None = None,
    capabilities=None,
) -> ChatbotChain:
    """
    Build and wire the full Prismage engine from config.

    All parameters fall back to PRISMAGE_* environment variables (see .env.example),
    then to hardcoded defaults. Explicit arguments take highest precedence.

    Args:
        config_dir:             path to metadata JSON directory          (PRISMAGE_CONFIG_DIR, default "config/metadata")
        prompts_dir:            path to prompt JSON directory            (PRISMAGE_PROMPTS_DIR, default "config/prompts")
        connection_string:      SQLAlchemy DB URL                        (DATABASE_URL)
        llm_provider:           "openai" or "anthropic"                  (PRISMAGE_LLM_PROVIDER, default "openai")
        llm_model:              override default model name              (PRISMAGE_LLM_MODEL)
        enable_fallback:        LangChain SQL chain fallback             (PRISMAGE_ENABLE_FALLBACK, default True)
        router_mode:            "embedding" (default) or "affinity"      (PRISMAGE_ROUTER_MODE)
        embedding_provider:     "openai" or "voyage"                     (PRISMAGE_EMBEDDING_PROVIDER, default "openai")
        embedding_model:        embedding model name                     (PRISMAGE_EMBEDDING_MODEL, default "text-embedding-3-small")
        embedding_cache_path:   on-disk cache path; "" disables caching  (PRISMAGE_EMBEDDING_CACHE_PATH)
        embedding_top_k:        top-K tables from embedding search       (PRISMAGE_EMBEDDING_TOP_K, default 3)

    Returns:
        ChatbotChain — call .answer(question) to query the engine.
    """
    load_dotenv()

    # ── Resolve config from env vars (explicit args take precedence) ──────────
    config_dir = config_dir or os.getenv("PRISMAGE_CONFIG_DIR", "config/metadata")
    prompts_dir = prompts_dir or os.getenv("PRISMAGE_PROMPTS_DIR", "config/prompts")
    llm_provider = llm_provider or os.getenv("PRISMAGE_LLM_PROVIDER", "openai")
    llm_model = llm_model or os.getenv("PRISMAGE_LLM_MODEL", "gpt-4o-mini")
    router_mode = router_mode or os.getenv("PRISMAGE_ROUTER_MODE", "embedding")
    embedding_provider = embedding_provider or os.getenv("PRISMAGE_EMBEDDING_PROVIDER", "openai")
    embedding_model = embedding_model or os.getenv("PRISMAGE_EMBEDDING_MODEL", "text-embedding-3-small")
    _cache_env = os.getenv("PRISMAGE_EMBEDDING_CACHE_PATH", ".cache/metadata_embeddings.json")
    embedding_cache_path = embedding_cache_path if embedding_cache_path is not None else (_cache_env or None)
    embedding_top_k = embedding_top_k or int(os.getenv("PRISMAGE_EMBEDDING_TOP_K", "3"))
    if enable_fallback is None:
        enable_fallback = os.getenv("PRISMAGE_ENABLE_FALLBACK", "true").lower() != "false"

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

    if router_mode == "embedding":
        from adapters.embeddings import create_embeddings
        from engine.metadata.embedding_store import MetadataEmbeddingStore
        from engine.query.embedding_router import EmbeddingTableRouter
        emb = create_embeddings(provider=embedding_provider, model=embedding_model)
        store = MetadataEmbeddingStore(config, emb, cache_path=embedding_cache_path)
        store.build()
        router = EmbeddingTableRouter(store, registry, top_k=embedding_top_k)
    else:
        router = TableRouter(registry)

    formula_engine = FormulaEngine(registry)
    having_engine = HavingEngine(registry, config.having_patterns)
    if capabilities is None:
        from engine.capabilities.base import EngineCapabilities
        capabilities = EngineCapabilities()
    query_builder = QueryBuilder(registry, router, formula_engine, having_engine, context, capabilities)

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


def build_plugin_engine(
    plugin: str,
    plugins_root: str | None = None,
    connection_string: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    **kwargs,
) -> ChatbotChain:
    """
    Build a ChatbotChain for a named plugin.

    Discovers the plugin directory under plugins_root/<plugin>/ and loads
    its plugin.json manifest to resolve config/prompts paths, then delegates
    to build_engine().

    Args:
        plugin:             plugin name (e.g. "haldiram-sales")
        plugins_root:       path to the plugins directory (PRISMAGE_PLUGINS_ROOT,
                            default "plugins")
        connection_string:  SQLAlchemy DB URL
        llm_provider:       "openai" or "anthropic"
        llm_model:          model name override
        **kwargs:           forwarded to build_engine()

    Returns:
        ChatbotChain for the requested plugin.
    """
    import os
    from pathlib import Path
    from engine.plugins.loader import PluginLoader

    root = plugins_root or os.getenv("PRISMAGE_PLUGINS_ROOT", "plugins")
    plugin_dir = str(Path(root) / plugin)

    return PluginLoader().load(
        plugin_dir=plugin_dir,
        connection_string=connection_string,
        llm_provider=llm_provider,
        llm_model=llm_model,
        **kwargs,
    )


def build_multi_engine(
    plugins_root: str | None = None,
    connection_string: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    **kwargs,
):
    """
    Discover and load all plugins under plugins_root/ into a PluginRegistry.

    Each subdirectory that contains a plugin.json is treated as a plugin.
    Returns a PluginRegistry whose .answer(plugin, question) method dispatches
    to the correct plugin engine.

    Args:
        plugins_root:       path to the plugins directory (PRISMAGE_PLUGINS_ROOT,
                            default "plugins")
        connection_string:  SQLAlchemy DB URL (shared across all plugins)
        llm_provider:       "openai" or "anthropic"
        llm_model:          model name override
        **kwargs:           forwarded to build_engine() for every plugin

    Returns:
        PluginRegistry with all discovered plugins loaded.
    """
    import os
    from pathlib import Path
    from engine.plugins.loader import PluginLoader
    from engine.plugins.registry import PluginRegistry

    root = plugins_root or os.getenv("PRISMAGE_PLUGINS_ROOT", "plugins")
    root_path = Path(root)

    if not root_path.exists():
        raise FileNotFoundError(f"Plugins root directory not found: {root_path.resolve()}")

    registry = PluginRegistry()
    loader = PluginLoader()

    for entry in sorted(root_path.iterdir()):
        if entry.is_dir() and (entry / "plugin.json").exists():
            try:
                chain = loader.load(
                    plugin_dir=str(entry),
                    connection_string=connection_string,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    **kwargs,
                )
                # Use the name from plugin.json as the registry key
                import json
                manifest = json.loads((entry / "plugin.json").read_text())
                name = manifest.get("name", entry.name)
                registry.register(name, chain)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to load plugin from {entry}: {e}"
                )

    return registry


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
