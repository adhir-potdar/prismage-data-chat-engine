"""
Prismage Data Chat Engine — main entry point.
Wires together all components and exposes a simple answer() interface.
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
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
    capabilities=None,
    enable_charts: bool = False,
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
    db_tables = [t.name for t in config.tables] if config.tables else None
    try:
        db = create_database(conn_str, include_tables=db_tables)
        logger.info("Database connected; scoped to config tables: %s", db_tables)
    except ValueError as exc:
        logger.warning(
            "Config tables not found in database (%s) — falling back to full reflection. "
            "Run create_tables.py to create missing tables. Details: %s",
            db_tables,
            exc,
        )
        db = create_database(conn_str)

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm = create_llm(provider=llm_provider, model=llm_model)

    # ── Query sub-components ─────────────────────────────────────────────────
    context = QueryContext()

    # Resolve capabilities before router so threshold/suffix methods are available
    if capabilities is None:
        from engine.capabilities.base import EngineCapabilities
        capabilities = EngineCapabilities()

    if router_mode == "embedding":
        from adapters.embeddings import create_embeddings
        from engine.metadata.embedding_store import MetadataEmbeddingStore
        from engine.query.embedding_router import EmbeddingTableRouter
        emb = create_embeddings(provider=embedding_provider, model=embedding_model)
        store = MetadataEmbeddingStore(config, emb, cache_path=embedding_cache_path)
        store.build()
        router = EmbeddingTableRouter(store, registry, capabilities)
    else:
        router = TableRouter(registry, capabilities)

    formula_engine = FormulaEngine(registry)
    having_engine = HavingEngine(registry, config.having_patterns)
    query_builder = QueryBuilder(registry, router, formula_engine, having_engine, context, capabilities)

    # ── Pipeline stages ──────────────────────────────────────────────────────
    parser = QuestionParser(llm, builder, capabilities)
    rules_engine = BusinessRulesEngine(config.rules, registry)
    builder_stage = QueryBuilderStage(rules_engine, query_builder)
    executor = QueryExecutor(db)
    responder = NLResponder(llm, builder)

    # ── Fallback (LangChain SQL chain) ────────────────────────────────────────
    fallback = None
    if enable_fallback:
        from langchain_classic.chains.sql_database.query import create_sql_query_chain
        fallback = create_sql_query_chain(llm, db)

    return ChatbotChain(parser, builder_stage, executor, responder, fallback,
                        enable_charts=enable_charts)


def build_plugin_engine(
    plugin: str,
    plugins_root: str | None = None,
    connection_string: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    **kwargs,
):
    """
    Build a chain for a named plugin.

    Discovers the plugin directory under plugins_root/<plugin>/ and loads
    its plugin.json manifest. SQL plugins (default) return a ChatbotChain;
    embedding plugins (mode="embedding") return an EmbeddingChain.
    Both share the same .answer(question) -> ChatResponse interface.

    Args:
        plugin:             plugin name (e.g. "haldiram-sales", "yield-management")
        plugins_root:       path to the plugins directory (PRISMAGE_PLUGINS_ROOT,
                            default "plugins")
        connection_string:  SQLAlchemy DB URL (SQL plugins only)
        llm_provider:       "openai" or "anthropic" (SQL plugins only)
        llm_model:          model name override (SQL plugins only)
        **kwargs:           forwarded to build_engine() for SQL plugins

    Returns:
        ChatbotChain (SQL plugin) or EmbeddingChain (embedding plugin).
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
    SQL plugins return ChatbotChain; embedding plugins return EmbeddingChain.
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


_SEP = "=" * 80


def _build_markdown_table(query_results: list) -> str:
    """
    Build padded markdown tables from the query_results list in ChatResponse.
    Each entry is a dict: {channel, columns, rows, row_count}.
    Numbers are formatted to 2 decimal places; column headers uppercased.
    """
    import decimal
    sections = []

    for result in query_results:
        channel = result.get("channel", "unknown")
        rows = result.get("rows", [])
        columns = result.get("columns", [])
        row_count = result.get("row_count", len(rows))

        if not rows or not columns:
            continue

        # Interleave val/vol pairs: cymtd_val|cymtd_vol|lymtd_val|lymtd_vol|...
        # Dimension columns (non-numeric) stay first; orphan suffixed cols go last.
        def _interleave_columns(cols):
            val_cols = [c for c in cols if c.endswith("_val")]
            vol_set = {c for c in cols if c.endswith("_vol")}
            dim_cols = [c for c in cols if not c.endswith("_val") and not c.endswith("_vol")]
            ordered = list(dim_cols)
            for vc in val_cols:
                ordered.append(vc)
                vol_c = vc[:-4] + "_vol"
                if vol_c in vol_set:
                    ordered.append(vol_c)
                    vol_set.discard(vol_c)
            ordered.extend(sorted(vol_set))  # orphan vol cols with no matching val
            return ordered if (val_cols or vol_set) else cols

        columns = _interleave_columns(columns)

        def _fmt(v):
            if isinstance(v, decimal.Decimal):
                return f"{float(v):.2f}"
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v) if v is not None else ""

        formatted = [{col: _fmt(row.get(col)) for col in columns} for row in rows]
        headers = [col.replace("_", " ").upper() for col in columns]

        # Compute column widths so cells are padded evenly
        widths = [len(h) for h in headers]
        for row in formatted:
            for i, col in enumerate(columns):
                widths[i] = max(widths[i], len(row[col]))

        hdr = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
        sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        data_rows = [
            "| " + " | ".join(row[col].ljust(widths[i]) for i, col in enumerate(columns)) + " |"
            for row in formatted
        ]

        channel_label = channel.replace("_", " ").title()
        sections.append(
            f"### {channel_label}\n"
            f"*Showing {row_count} of {row_count} rows*\n\n"
            f"**{channel_label} Sales**\n"
            + hdr + "\n" + sep + "\n" + "\n".join(data_rows)
        )

    return "\n\n".join(sections)


def _run_question(engine, question: str, include_sql: bool, include_chart: bool = False) -> None:
    """Answer one question and print the result."""
    t_start = time.time()
    start_dt = datetime.now()

    print(f"\n{_SEP}")
    print(f"Q: {question}")
    print(f"⏰ Started at: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{_SEP}\n")

    response: ChatResponse = engine.answer(question, verbose=True)

    if include_sql and response.sql_queries:
        print(f"\n{_SEP}")
        print("SQL QUERIES")
        print(_SEP)
        for i, q in enumerate(response.sql_queries, 1):
            lbl = q.get("channel") or q.get("table", f"query {i}")
            print(f"\n-- Query {i}: {lbl}")
            print(q.get("sql", ""))
        print(_SEP)

    t_total = time.time() - t_start
    end_dt = datetime.now()

    if not response.success:
        print(f"\n{_SEP}")
        print("ERROR")
        print(_SEP)
        print(response.error or response.answer)
    else:
        if response.summary:
            print(f"\n{_SEP}")
            print("QUICK SUMMARY")
            print(_SEP)
            print(response.summary)

        if response.detail:
            print(f"\n{_SEP}")
            print("ANALYSIS (CONCISE)")
            print(_SEP)
            print(response.detail)
        elif response.answer and not response.summary:
            print(f"\n{_SEP}")
            print("ANSWER")
            print(_SEP)
            print(response.answer)

        if response.query_results:
            table_str = _build_markdown_table(response.query_results)
            if table_str:
                print(f"\n{_SEP}")
                print("TABULAR OUTPUT")
                print(_SEP)
                print(table_str)
                print(_SEP)

        if include_chart and response.vega_lite_spec:
            import json as _json
            for chart in response.vega_lite_spec:
                channel = (chart.get("channel") or "").upper()
                label = f"CHART SPEC (VEGA-LITE){' — ' + channel if channel else ''}"
                print(f"\n{_SEP}")
                print(label)
                print(_SEP)
                print(_json.dumps(chart["spec"], indent=2, ensure_ascii=False))
                print(_SEP)

    print(f"\n{_SEP}")
    print(f"⏰ Completed at: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total time: {t_total:.2f}s")
    print(f"{_SEP}\n")


def main():
    """Interactive CLI mode.

    Usage:
        python -m api.chatbot                          # generic engine (config/metadata/)
        python -m api.chatbot --plugin haldiram-sales  # named plugin
    """

    parser = argparse.ArgumentParser(description="Prismage Data Chat Engine — interactive CLI")
    parser.add_argument("--plugin", metavar="NAME", help="Plugin name to load (e.g. haldiram-sales)")
    parser.add_argument("--include-sql", action="store_true", help="Print the generated SQL queries for each question")
    parser.add_argument("--chart", action="store_true", help="Generate and print a Vega-Lite v5 chart spec after each answer")
    parser.add_argument("--question", metavar="TEXT", help="Answer a single question and exit (non-interactive mode)")
    args = parser.parse_args()

    if args.plugin:
        # Validate plugin exists before attempting to load
        _plugins_root = os.getenv("PRISMAGE_PLUGINS_ROOT", "plugins")
        _plugin_path = Path(_plugins_root) / args.plugin
        if not _plugin_path.exists() or not (_plugin_path / "plugin.json").exists():
            _available = sorted(
                d.name for d in Path(_plugins_root).iterdir()
                if d.is_dir() and (d / "plugin.json").exists()
            ) if Path(_plugins_root).exists() else []
            print(f"ERROR: Plugin '{args.plugin}' not found in '{_plugins_root}/'.")
            if _available:
                print(f"Available plugins: {', '.join(_available)}")
            else:
                print(f"No plugins found in '{_plugins_root}/'. Run setup scripts first.")
            sys.exit(1)

        print(f"Loading plugin '{args.plugin}'...")
        try:
            engine = build_plugin_engine(args.plugin, enable_charts=args.chart)
        except Exception as e:
            print(f"ERROR: Failed to load plugin '{args.plugin}': {e}")
            sys.exit(1)
        label = args.plugin
    else:
        engine = build_engine(enable_charts=args.chart)
        label = "generic"

    # ── Single-question mode (--question "...") ───────────────────────────────
    if args.question:
        _run_question(engine, args.question.strip(), args.include_sql, args.chart)
        return

    print(f"Prismage Data Chat Engine [{label}] — type 'exit' to quit.\n")

    try:
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if question.lower() in ("exit", "quit"):
                break
            if not question:
                continue
            _run_question(engine, question, args.include_sql, args.chart)

    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
