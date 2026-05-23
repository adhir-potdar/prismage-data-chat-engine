# prismage-data-chat-engine

A generic, metadata-driven data chatbot engine. Connect any SQL database, describe your domain
in JSON config files, and immediately ask natural language questions — no hardcoding of
dimensions, metrics, formulas, or prompts.

```
User: "Which regions are both below target and below last year MTD?"
  → ParsedIntent  → BusinessRules  → QueryBuilder  → SQL  → NL Answer
```

## Features

- **Fully metadata-driven** — dimensions, metrics, formulas, tables, and business rules
  all defined in JSON. Zero domain knowledge hardcoded in Python.
- **Five metric categories** — absolute (SUM/COUNT), average (AVG), cumulative (MTD/YTD/QTD),
  percentage (ratio formulas), and formula (complex multi-component with runtime variables).
- **Business Rules Engine** — trigger-action rules fire between parsing and query building.
  Rules can add metrics, inject formulas, create inline formulas, override existing formulas,
  set output hints, filter channels, and control sort order.
- **Dual table routing modes** — `affinity` mode resolves tables by set-intersection of
  `table_affinity` lists (deterministic, no API calls); `embedding` mode (default) uses
  OpenAI `text-embedding-3-small` to rank tables by semantic similarity to the question,
  with on-disk caching so embeddings are built only once.
- **Deterministic SQL** — the metadata-driven query builder produces predictable, formula-aware
  SQL without relying on an LLM for query construction.
- **LangChain integration** — uses LangChain for LLM calls (Stage 1 parsing, Stage 4 response)
  and `QuerySQLDataBaseTool` for query execution. `create_sql_query_chain` available as
  a low-confidence fallback.
- **LangSmith tracing** — `@traceable` on `ChatbotChain.answer()` and `_run_fallback()`;
  supports LangSmith Hub prompt versioning and regression datasets.
- **Multi-domain examples** — generic e-commerce config and an FMCG sales example
  (primary/secondary channel separation, MTD/LY comparisons) included out of the box.
- **Plugin system** — package a domain as a self-contained plugin directory with
  `plugin.json`, metadata, prompts, and optional capability overrides. Load by name with
  `build_plugin_engine("my-plugin")` or load all plugins at once with `build_multi_engine()`.
- **Engine capability overrides** — plugins can subclass `EngineCapabilities` to change
  how specific SQL clauses are built (e.g. snapshot date filter) without touching engine code.

## Quick Start

```bash
pip install -e .

# Copy and fill in environment variables
cp .env.example .env
# Edit .env — set DATABASE_URL, OPENAI_API_KEY at minimum

# Run the e-commerce demo
python examples/ecommerce/demo.py

# Run the FMCG sales demo
python examples/fmcg_sales/demo.py
```

## Repository Structure

```
prismage-data-chat-engine/
├── config/                        # Default generic e-commerce domain config
│   ├── metadata/
│   │   ├── dimensions.json        # Dimensions with aliases and table affinity
│   │   ├── metrics.json           # Metrics (all 5 categories) with formula refs
│   │   ├── formulas.json          # SQL expression templates with component mapping
│   │   ├── tables.json            # Tables with dimension/metric membership
│   │   └── business_rules.json    # Domain context, metric hints, HAVING patterns, rules
│   └── prompts/
│       ├── question_parser.json   # Stage 1 prompt template (Jinja2)
│       ├── query_generator.json   # Stage 2 fallback prompt template
│       └── nl_response.json       # Stage 4 NL response template
│
├── models/                        # Pydantic data models
│   ├── metadata.py                # Dimension, Metric, Formula, Table, BusinessRule
│   ├── intent.py                  # ParsedIntent, HavingConfig, SortConfig, OutputHint
│   └── query.py                   # BuiltQuery, QueryResult, ChatResponse
│
├── engine/
│   ├── rules/
│   │   └── engine.py              # BusinessRulesEngine — trigger evaluation + actions
│   ├── metadata/
│   │   ├── loader.py              # MetadataLoader — reads all 5 JSON files
│   │   ├── registry.py            # MetadataRegistry — O(1) alias lookups, render helpers
│   │   ├── validator.py           # MetadataValidator — startup cross-reference checks
│   │   └── embedding_store.py     # MetadataEmbeddingStore — embeds table docs, find_tables()
│   ├── query/
│   │   ├── router.py              # TableRouter — affinity set-intersection routing (default off)
│   │   ├── embedding_router.py    # EmbeddingTableRouter — semantic similarity routing (default)
│   │   ├── formula_engine.py      # FormulaEngine — expands SQL templates + runtime vars
│   │   ├── having_engine.py       # HavingEngine — builds HAVING clauses from patterns
│   │   └── builder.py             # QueryBuilder — assembles full SQL per table
│   ├── pipeline/
│   │   ├── question_parser.py     # Stage 1: question → ParsedIntent (LLM)
│   │   ├── query_builder.py       # Stage 2: intent → SQL (deterministic)
│   │   ├── query_executor.py      # Stage 3: SQL → rows + programmatic enumeration
│   │   └── nl_responder.py        # Stage 4: rows → natural language answer (LLM)
│   ├── prompts/
│   │   ├── prompt_library.py      # PromptLibrary — loads JSON or LangSmith Hub prompts
│   │   └── prompt_builder.py      # PromptBuilder — renders Jinja2 with live metadata
│   ├── capabilities/
│   │   └── base.py                # EngineCapabilities — overridable SQL-building behaviours
│   ├── plugins/
│   │   ├── loader.py              # PluginLoader — loads plugin directory into ChatbotChain
│   │   └── registry.py            # PluginRegistry — multi-plugin dispatch
│   └── chains/
│       └── chatbot_chain.py       # ChatbotChain — orchestrates all stages + fallback
│
├── adapters/
│   ├── database.py                # create_database() → SQLDatabase wrapper
│   ├── llm.py                     # create_llm() factory (OpenAI / Anthropic)
│   └── embeddings.py              # create_embeddings() factory (OpenAI / Voyage)
│
├── api/
│   └── chatbot.py                 # build_engine(), build_plugin_engine(), build_multi_engine()
│
├── plugins/
│   ├── empty-plugin/              # starter template — copy and rename to create a new plugin
│   └── <your-plugin>/             # one directory per domain plugin
│
├── examples/
│   ├── ecommerce/                 # Self-contained e-commerce domain example
│   │   ├── config/metadata/       # Domain-specific JSON overrides
│   │   └── demo.py
│   └── fmcg_sales/                # FMCG primary/secondary channel example
│       ├── config/metadata/
│       └── demo.py
│
├── tests/
│   ├── fixtures/                  # Minimal JSON configs for unit tests
│   ├── test_metadata_loader.py
│   ├── test_rules_engine.py
│   ├── test_query_builder.py
│   └── test_pipeline.py
│
├── .env.example                   # All PRISMAGE_* environment variables documented
├── docs/DESIGN.md                 # Detailed architecture and design document
├── pyproject.toml
└── requirements.txt
```

## Plugin System

The recommended way to onboard a new domain is to create a plugin — a self-contained
directory under `plugins/` that bundles metadata, prompts, and optional capability overrides.

```bash
# 1. Copy the starter template
cp -r plugins/empty-plugin plugins/my-plugin

# 2. Edit plugin.json — set name and description
# 3. Fill in config/metadata/ JSON files (dimensions, metrics, formulas, tables, rules)
# 4. Tune config/prompts/ templates
# 5. (Optional) override SQL behaviours in capabilities.py
```

**Load a single plugin:**

```python
from api.chatbot import build_plugin_engine

engine = build_plugin_engine("my-plugin")
response = engine.answer("Show top 5 products by revenue this month")
print(response.summary)
print(response.detail)
```

**Load all plugins at once:**

```python
from api.chatbot import build_multi_engine

registry = build_multi_engine()                            # auto-discovers all plugins/
response = registry.answer("my-plugin", "Top 5 products by revenue")
```

**Use without the plugin system** (direct config directory):

```python
from api.chatbot import build_engine

engine = build_engine(config_dir="my_domain/config/metadata")
response = engine.answer("Show top 5 products by revenue this month")
```

All engine parameters (`router_mode`, `llm_provider`, `llm_model`, etc.) can also be set
via `.env` — copy `.env.example` and fill in your values.

See [DESIGN.md](docs/DESIGN.md) → Plugin System and Engine Capabilities sections for the full
architecture, `PluginLoader` internals, `PluginRegistry`, and how to add new overridable
engine behaviours.

## Interactive CLI

`api/chatbot.py` includes a built-in interactive CLI for quick testing without writing any code:

```bash
# Generic engine (uses config/metadata/)
python -m api.chatbot

# Named plugin
python -m api.chatbot --plugin haldiram-sales

# Named plugin — also print the generated SQL for every question
python -m api.chatbot --plugin haldiram-sales --include-sql
```

**CLI flags:**

| Flag | Description |
|---|---|
| `--plugin NAME` | Load a named plugin from `plugins/<NAME>/` |
| `--include-sql` | Print the SQL query (or queries) generated for each question |

When `--include-sql` is set, a **SQL QUERIES** block is printed after each response showing the exact SQL sent to the database, labelled by channel (primary / secondary). Useful for debugging parser output and verifying HAVING, GROUP BY, and WHERE clauses.

## Running Tests

```bash
pytest tests/
```

## Design

See [DESIGN.md](docs/DESIGN.md) for the full architecture walkthrough, pipeline diagram, metadata
schema reference, business rules guide, formula engine, LangChain/LangSmith integration details,
plugin system internals, engine capabilities override pattern, and onboarding guide.

## License

MIT — see [LICENSE](LICENSE).
