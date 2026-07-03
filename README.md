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
- **Embedding plugin mode** — alternative to SQL for pre-aggregated vector datasets. Set
  `"mode": "embedding"` in `plugin.json` to activate. Uses OpenAI vector embeddings to search
  across time-granularity collections (DOD, WTD, MTD, QTD, MOM, QOQ), runs parallel batch LLM
  analysis, and synthesizes a multi-granularity answer with data period display. Supports
  dimension hierarchies, date range filtering from natural language, and deterministic
  result selection.

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
│   │   ├── loader.py              # PluginLoader — loads SQL or embedding plugin into chain
│   │   └── registry.py            # PluginRegistry — multi-plugin dispatch
│   ├── chains/
│   │   ├── chatbot_chain.py       # ChatbotChain — SQL pipeline (parse → SQL → NL answer)
│   │   └── embedding_chain.py     # EmbeddingChain — vector search pipeline (parse → search → synthesize)
│   ├── charting/
│   │   ├── chart_generator.py     # generate_vega_spec() — LLM + Pydantic + post-processing heuristics
│   │   └── vega_types.yaml        # Trino column type → Vega-Lite type mapping
│   └── embedding/
│       ├── question_parser.py     # Extracts metrics, dimensions, date range from question
│       ├── collection_finder.py   # Finds matching vector collections by dimension + granularity + date
│       ├── searcher.py            # Parallel vector similarity search across collections
│       ├── analyzer.py            # Batch LLM analysis of search results per granularity
│       ├── synthesizer.py         # Multi-granularity synthesis into final answer
│       ├── orchestrator.py        # 3-phase pipeline: search → analyze → synthesize
│       └── date_utils.py          # Date parsing, collection name date extraction
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
├── chart-preview/                 # Angular 20 UI for interactive Vega-Lite spec preview
│   ├── src/app/
│   │   ├── app.ts                 # Rendering logic — step-based sizing, debounce, resize
│   │   ├── app.html               # Split-pane layout (JSON editor left, chart right)
│   │   └── app.css                # Layout styles
│   └── README.md                  # Setup and usage guide
│
├── .env.example                   # All PRISMAGE_* environment variables documented
├── docs/DESIGN.md                 # Detailed architecture and design document
├── pyproject.toml
└── requirements.txt
```

## Plugin System

The recommended way to onboard a new domain is to create a plugin — a self-contained
directory under `plugins/` that bundles metadata, prompts, and optional capability overrides.

Two plugin modes are supported: **SQL** (default) for live database queries, and **Embedding**
for pre-aggregated vector datasets.

### SQL Plugin

```bash
# 1. Copy the starter template
cp -r plugins/empty-plugin plugins/my-plugin

# 2. Edit plugin.json — set name and description (mode defaults to "sql")
# 3. Fill in config/metadata/ JSON files (dimensions, metrics, formulas, tables, rules)
# 4. Tune config/prompts/ templates
# 5. (Optional) override SQL behaviours in capabilities.py
```

**Load and query:**

```python
from api.chatbot import build_plugin_engine

engine = build_plugin_engine("my-plugin")
response = engine.answer("Show top 5 products by revenue this month")
print(response.summary)
print(response.detail)
```

### Embedding Plugin

For domains where data is pre-aggregated and stored as vector embeddings:

**`plugin.json`:**
```json
{
  "name": "my-embedding-plugin",
  "mode": "embedding",
  "namespace": "my_namespace",
  "llm_model": "gpt-4o-mini"
}
```

**`config/schema.json`** — defines dimensions, granularities, and search parameters.
See [DESIGN.md](docs/DESIGN.md) → Embedding Plugin Engine for the full schema reference.

**`config/prompts.json`** — LLM prompt templates for question extraction, batch analysis,
and multi-granularity synthesis.

**Load and query:**

```python
from api.chatbot import build_plugin_engine

engine = build_plugin_engine("my-embedding-plugin")
response = engine.answer("Ask a question about your domain")
print(response.summary)
print(response.detail)
```

### Load all plugins at once

```python
from api.chatbot import build_multi_engine

registry = build_multi_engine()   # auto-discovers all plugins/ (both SQL and embedding)
response = registry.answer("my-plugin", "Top 5 products by revenue")
```

### Use without the plugin system (SQL only)

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

# Named SQL plugin
python -m api.chatbot --plugin my-sql-plugin

# Named SQL plugin — also print the generated SQL for every question
python -m api.chatbot --plugin my-sql-plugin --include-sql

# Named SQL plugin — generate a Vega-Lite chart spec after every answer
python -m api.chatbot --plugin my-sql-plugin --chart

# Combine SQL + chart output
python -m api.chatbot --plugin my-sql-plugin --include-sql --chart

# Named embedding plugin — interactive session
python -m api.chatbot --plugin my-embedding-plugin

# Named embedding plugin — single question (non-interactive) with chart
python -m api.chatbot --plugin my-embedding-plugin --question "Which items are below threshold?" --chart
```

**CLI flags:**

| Flag | Description |
|---|---|
| `--plugin NAME` | Load a named plugin from `plugins/<NAME>/` |
| `--include-sql` | Print the SQL query (or queries) generated for each question (SQL plugins only) |
| `--chart` | Generate and print a Vega-Lite v5 chart spec after each answer |
| `--question TEXT` | Run a single question and exit (non-interactive mode) |

When `--include-sql` is set, a **SQL QUERIES** block is printed after each response showing the exact SQL sent to the database, labelled by channel (primary / secondary). Useful for debugging parser output and verifying HAVING, GROUP BY, and WHERE clauses.

When `--chart` is set, a **CHART SPEC (VEGA-LITE)** block is printed after the answer containing a complete Vega-Lite v5 JSON spec with `$schema` and `data.values` already injected. If the response has multiple result groups (e.g. primary and secondary channels), one chart spec is printed per group, each labelled with its channel. The spec can be pasted directly into the [chart-preview UI](chart-preview/README.md) at `http://localhost:8050` to render the chart interactively.

**How chart generation works:**

- The LLM generates a spec skeleton (mark + encoding only — no `$schema`, no `data`).
- Pydantic validates the skeleton; `$schema` and `data.values` are injected programmatically.
- Post-processing applies data-driven heuristics to fix common LLM mistakes:
  - Flips vertical bars to horizontal when labels exceed 10 characters.
  - Assigns the correct nominal field to y-axis vs. color using a within-group variation algorithm — no field names are hardcoded.
  - Adds `yOffset` for side-by-side grouped bars when there are multiple nominal dimensions.
  - Injects tooltip encoding for all columns.
  - Enforces legend at bottom for all color/size/shape encodings.
- For multi-metric data (`_val`/`_vol` column pairs), the engine folds wide-format rows into long format (`metric`, `amount`) and re-runs the LLM with the simplified schema before applying heuristics.
- For period-comparison data (`period1_value`/`period2_value`), the engine folds into a `period`/`value` long format and re-runs if the LLM skips the fold transform.

**Chart spec examples** — paste any of these into the chart-preview UI at `http://localhost:8050`:

<details>
<summary>i. Single metric — revenue by region (horizontal bar)</summary>

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "title": "Revenue by Region (MTD)",
  "mark": { "type": "bar", "orient": "horizontal" },
  "encoding": {
    "y": { "field": "region", "type": "nominal" },
    "x": { "field": "revenue", "type": "quantitative", "title": "Revenue (₹ Lakh)" },
    "color": { "field": "region", "type": "nominal", "legend": { "orient": "bottom" } },
    "tooltip": [
      { "field": "region",  "type": "nominal" },
      { "field": "revenue", "type": "quantitative" }
    ]
  },
  "data": {
    "values": [
      { "region": "North",  "revenue": 42.3 },
      { "region": "South",  "revenue": 38.7 },
      { "region": "East",   "revenue": 27.1 },
      { "region": "West",   "revenue": 51.5 },
      { "region": "Central","revenue": 19.8 }
    ]
  }
}
```
</details>

<details>
<summary>ii. Multi-metric — actual vs target vs last year side-by-side per product</summary>

Wide-format `_val`/`_vol` pairs are automatically folded into `metric` + `amount` long format.
The post-processing assigns `metric` to `color`+`yOffset` and the primary business dimension
to the y-axis, giving one group of side-by-side bars per product.

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "title": "Actual vs Target vs Last Year by Product",
  "mark": { "type": "bar", "orient": "horizontal" },
  "encoding": {
    "y":       { "field": "product",  "type": "nominal" },
    "x":       { "field": "amount",   "type": "quantitative", "title": "Units Sold" },
    "color":   { "field": "metric",   "type": "nominal", "legend": { "orient": "bottom" } },
    "yOffset": { "field": "metric",   "type": "nominal" },
    "tooltip": [
      { "field": "product", "type": "nominal" },
      { "field": "metric",  "type": "nominal" },
      { "field": "amount",  "type": "quantitative" }
    ]
  },
  "data": {
    "values": [
      { "product": "Widget A", "metric": "actual", "amount": 320 },
      { "product": "Widget A", "metric": "target", "amount": 400 },
      { "product": "Widget A", "metric": "last_year", "amount": 290 },
      { "product": "Widget B", "metric": "actual", "amount": 180 },
      { "product": "Widget B", "metric": "target", "amount": 200 },
      { "product": "Widget B", "metric": "last_year", "amount": 210 },
      { "product": "Widget C", "metric": "actual", "amount": 540 },
      { "product": "Widget C", "metric": "target", "amount": 500 },
      { "product": "Widget C", "metric": "last_year", "amount": 480 }
    ]
  }
}
```
</details>

<details>
<summary>iii. Period comparison — current vs last year MTD per region</summary>

`period1_value`/`period2_value` columns are folded into `period`+`value` long format.
Each region gets two side-by-side bars, one per period.

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "title": "Revenue: Current MTD vs Last Year MTD by Region",
  "mark": { "type": "bar", "orient": "horizontal" },
  "encoding": {
    "y":       { "field": "region", "type": "nominal" },
    "x":       { "field": "value",  "type": "quantitative", "title": "Revenue (₹ Lakh)" },
    "color":   { "field": "period", "type": "nominal", "legend": { "orient": "bottom" } },
    "yOffset": { "field": "period", "type": "nominal" },
    "tooltip": [
      { "field": "region", "type": "nominal" },
      { "field": "period", "type": "nominal" },
      { "field": "value",  "type": "quantitative" }
    ]
  },
  "data": {
    "values": [
      { "region": "North",  "period": "Jun 2025", "value": 42.3 },
      { "region": "North",  "period": "Jun 2024", "value": 38.1 },
      { "region": "South",  "period": "Jun 2025", "value": 38.7 },
      { "region": "South",  "period": "Jun 2024", "value": 41.2 },
      { "region": "East",   "period": "Jun 2025", "value": 27.1 },
      { "region": "East",   "period": "Jun 2024", "value": 24.9 },
      { "region": "West",   "period": "Jun 2025", "value": 51.5 },
      { "region": "West",   "period": "Jun 2024", "value": 47.8 }
    ]
  }
}
```
</details>

<details>
<summary>iv. Multi-dimension — city × region grouped bar</summary>

When there are two nominal dimensions, the post-processing algorithm assigns the dimension with
highest within-group variation to `color`+`yOffset` and the most-granular outer dimension to
the y-axis rows. Here `region` (4 distinct) drives color and `city` (many distinct) drives rows,
giving each city 1 bar per region.

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "title": "MTD Revenue by City and Region",
  "mark": { "type": "bar", "orient": "horizontal" },
  "encoding": {
    "y":       { "field": "city",    "type": "nominal" },
    "x":       { "field": "revenue", "type": "quantitative", "title": "Revenue (₹ Lakh)" },
    "color":   { "field": "region",  "type": "nominal", "legend": { "orient": "bottom" } },
    "yOffset": { "field": "region",  "type": "nominal" },
    "tooltip": [
      { "field": "city",    "type": "nominal" },
      { "field": "region",  "type": "nominal" },
      { "field": "revenue", "type": "quantitative" }
    ]
  },
  "data": {
    "values": [
      { "city": "Mumbai",    "region": "West",  "revenue": 28.4 },
      { "city": "Delhi",     "region": "North", "revenue": 22.1 },
      { "city": "Bangalore", "region": "South", "revenue": 19.7 },
      { "city": "Chennai",   "region": "South", "revenue": 14.3 },
      { "city": "Pune",      "region": "West",  "revenue": 12.9 },
      { "city": "Hyderabad", "region": "South", "revenue": 11.5 },
      { "city": "Kolkata",   "region": "East",  "revenue": 10.8 },
      { "city": "Ahmedabad", "region": "West",  "revenue":  9.6 }
    ]
  }
}
```
</details>

For embedding plugins, the output includes a **QUICK SUMMARY** (direct answer) and an **ANALYSIS (CONCISE)** section with per-granularity metric breakdowns, dimension values, period-over-period changes, and a data period indicator showing what date range was used.

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
