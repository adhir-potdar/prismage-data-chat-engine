# prismage-data-chat-engine — Design Document

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Pipeline Stages](#pipeline-stages)
4. [Metadata Schema](#metadata-schema)
5. [Business Rules Engine](#business-rules-engine)
6. [Query Builder](#query-builder)
7. [Formula Engine](#formula-engine)
8. [HAVING Engine](#having-engine)
9. [Table Router](#table-router)
10. [Embedding Router](#embedding-router)
11. [LangChain Integration](#langchain-integration)
12. [LangSmith Integration](#langsmith-integration)
13. [Metadata Registry](#metadata-registry)
14. [Prompt System](#prompt-system)
15. [Programmatic Enumeration](#programmatic-enumeration)
16. [Onboarding a New Domain](#onboarding-a-new-domain)
17. [Plugin System](#plugin-system)
18. [Engine Capabilities](#engine-capabilities)
19. [Embedding Plugin Engine](#embedding-plugin-engine)

---

## Overview

`prismage-data-chat-engine` is a generic, metadata-driven data chatbot engine. The engine
converts natural language questions into SQL queries and produces human-readable answers,
without any hardcoding of domain-specific knowledge in Python. All domain knowledge lives
in JSON configuration files.

**Design principles:**

- **Zero hardcoding** — all dimensions, metrics, formulas, tables, and business rules
  are defined in config files. The engine is domain-agnostic.
- **Deterministic query building** — SQL is assembled by a metadata-driven builder, not
  an LLM, ensuring predictable, formula-aware output.
- **LLM at the boundaries** — LLMs are used only for natural language understanding (Stage 1)
  and natural language generation (Stage 4). SQL construction (Stage 2/3) is fully programmatic.
- **Extensible business rules** — trigger-action rules enrich the parsed intent before
  query building, allowing domain experts to encode analytical patterns without code changes.

---

## Architecture Diagram

### Layer Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         CONFIGURATION LAYER                                 ║
║                                                                              ║
║  config/metadata/                          config/prompts/                   ║
║  ┌──────────────┐  ┌──────────────┐        ┌─────────────────────────────┐  ║
║  │dimensions    │  │metrics       │        │question_parser.json         │  ║
║  │.json         │  │.json         │        │query_generator.json         │  ║
║  ├──────────────┤  ├──────────────┤        │nl_response.json             │  ║
║  │formulas      │  │tables        │        └─────────────────────────────┘  ║
║  │.json         │  │.json         │                    │                    ║
║  ├──────────────┤  └──────────────┘                    │                    ║
║  │business_     │                                      │                    ║
║  │rules.json    │                                      │                    ║
║  └──────────────┘                                      │                    ║
║        │                                               │                    ║
╚════════╪═══════════════════════════════════════════════╪════════════════════╝
         │                                               │
         ▼                                               ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                           ENGINE LAYER                                       ║
║                                                                              ║
║  ┌─────────────────────────────┐     ┌────────────────────────────────────┐ ║
║  │     METADATA SUBSYSTEM      │     │        PROMPT SUBSYSTEM            │ ║
║  │                             │     │                                    │ ║
║  │  MetadataLoader             │     │  PromptLibrary                     │ ║
║  │    └─► MetadataConfig       │     │   (JSON files or LangSmith Hub)    │ ║
║  │          │                  │     │                                    │ ║
║  │          ▼                  │     │  PromptBuilder                     │ ║
║  │  MetadataValidator          │     │   (Jinja2 + live metadata inject)  │ ║
║  │    (startup checks)         │     └────────────────────────────────────┘ ║
║  │          │                  │                                            ║
║  │          ▼                  │                                            ║
║  │  MetadataRegistry           │                                            ║
║  │   alias index (O(1))        │                                            ║
║  │   render_*() helpers        │                                            ║
║  │   register/override formula │                                            ║
║  └──────────────┬──────────────┘                                            ║
║                 │                                                            ║
║                 │ (registry shared by all subsystems below)                 ║
║                 │                                                            ║
║  ┌──────────────▼──────────────┐     ┌────────────────────────────────────┐ ║
║  │     RULES SUBSYSTEM         │     │       QUERY SUBSYSTEM              │ ║
║  │                             │     │                                    │ ║
║  │  BusinessRulesEngine        │     │  EmbeddingTableRouter  (default)   │ ║
║  │   trigger evaluators:       │     │   MetadataEmbeddingStore           │ ║
║  │    keyword                  │     │   cosine similarity search         │ ║
║  │    having_pattern           │     │   on-disk embedding cache          │ ║
║  │    formula_requested        │     │                 OR                 │ ║
║  │    metric_present           │     │  TableRouter  (affinity mode)      │ ║
║  │    metric_category          │     │   table_affinity set intersection  │ ║
║  │    compound / any_of        │     │   channel filter                   │ ║
║  │   action handlers:          │     │                                    │ ║
║  │    ensure_metrics/dims      │     │  FormulaEngine                     │ ║
║  │    ensure/create/override   │     │   component → db_column resolve    │ ║
║  │    formula                  │     │   runtime_var injection            │ ║
║  │    set_output_hint          │     │                                    │ ║
║  │    set_channel              │     │  HavingEngine                      │ ║
║  │    set_sort                 │     │   vs_average / gap_to_target       │ ║
║  └─────────────────────────────┘     │   metric_comparison                │ ║
║                                      │                                    │ ║
║                                      │  QueryBuilder                      │ ║
║                                      │   SELECT / FROM / WHERE            │ ║
║                                      │   GROUP BY / HAVING / ORDER BY     │ ║
║                                      │   LIMIT (deterministic SQL)        │ ║
║                                      └────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════════╝
         │
         ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                        INFRASTRUCTURE LAYER                                  ║
║                                                                              ║
║  adapters/database.py          adapters/llm.py          adapters/embeddings.py║
║  SQLDatabase (LangChain)       ChatOpenAI /             OpenAIEmbeddings /   ║
║  via SQLAlchemy URI            ChatAnthropic            VoyageAIEmbeddings   ║
╚══════════════════════════════════════════════════════════════════════════════╝
         │
         ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                             API LAYER                                        ║
║                                                                              ║
║  api/chatbot.py  →  build_engine()  →  ChatbotChain.answer(question)         ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

### Runtime Pipeline (Question → Answer)

```
 USER QUESTION
      │
      ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │  ChatbotChain  (@traceable — LangSmith)                                    │
 │                                                                            │
 │  ┌──────────────────────────────────────────────────────────────────────┐  │
 │  │  STAGE 1 — QuestionParser                          [LLM]             │  │
 │  │                                                                      │  │
 │  │  PromptBuilder injects:                                              │  │
 │  │    • dimension list (render_dimensions)                              │  │
 │  │    • metric list (render_metrics)                                    │  │
 │  │    • domain_context from business_rules.json                         │  │
 │  │                                                                      │  │
 │  │  ChatPromptTemplate | LLM | PydanticOutputParser                     │  │
 │  │         └──────────────────────────────────► ParsedIntent (JSON)    │  │
 │  │                                              confidence: 0.0 – 1.0  │  │
 │  └───────────────────────────────────────────────────┬──────────────────┘  │
 │                                                      │                     │
 │                              confidence ≥ 0.7 ?      │                     │
 │                          ┌───────── YES ─────────────┘                     │
 │                          │                    NO ──────────────────────┐   │
 │                          ▼                                             │   │
 │  ┌───────────────────────────────────────────────┐                     │   │
 │  │  BusinessRulesEngine.enrich()  [Pure Python]  │                     │   │
 │  │                                               │                     │   │
 │  │  For each rule in business_rules.json:        │                     │   │
 │  │    evaluate trigger (keyword / having /        │                     │   │
 │  │      metric_category / compound …)            │                     │   │
 │  │    apply matching actions:                    │                     │   │
 │  │      ensure_metrics / ensure_formula          │                     │   │
 │  │      create_formula / override_formula        │                     │   │
 │  │      set_channel / set_sort / set_output_hint │                     │   │
 │  │    record applied_rules on intent             │                     │   │
 │  └───────────────────────┬───────────────────────┘                     │   │
 │                          │  enriched ParsedIntent                      │   │
 │                          ▼                                             │   │
 │  ┌───────────────────────────────────────────────┐                     │   │
 │  │  STAGE 2 — QueryBuilderStage  [Pure Python]   │                     │   │
 │  │                                               │                     │   │
 │  │  EmbeddingTableRouter (default)               │                     │   │
 │  │    embed intent terms → cosine similarity     │                     │   │
 │  │    top-K tables ranked by score               │                     │   │
 │  │         OR                                    │                     │   │
 │  │  TableRouter (affinity mode)                  │                     │   │
 │  │    intersect table_affinity sets              │                     │   │
 │  │    filter by intent.channels                  │                     │   │
 │  │    → list[TableGroup]                         │                     │   │
 │  │                                               │                     │   │
 │  │  For each TableGroup:                         │                     │   │
 │  │    SELECT  ← dims + metrics + formulas        │                     │   │
 │  │    FROM    ← table.name                       │                     │   │
 │  │    WHERE   ← filters + date range BETWEEN     │                     │   │
 │  │    GROUP BY← dimension db_columns             │                     │   │
 │  │    HAVING  ← HavingEngine (pattern dispatch)  │                     │   │
 │  │    ORDER BY← intent.sort                      │                     │   │
 │  │    LIMIT   ← intent.limit or 100              │                     │   │
 │  │                                               │                     │   │
 │  │  FormulaEngine                                │                     │   │
 │  │    {component} → metric.db_column             │                     │   │
 │  │    {runtime_var} → QueryContext values        │                     │   │
 │  │    → expanded SQL expression                  │                     │   │
 │  │                                               │                     │   │
 │  └───────────────────────┬───────────────────────┘                     │   │
 │                          │  list[BuiltQuery]                           │   │
 │                          │  no queries? ─────────────────────────────► │   │
 │                          ▼                                             │   │
 │  ┌───────────────────────────────────────────────┐                     │   │
 │  │  STAGE 3 — QueryExecutor    [LangChain]       │                     │   │
 │  │                                               │                     │   │
 │  │  QuerySQLDataBaseTool.run(sql)                │                     │   │
 │  │    └─► SQLDatabase (SQLAlchemy)               │                     │   │
 │  │          └─► database rows                    │                     │   │
 │  │                                               │                     │   │
 │  │  Build programmatic enumeration:              │                     │   │
 │  │    "Row 1: dim_a=value_1, metric_a=1250 …"    │                     │   │
 │  │    (row-by-row text, prevents LLM filtering)  │                     │   │
 │  └───────────────────────┬───────────────────────┘                     │   │
 │                          │  enumeration + output_hints                 │   │
 │                          ▼                                             │   │
 │  ┌───────────────────────────────────────────────┐                     │   │
 │  │  STAGE 4 — NLResponder              [LLM]     │                     │   │
 │  │                                               │                     │   │
 │  │  Prompt: enumeration + original question      │                     │   │
 │  │          + output format hint                 │                     │   │
 │  │  Instruction: include ALL rows, do not        │                     │   │
 │  │    re-filter or re-rank                       │                     │   │
 │  │  LLM produces human-readable answer           │                     │   │
 │  └───────────────────────┬───────────────────────┘                     │   │
 │                          │                                             │   │
 │                          ▼                    ┌────────────────────────┘   │
 │                    ChatResponse         ┌─────▼────────────────────────┐   │
 │                    .answer              │  FALLBACK (@traceable)       │   │
 │                    .applied_rules       │  create_sql_query_chain      │   │
 │                    .used_fallback       │  (LangChain LLM SQL chain)   │   │
 │                    .sql_queries         │  tagged in LangSmith for     │   │
 │                                         │  metadata coverage review    │   │
 │                                         └──────────────────────────────┘   │
 └────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
 NL ANSWER
```

---

### LangChain & LangSmith Integration Points

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │  LangChain Usage                                                     │
 │                                                                      │
 │  Stage 1  ChatPromptTemplate ──► LLM ──► PydanticOutputParser       │
 │           (langchain_core)        │       (ParsedIntent)            │
 │                                   └─► LangSmith auto-trace          │
 │                                                                      │
 │  Stage 3  QuerySQLDataBaseTool.run(sql)                              │
 │           (langchain_community.tools.sql_database)                  │
 │                                                                      │
 │  Stage 4  ChatPromptTemplate ──► LLM ──► StrOutputParser            │
 │           (langchain_core)        └─► LangSmith auto-trace          │
 │                                                                      │
 │  Fallback create_sql_query_chain(llm, db)                           │
 │           (langchain_community.chains)                              │
 │           @traceable(name="fallback_sql_chain")                     │
 └──────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────────────────────────────────────────────────────┐
 │  LangSmith Observability                                             │
 │                                                                      │
 │  ChatbotChain.answer()       @traceable  ← full request trace       │
 │  ChatbotChain._run_fallback()@traceable  ← fallback flagged         │
 │                                                                      │
 │  Captured per trace:                                                 │
 │    • question (input)                                               │
 │    • ParsedIntent (dimensions, metrics, confidence)                 │
 │    • applied_rules                                                  │
 │    • SQL queries built                                              │
 │    • row_count from execution                                       │
 │    • used_fallback flag                                             │
 │                                                                      │
 │  Prompt versioning: PromptLibrary loads from LangSmith Hub          │
 │    source: hub:org/question-parser-v2                               │
 │                                                                      │
 │  Regression datasets: 112 test questions → LangSmith dataset        │
 │    langsmith evaluate --dataset "prismage-domain-qa"             │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1 — QuestionParser (`engine/pipeline/question_parser.py`)

Converts a natural language question into a structured `ParsedIntent` using an LLM.

**Flow:**
1. `PromptBuilder.build_parser_prompt()` injects live metadata (dimension list, metric list,
   domain context) into the Jinja2 template from `config/prompts/question_parser.json`.
2. `ChatPromptTemplate | LLM | PydanticOutputParser` chain produces a `ParsedIntent`.
3. If `intent.confidence < 0.7`, `use_fallback=True` is returned and the chatbot chain
   routes to the LangChain fallback instead of Stage 2.

**ParsedIntent fields (from Stage 1):**
- `dimensions` — canonical dimension names requested
- `metrics` — canonical metric names (absolute/average/cumulative)
- `formula_metrics` — percentage/formula metric names
- `having` — optional `HavingConfig` (type + polarity + conditions)
- `filters` — dimension equality filters (e.g. `{"dim_name": "value_x"}`)
- `date_range` — optional `DateRange(start, end)` in `YYYYMMDD`
- `sort` — optional `SortConfig(metric, direction)`
- `limit` — optional row limit
- `confidence` — float 0.0–1.0 from the LLM

### Business Rules Enrichment (between Stage 1 and Stage 2)

`BusinessRulesEngine.enrich()` iterates all rules from `business_rules.json` and applies
matching rules to the `ParsedIntent` before query building. See [Business Rules Engine](#business-rules-engine).

**Added to ParsedIntent by rules:**
- `channels` — table channel filter (e.g. `["primary", "secondary"]`)
- `output_hints` — format, column ordering hints for Stage 4
- `applied_rules` — names of rules that fired (for debugging + LangSmith)
- `inline_formulas` — formulas created dynamically by rules
- `formula_overrides` — formula expressions overridden for this request

### Stage 2 — QueryBuilderStage (`engine/pipeline/query_builder.py`)

Converts an enriched `ParsedIntent` into a list of `BuiltQuery` objects — one per resolved table.
Pure Python, no LLM. See [Query Builder](#query-builder).

### Stage 3 — QueryExecutor (`engine/pipeline/query_executor.py`)

Executes each `BuiltQuery` using LangChain's `QuerySQLDataBaseTool` and builds a
**programmatic enumeration** — a row-by-row text representation passed to the LLM in Stage 4
instead of raw SQL output. This prevents LLM re-filtering and ensures all rows are included
in the answer.

### Stage 4 — NLResponder (`engine/pipeline/nl_responder.py`)

Uses an LLM with the programmatic enumeration + original question + output hints to produce
a natural language answer. The prompt explicitly instructs the LLM not to re-filter or omit
rows — its job is interpretation and presentation only.

---

## Metadata Schema

All metadata lives under `config/metadata/` (or a domain-specific override directory).

### `dimensions.json`

Each dimension represents a grouping column (e.g. category, location, segment).

```json
{
  "name": "dim_name",
  "aliases": ["alias_1", "alias_2"],
  "db_column": "dim_column",
  "table_affinity": ["table_a", "table_b"],
  "hierarchy_name": "hierarchy_group",
  "hierarchy_level": 1
}
```

### `metrics.json`

Five metric categories are supported:

| Category    | Aggregation | Typical Use |
|-------------|-------------|-------------|
| `absolute`  | SUM / COUNT | Revenue, quantity, orders |
| `average`   | AVG         | Satisfaction score, delivery days |
| `cumulative`| SUM         | MTD, YTD, QTD running totals |
| `percentage`| (formula)   | Achievement %, growth %, margin % |
| `formula`   | (formula)   | AOV, run rate, gap-to-target |

```json
{ "name": "metric_name", "aliases": ["alias_a", "alias_b"],
  "db_column": "metric_col", "aggregate_fn": "SUM",
  "category": "absolute", "table_affinity": ["table_a", "table_b"] }

{ "name": "formula_metric", "aliases": ["alias_c", "alias_d %"],
  "db_column": null, "aggregate_fn": null,
  "category": "percentage", "formula_ref": "formula_metric",
  "table_affinity": ["table_a"] }
```

### `formulas.json`

SQL expression templates for percentage and formula metrics.

```json
{
  "name": "formula_metric",
  "display": "Formula Metric %",
  "expression": "(SUM({metric_a}) - SUM({metric_b})) / NULLIF(SUM({metric_a}), 0) * 100",
  "components": ["metric_a", "metric_b"],
  "runtime_vars": [],
  "window": false
}
```

**Formula placeholders** are resolved by the `FormulaEngine`:
- `{revenue}` → `db_column` of the `revenue` metric
- `{days_elapsed}` → value from `QueryContext.days_elapsed`

### `tables.json`

Defines database tables and which dimensions/metrics each contains.

```json
{
  "name": "fact_table",
  "channel": "primary",
  "date_column": "date_col",
  "description": "Main fact table",
  "dimensions": ["dim_a", "dim_b", "dim_c"],
  "metrics": ["metric_name", "cumulative_metric", "target_metric"]
}
```

### `business_rules.json`

Contains domain context, metric hints for the parser, HAVING clause patterns,
and trigger-action rules. See [Business Rules Engine](#business-rules-engine).

---

## Business Rules Engine

`BusinessRulesEngine` (`engine/rules/engine.py`) runs between Stage 1 and Stage 2.
It iterates all rules and, for each matching trigger, applies the configured actions to
the `ParsedIntent` in place.

### Trigger Types

| Type | Description | Required fields |
|------|-------------|-----------------|
| `keyword` | Fires if any phrase appears in the question (case-insensitive) | `phrases` |
| `having_pattern` | Fires if intent.having matches type + optional polarity | `having_type`, optional `polarity` |
| `formula_requested` | Fires if a specific formula is in `formula_metrics` | `formula` |
| `metric_present` | Fires if any of the listed metrics is in intent | `metrics` |
| `metric_category` | Fires if any intent metric belongs to a category | `category` |
| `compound` | Fires only if ALL sub-triggers match (AND logic) | `conditions` |
| `any_of` | Fires if ANY sub-trigger matches (OR logic) | `conditions` |

### Action Types

| Type | Description | Required fields |
|------|-------------|-----------------|
| `ensure_metrics` | Adds metrics to intent if not already present | `metrics` |
| `ensure_dimensions` | Adds dimensions to intent if not already present | `dimensions` |
| `ensure_formula` | Adds a formula metric to intent if not already present | `formula` |
| `create_formula` | Registers a new inline formula and adds it to intent | `name`, `expression`, `components` |
| `override_formula` | Replaces an existing formula definition for this request | `formula`, `expression`, `components` |
| `set_output_hint` | Sets output format and column hints for Stage 4 | `format` |
| `set_channel` | Restricts table routing to specific channels | `channels` |
| `set_sort` | Sets ORDER BY metric and direction | `metric`, `direction` |

### Example Rule

```json
{
  "name": "highlight_adds_achievement_pct",
  "trigger": { "type": "keyword", "phrases": ["highlight", "underperforming"] },
  "actions": [{ "type": "ensure_formula", "formula": "target_achievement_pct" }]
}
```

This fires when the user question contains "highlight" or "underperforming" and adds
`target_achievement_pct` to `intent.formula_metrics` before SQL is built.

---

## Query Builder

`QueryBuilder` (`engine/query/builder.py`) assembles SQL deterministically from the
enriched `ParsedIntent`. No LLM is involved.

**Pipeline per resolved table:**

```
SELECT  ← dimensions (no agg) + absolute/avg/cumulative metrics (agg fn) + formula expansions
FROM    ← table name
WHERE   ← dimension equality filters + date range BETWEEN
GROUP BY← dimension db_columns
HAVING  ← from HavingEngine (metric_comparison / vs_average / gap_to_target)
ORDER BY← from intent.sort
LIMIT   ← from intent.limit or default 100
```

**Metric aggregation mapping:**

- `absolute` / `cumulative` → `SUM(db_column) AS metric_name`
- `average` → `AVG(db_column) AS metric_name`
- `percentage` / `formula` → expanded SQL expression from `FormulaEngine`

**Table filtering:** only dimensions/metrics actually present in the table's membership
lists are included; unknown metrics for a table are silently skipped.

---

## Formula Engine

`FormulaEngine` (`engine/query/formula_engine.py`) resolves formula expression templates
into executable SQL.

**Resolution steps:**
1. Look up the `Formula` by name in the registry.
2. For each `{component}` placeholder, resolve the metric's `db_column` from the registry.
3. For each `{runtime_var}` (e.g. `{days_elapsed}`), inject the value from `QueryContext`.
4. Return the expanded SQL expression string.

**QueryContext fields:**
- `days_elapsed` — days elapsed in the current period
- `total_days` — total calendar days in the period
- `days_remaining` — `total_days - days_elapsed`

Example expansion:
```
Template: "SUM({metric_a}) / NULLIF({days_elapsed}, 0) * {total_days}"
  → "SUM(metric_a_col) / NULLIF(15, 0) * 31"
```

---

## HAVING Engine

`HavingEngine` (`engine/query/having_engine.py`) builds HAVING clauses from the
`HavingConfig` attached to a `ParsedIntent`.

**Pattern types** (defined in `business_rules.json` under `having_patterns`):

| Type | Example Use | SQL Template |
|------|-------------|--------------|
| `vs_average` | "above average revenue" | `HAVING SUM(revenue) > (SELECT AVG(revenue) FROM orders)` |
| `gap_to_target` | "gap to target > 10000" | `HAVING (SUM(target) - SUM(revenue)) > 10000` |
| `metric_comparison` | "both below target and last year" | `HAVING SUM(current_metric) < SUM(target) AND SUM(current_metric) < SUM(prior_metric)` |

Multi-condition patterns are joined with `AND` or `OR` as configured in
`having_patterns[*].multi_condition_join`.

---

## Table Router

Two routing modes are available. The active router is selected via `router_mode` in
`build_engine()` or the `PRISMAGE_ROUTER_MODE` environment variable.

### Affinity Router — `router_mode="affinity"` (`engine/query/router.py`)

Resolves tables by set intersection of `table_affinity` lists declared in JSON config.

**Algorithm:**
1. Collect `table_affinity` sets for all requested dimensions and metrics.
2. Find tables that appear in all affinity sets (intersection).
3. Filter by `intent.channels` if set by a business rule.
4. Return a `TableGroup` list (table name + channel).

Best for: fully enumerated configs, offline/no-API environments, strict determinism.

---

## Embedding Router

### Semantic Router — `router_mode="embedding"` (default) (`engine/query/embedding_router.py`)

Resolves tables by cosine similarity between the intent terms and embedded table metadata
documents. More robust for questions with unfamiliar phrasing or incomplete `table_affinity`
lists.

**Components:**

`MetadataEmbeddingStore` (`engine/metadata/embedding_store.py`):
- Builds one text document per table at startup:
  ```
  Table: fact_table | Channel: primary | Description: Main fact table |
  Dimensions: dim_a(alias_1, alias_2), dim_b(alias_3, alias_4) |
  Metrics: metric_a(alias_5, alias_6), metric_b(alias_7)
  ```
  Dimension and metric aliases (up to 3 per item) are included so the embedding
  captures natural language synonyms, not just canonical names.
- Embeds all documents using the configured embeddings model
  (default: OpenAI `text-embedding-3-small`).
- Caches vectors to disk as JSON (default: `.cache/metadata_embeddings.json`).
  The cache is keyed by an MD5 hash of all document content — rebuilt automatically
  when any dimension, metric, or table config changes.

**Cache lifecycle:**

```
store.build()  (called once at engine startup)
  │
  ├─ _build_documents()        always runs — builds text docs, no API call
  │
  ├─ cache file exists?
  │     YES → read JSON → compare stored MD5 hash vs current doc hash
  │               MATCH    → load vectors from file, done  ← zero API calls
  │               MISMATCH → fall through to embed
  │               CORRUPT  → silently fall through to embed
  │     NO  → fall through to embed
  │
  └─ _embed_and_cache()
        embeddings.embed_documents([N table docs])  ← ONE OpenAI call, N texts
        write {hash, vectors} to cache file
```

**Per-question cost:**

| Operation | When | OpenAI call? |
|---|---|---|
| `embed_documents` (table vectors) | Once at first startup, or after config change | Yes — N texts in one call |
| Load from cache | Every restart after first build | No |
| `embed_query` (question vector) | Every question | Yes — 1 text per query |

Table vectors are never re-fetched from the API at runtime — only the question embedding
is unavoidable per query. `invalidate_cache()` forces a full rebuild on the next
`build()` call.

`EmbeddingTableRouter` (`engine/query/embedding_router.py`):
- On every question, embeds the user's raw question text via `store.find_tables(question, top_k)`.
- Inside `find_tables`: calls `embeddings.embed_query(question)` then computes cosine
  similarity between the question vector and every cached table vector.
- Returns up to `top_k` table names sorted by descending similarity score.
- `EmbeddingTableRouter.resolve()` then applies `intent.channels` filter to the ranked
  list and returns matching tables as `TableGroup` objects for the query builder.

Note: the question is embedded directly (raw text), not the parsed intent fields.
This means table routing happens against the original phrasing before any LLM parsing.

**Choosing a mode:**

| | `embedding` (default) | `affinity` |
|---|---|---|
| Table selection | Cosine similarity on raw question | Exact set intersection on parsed intent |
| Requires API on startup | Yes (first run only, then cached) | No |
| Handles incomplete affinity lists | Yes | No |
| Deterministic | No (model-dependent) | Yes |
| Handles unfamiliar phrasing | Yes | Only if aliases cover the phrasing |

---

## LangChain Integration

LangChain is used at three points:

1. **Stage 1 — Question parsing**: `ChatPromptTemplate | LLM | PydanticOutputParser` chain.
   The LLM is given rendered metadata (dimension/metric lists) and returns a JSON
   `ParsedIntent` validated by Pydantic.

2. **Stage 3 — Query execution**: `QuerySQLDataBaseTool` from `langchain_community.tools`.
   Executes the SQL built in Stage 2 against the configured `SQLDatabase`.

3. **Fallback — LangChain SQL chain**: When Stage 1 confidence < 0.7 or no tables are
   resolved, `create_sql_query_chain` (from `langchain_community.chains`) takes over.
   This uses a full LLM-driven SQL generation approach. All fallback runs are tagged
   in LangSmith for metadata coverage review.

**Why not use LangChain's SQL chain as the primary builder?**

LangChain's SQL chain works well for simple queries but does not understand domain-specific
formula logic (e.g. run rate = MTD / days_elapsed × total_days), business rule enrichments,
or HAVING clause patterns. The metadata-driven builder is more predictable and formula-aware.
The LangChain chain serves as a safety net for questions that fall outside the metadata coverage.

---

## LangSmith Integration

```python
from langsmith import traceable

class ChatbotChain:
    @traceable(name="chatbot_answer")
    def answer(self, question: str) -> ChatResponse:
        ...

    @traceable(name="fallback_sql_chain")
    def _run_fallback(self, question: str) -> ChatResponse:
        ...
```

**Tracing details captured:**
- `question` input
- `ParsedIntent` from Stage 1 (serialized)
- `applied_rules` list
- SQL queries built
- Row count from execution
- `used_fallback` flag

**LangSmith Hub prompt versioning:**
`PromptLibrary` can load prompts from the LangSmith Hub by prefixing the prompt name
with `hub:`. This enables prompt version pinning and A/B testing:

```json
{
  "source": "langsmith_hub",
  "hub_ref": "your-org/question-parser-v2"
}
```

**Regression datasets:**
Existing test questions can be uploaded as a LangSmith dataset for automated evaluation.
Run `langsmith dataset create` with your question/answer pairs, then evaluate with
`langsmith evaluate`.

---

## Metadata Registry

`MetadataRegistry` (`engine/metadata/registry.py`) is the runtime lookup service
built from `MetadataConfig` at engine startup. All lookups are O(1) via pre-built
index dictionaries.

**Key methods:**

| Method | Purpose |
|--------|---------|
| `resolve_dimension_alias(phrase)` | "alias_1" → "dim_name" |
| `resolve_metric_alias(phrase)` | "alias_5" → "metric_name" |
| `get_db_column(dim_name)` | "dim_name" → "dim_column" (SQL column) |
| `get_metric_column(metric_name)` | "metric_name" → "metric_col" |
| `get_aggregate_fn(metric_name)` | "metric_name" → "SUM" |
| `get_metric_category(name)` | "cumulative_metric" → MetricCategory.CUMULATIVE |
| `table_has_dimension(table, dim)` | True if dim in table's dimension list |
| `table_has_metric(table, metric)` | True if metric in table's metric list |
| `register_formula(name, ...)` | Add an inline formula from a business rule |
| `override_formula(name, ...)` | Override formula expression (request-scoped) |
| `restore_formula_overrides()` | Restore originals after request completes |
| `render_dimensions()` | Human-readable list for prompt injection |
| `render_metrics()` | Human-readable list for prompt injection |

---

## Prompt System

### PromptLibrary (`engine/prompts/prompt_library.py`)

Loads prompt templates from:
- Local JSON files (`config/prompts/*.json`)
- LangSmith Hub (`hub:org/prompt-name`)

### PromptBuilder (`engine/prompts/prompt_builder.py`)

Renders Jinja2 templates from the library, injecting live metadata:

```jinja2
You are a data analyst assistant. Available dimensions:
{{ dimensions }}

Available metrics:
{{ metrics }}

Domain context:
{{ domain_context }}

Question: {{ question }}
Return a JSON ParsedIntent.
```

The builder calls `registry.render_dimensions()` and `registry.render_metrics()` to
produce the enumerated lists injected into every Stage 1 prompt. This ensures the LLM
always sees the current, complete metadata without any caching issues.

---

## Programmatic Enumeration

Instead of passing raw SQL output (markdown tables, CSV) to the Stage 4 LLM, the executor
builds a **programmatic enumeration** — a row-by-row text listing:

```
Row 1: dim_a=value_1, metric_a=1250, formula_metric=32.4
Row 2: dim_a=value_2, metric_a=980, formula_metric=28.7
Row 3: dim_a=value_3, metric_a=1100, formula_metric=30.1
...
Total rows: 3
```

**Why this matters:**

When an LLM receives a markdown table, it tends to re-interpret it: omitting rows,
re-ranking, rounding values, or adding qualifiers like "(not available)". The enumerated
format treats each row as a named tuple, making it harder for the LLM to silently
filter or transform results. The Stage 4 prompt explicitly instructs the LLM to include
all rows and not re-filter.

---

## Onboarding a New Domain

### Step 1 — Copy the config template

```bash
cp -r config/ my_domain/config/
```

### Step 2 — Edit `dimensions.json`

List every column your users might group by. For each:
- Choose a canonical `name` (snake_case)
- Add natural language `aliases` that users commonly say
- Set `db_column` to the actual SQL column name
- Set `table_affinity` to the tables where this column exists

### Step 3 — Edit `metrics.json`

For each KPI:
- Choose `category`: `absolute`, `average`, `cumulative`, `percentage`, or `formula`
- For `absolute`/`average`/`cumulative`: set `db_column` and `aggregate_fn`
- For `percentage`/`formula`: set `formula_ref` pointing to a formula in `formulas.json`
- Set `table_affinity` to tables containing this metric

### Step 4 — Edit `formulas.json`

For each percentage/formula metric:
- Write the `expression` as a SQL template with `{component}` placeholders
- List `components` as canonical metric names (their `db_column` values fill placeholders)
- List `runtime_vars` if the formula needs `days_elapsed`, `total_days`, or `days_remaining`

### Step 5 — Edit `tables.json`

For each database table:
- List `dimensions` and `metrics` by their canonical names
- Set `channel` if you have multiple parallel tables (e.g. primary/secondary)
- Set `date_column` for date range filtering

### Step 6 — Edit `business_rules.json`

- Write `domain_context` (a few sentences describing your data for the LLM)
- Add `metric_hints` to help the parser map natural language phrases to HAVING conditions
- Add `rules` for analytical patterns in your domain

### Step 7 — Wire it up

```python
from api.chatbot import build_engine

engine = build_engine(
    config_dir="my_domain/config/metadata",
    db_uri="postgresql://user:pass@host/mydb",
)

response = engine.answer("Show top 10 products by revenue this month")
print(response.answer)
```

### Step 8 — Validate at startup

The `MetadataValidator` runs automatically when `MetadataLoader.load()` is called.
It checks:
- All `formula_ref` values point to an existing formula
- All formula `components` point to existing metrics with `db_column`
- All table `dimensions`/`metrics` lists reference known canonical names
- All rule action `formula` and `metrics` references exist in metadata

Fix any errors reported before running the engine.

---

## Plugin System

The plugin system packages a domain config (metadata + prompts + capabilities) into a
self-contained directory that the engine loads by name. This separates domain knowledge
from engine code and allows multiple domains to be served from one engine instance.

Two plugin modes are supported: **SQL** (default) for live database queries, and
**Embedding** for pre-aggregated vector datasets. The mode is set by the `"mode"` field
in `plugin.json` (defaults to `"sql"` when absent).

### SQL plugin directory layout

```
plugins/
└── my-plugin/
    ├── plugin.json            # manifest — name, version, config/prompts paths
    ├── __init__.py            # Python package marker
    ├── capabilities.py        # optional — EngineCapabilities subclass
    ├── readme.txt             # plugin documentation
    ├── config/
    │   ├── metadata/          # same 5 JSON files as a standalone domain
    │   └── prompts/           # question_parser.json, nl_response.json
    ├── docs/                  # optional domain documentation
    └── tests/                 # optional test scripts
```

### Embedding plugin directory layout

```
plugins/
└── my-embedding-plugin/
    ├── plugin.json            # manifest — name, mode: "embedding", namespace, llm_model
    ├── config/
    │   ├── schema.json        # dimensions, granularities, search params, metric_names
    │   ├── prompts.json       # LLM prompt templates (question_extraction, batch_analysis, synthesis)
    │   └── kpi_metrics.csv    # metric definitions (Metric_Name, Metric_Type, Metric_Formula)
    └── docs/                  # optional domain documentation
```

**`plugin.json` fields — SQL mode:**

| Field | Required | Description |
|---|---|---|
| `name` | yes | Plugin identifier — must match directory name |
| `version` | yes | Semver string |
| `description` | yes | One-line summary |
| `config_dir` | yes | Relative path to metadata JSON directory |
| `prompts_dir` | yes | Relative path to prompt JSON directory |
| `capabilities` | no | Relative path to capabilities override file |
| `docs_dir` | no | Relative path to docs directory |
| `tests_dir` | no | Relative path to tests directory |

**`plugin.json` fields — Embedding mode:**

| Field | Required | Description |
|---|---|---|
| `name` | yes | Plugin identifier — must match directory name |
| `mode` | yes | Must be `"embedding"` |
| `namespace` | yes | Vector database namespace to search |
| `llm_model` | no | LLM model name (default: `gpt-4o-mini`) |
| `description` | no | One-line summary |

### Loading a plugin

**Single plugin:**

```python
from api.chatbot import build_plugin_engine

engine = build_plugin_engine("my-plugin")
response = engine.answer("Show top 5 products by revenue")
print(response.answer)
```

`build_plugin_engine(plugin, plugins_root="plugins")` resolves the plugin directory,
reads `plugin.json`, and delegates to `PluginLoader.load()`.

**All plugins at once:**

```python
from api.chatbot import build_multi_engine

registry = build_multi_engine()          # scans plugins/ for all plugin.json entries
response = registry.answer("my-plugin", "Show top 5 products by revenue")
```

`build_multi_engine()` returns a `PluginRegistry`. Every subdirectory under `plugins/`
that contains `plugin.json` is loaded automatically. Failed plugins are logged and skipped
so one broken plugin does not prevent others from loading.

### PluginLoader internals

`engine/plugins/loader.py` — `PluginLoader.load(plugin_dir)`:

**SQL mode** (`"mode"` absent or `"sql"`):

1. Reads `plugin.json` to resolve `config_dir` and `prompts_dir`.
2. Calls `_load_capabilities(plugin_path)` — see [Engine Capabilities](#engine-capabilities).
3. Delegates to `api.chatbot.build_engine(config_dir, prompts_dir, capabilities=...)`.
4. Logs the plugin name and resolved capabilities class.

**Embedding mode** (`"mode": "embedding"`):

1. Reads `plugin.json` to get `namespace`, `llm_model`, and optional `enable_charts`.
2. Reads `config/schema.json` and `config/prompts.json` from the plugin directory.
3. Constructs and returns an `EmbeddingChain` instance — see
   [Embedding Plugin Engine](#embedding-plugin-engine).

No plugin-specific logic lives in the loader — all engine wiring is handled by `build_engine()`
(SQL) or `EmbeddingChain.__init__()` (embedding).

### PluginRegistry

`engine/plugins/registry.py` — holds named `ChatbotChain` instances:

```python
registry.register("my-plugin", chain)  # called by build_multi_engine()
registry.names()                        # ["my-plugin", ...]
response = registry.answer("my-plugin", question)
```

If the plugin name is not found, `.answer()` returns a `ChatResponse` with
`success=False` and an informative error message listing available plugins.

### Starter template

`plugins/empty-plugin/` is a ready-to-copy starter template. Copy it, rename it, fill in
the five JSON config files, and optionally add capability overrides. See
`plugins/empty-plugin/readme.txt` for the Quick Start steps.

---

## Engine Capabilities

`EngineCapabilities` (`engine/capabilities/base.py`) is a base class that defines
overridable SQL-building behaviours. Each method has a generic default that is correct
for standard time-series tables. Plugins that require non-standard SQL (e.g. periodic
snapshot tables) subclass `EngineCapabilities` in their `capabilities.py` and override
only the methods they need.

### How capabilities are injected

```
build_plugin_engine("my-plugin")
    └─ PluginLoader.load()
           └─ _load_capabilities(plugin_path)
                  ├─ capabilities.py absent?  → EngineCapabilities()   (engine defaults)
                  ├─ capabilities.py present? → importlib loads module
                  │      → finds first class that is a proper subclass of EngineCapabilities
                  │      → instantiates it → MyPluginCapabilities()
                  └─ passes instance to build_engine(capabilities=MyPluginCapabilities())
                         └─ QueryBuilder(..., capabilities=MyPluginCapabilities())
                                └─ _build_where() delegates to capabilities methods
```

`importlib.util.spec_from_file_location` is used so `capabilities.py` does not need to
be on `sys.path`. If the file fails to import, a warning is logged and engine defaults
are used — the plugin still loads.

### Currently overridable methods

**`build_date_filter(table, date_col, date_range, table_meta) → str`**

Called when `intent.date_range` is set and `date_col` is known.

| | SQL |
|---|---|
| Default | `date_col BETWEEN 'start' AND 'end'` |
| Override example | `date_col = (SELECT MAX(date_col) FROM table WHERE date_col >= start AND date_col <= end)` |

When to override: your table stores periodic snapshots (one complete picture per load
date). `BETWEEN` would span multiple load dates and inflate totals. Use `MAX(date_col)
within range` to select only the latest snapshot in the requested period.

**`build_snapshot_filter(table, date_col, table_meta) → str`**

Called when no date range is given and the table has `date_mode = "snapshot"` in
`tables.json`.

| | SQL |
|---|---|
| Default | `date_col = (SELECT MAX(date_col) FROM table)` |

Override when you need a different "latest row" strategy (e.g. a separate audit table
that tracks the last loaded date per source table).

### Adding a new overridable behaviour

1. Add a method with a sensible generic default to `engine/capabilities/base.py`.
2. Replace the hardcoded logic in the relevant engine file with
   `self.capabilities.<method>(...)`.
3. Override the method in your plugin's `capabilities.py` when needed.

```python
# 1. engine/capabilities/base.py
def build_null_filter(self, col: str) -> str:
    return f"{col} IS NOT NULL"

# 2. engine/query/builder.py
conditions.append(self.capabilities.build_null_filter(col))

# 3. plugins/my-plugin/capabilities.py
def build_null_filter(self, col):
    return f"({col} IS NOT NULL AND {col} NOT IN ('', 'N/A', 'null'))"
```

### QueryBuilder integration points

`engine/query/builder.py` uses capabilities in `_build_where()`:

```
_build_where(table, intent)
    │
    ├─ intent.date_range set AND date_col known
    │       → capabilities.build_date_filter(table, date_col, intent.date_range, table_meta)
    │
    └─ no date_range AND table.date_mode == "snapshot"
            → capabilities.build_snapshot_filter(table, date_col, table_meta)
```

All other `QueryBuilder` logic (SELECT, GROUP BY, HAVING, ORDER BY, LIMIT) uses engine
defaults and is not routed through capabilities. Extend `EngineCapabilities` to make
additional clauses overridable.

---

## Embedding Plugin Engine

The Embedding Plugin Engine is an alternative pipeline for domains where data is
pre-aggregated and stored as vector embeddings rather than in a SQL database. It uses
semantic similarity search across time-granularity collections, runs parallel LLM
analysis, and synthesizes a multi-granularity answer.

**When to use:** any domain where data is pre-aggregated into periodic snapshots stored
as vector embeddings, and answers are assembled from those records rather than built
by SQL at query time.

### Components

| Component | File | Role |
|---|---|---|
| `EmbeddingChain` | `engine/chains/embedding_chain.py` | Top-level `.answer()` interface — wires all components |
| `QuestionParser` | `engine/embedding/question_parser.py` | Extracts metrics, dimensions, granularities, date range from question |
| `CollectionFinder` | `engine/embedding/collection_finder.py` | Selects vector collections by dimension + granularity + date range |
| `Searcher` | `engine/embedding/searcher.py` | Parallel vector similarity search across collections |
| `Analyzer` | `engine/embedding/analyzer.py` | Batch LLM analysis of search results per granularity |
| `Synthesizer` | `engine/embedding/synthesizer.py` | Per-granularity batch merging + final collective synthesis |
| `Orchestrator` | `engine/embedding/orchestrator.py` | 3-phase pipeline: search → analyze → synthesize |
| `date_utils` | `engine/embedding/date_utils.py` | Date parsing, collection name date extraction |

### Pipeline

```
USER QUESTION
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│  EmbeddingChain.answer()                                       │
│                                                                │
│  Step 1 — QuestionParser.parse()                  [LLM]       │
│    Extract: metrics, dimensions, granularities,               │
│             date range (raw text + YYYYMMDD bounds)           │
│                                                                │
│  Step 2 — QuestionParser.validate()               [Pure Python]│
│    Normalize metric names against kpi_metrics.csv             │
│    Reject if no valid metrics found                           │
│                                                                │
│  Step 3 — CollectionFinder                        [Pure Python]│
│    Select dimension (with fallback hierarchy)                 │
│    find_fast() — metadata service query                       │
│      OR find_slow() — scan pipeline collections              │
│    Filter by granularity + date range overlap                 │
│    → collections_by_granularity { gran: [coll, ...] }        │
│                                                                │
│  Step 4 — Orchestrator.run()                      [Async]     │
│    Phase 1: Parallel vector search across all granularities   │
│    Phase 2: Global top-K with granularity distribution        │
│    Phase 3: Batch analysis → collective synthesis             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
  ChatResponse(.answer, .summary, .detail)
```

### Orchestrator — 3-phase algorithm

**Phase 1 — Parallel search**

`Searcher.search_granularity()` runs concurrently for every granularity in
`collections_by_granularity`. Each call performs a vector similarity search against its
collections and returns `SearchResult(granularity, results=[(record, similarity), ...])`.

**Phase 2 — Global top-K with granularity distribution**

`_distribute_top_k()` selects the best `top_k_global` records distributed across
granularities:

1. Rank granularities by `(-result_count, -max_similarity, name_ASC)` — the alphabetical
   tie-breaker ensures deterministic output when multiple granularities have equal scores.
2. Guarantee at least one result from each of the top `max_granularities_for_top_results`
   granularities.
3. Fill remaining quota in priority order, deduplicating by `chunk_id`.

**Phase 3 — Analysis and synthesis**

In hybrid mode (default): split each granularity's results into batches of
`max_results_per_batch`, analyze all batches in parallel via `Analyzer.analyze_batch()`
(LLM call per batch), then combine batch insights per granularity.

In standard mode: analyze all results for each granularity together in one LLM call.

`Synthesizer.synthesize_collective()` combines all granularity insights into one
comprehensive answer with a structured header showing dimension, granularities analyzed,
and data period.

### `schema.json` — configuration reference

```json
{
  "dimensions": {
    "hierarchy": ["dim_a", "dim_b", "overall"],
    "fallback": {"dim_a": "overall"},
    "known_values": {"dim_b": ["VALUE_1", "VALUE_2"]}
  },
  "time_granularities": ["gran_a", "gran_b", "gran_c"],
  "granularity_names": {
    "gran_a": "Granularity A Label",
    "gran_b": "Granularity B Label",
    "gran_c": "Granularity C Label"
  },
  "search": {
    "top_k_global": 5,
    "max_granularities_for_top_results": 3,
    "max_results_per_batch": 2,
    "similarity_threshold": 0.2,
    "max_results_per_collection": 5,
    "enable_hybrid_parallelization": true,
    "enable_per_granularity_synthesis": false
  },
  "metrics_file": "config/kpi_metrics.csv"
}
```

| Key | Description |
|---|---|
| `dimensions.hierarchy` | Dimension resolution order — first available dimension wins |
| `dimensions.fallback` | Maps a dimension to its fallback when unavailable |
| `dimensions.known_values` | Allowed values per dimension for filtering |
| `time_granularities` | Ordered list of granularities to search (controls priority) |
| `granularity_names` | Human-readable labels used in output headers |
| `search.top_k_global` | Total result records selected across all granularities |
| `search.max_granularities_for_top_results` | Max granularities guaranteed a slot in top-K |
| `search.max_results_per_batch` | Records per LLM analysis batch |
| `search.similarity_threshold` | Minimum cosine similarity to include a result |
| `search.enable_hybrid_parallelization` | Batch analysis in parallel (default `true`) |
| `search.enable_per_granularity_synthesis` | Extra LLM call to merge batches per granularity |

### Data period display

The output header always includes a data period line derived from the collection names
actually used in the search. Collection names encode their date range in the format:
`{dim}_{gran}_{p1_start}_{p1_end}_vs_{p2_start}_{p2_end}_date_{date}`.

`date_utils.extract_dates_from_collection_name()` parses this pattern. The synthesizer
computes `min(all_period_starts)` to `max(all_period_ends)` across all filtered collections.

When the question specifies a date range, both the requested and actual data periods
are shown:

```
📅 Requested: <user date text> | Actual Data: <collection start> to <collection end>
```

When no date range is specified:

```
📅 Data Period: <collection start> to <collection end>
```

### Output format

`EmbeddingChain._split_answer()` splits the LLM output into `summary` (first paragraph)
and `detail` (remainder). `ChatResponse.summary` holds the quick answer; `ChatResponse.detail`
holds the full structured analysis with header and per-granularity sections.
