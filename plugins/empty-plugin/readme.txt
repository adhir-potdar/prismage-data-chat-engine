PLUGIN TEMPLATE — prismage-data-chat-engine
============================================

This directory is a starter template for creating a new plugin.
Copy this folder, rename it, and fill in the placeholders.


DIRECTORY STRUCTURE
-------------------
your-plugin/
  plugin.json                        — manifest (required)
  __init__.py                        — empty, marks as Python package
  capabilities.py                    — optional; override engine SQL-building behaviours
  readme.txt                         — this file; document your plugin here
  config/
    metadata/
      dimensions.json                — grouping columns (WHO/WHAT/WHERE)
      metrics.json                   — numeric columns to aggregate
      formulas.json                  — dynamic computed metrics (SQL expressions)
      tables.json                    — DB table list with channel and date config
      business_rules.json            — domain context, HAVING patterns, auto-enrichment rules
    prompts/
      question_parser.json           — Stage 1 LLM system prompt (Jinja2 template)
      nl_response.json               — Stage 4 LLM system prompt (Jinja2 template)
  docs/                              — optional; domain documentation
  tests/                             — optional; question test scripts


QUICK START
-----------
1. Copy this folder:
     cp -r plugins/empty-plugin plugins/my-plugin

2. Edit plugin.json:
     - Set "name" to match the folder name exactly
     - Update "description"

3. Fill in config/metadata/ files:
     - dimensions.json   — add your DB grouping columns
     - metrics.json      — add your DB numeric columns with aggregate_fn
     - tables.json       — add your DB table names, link to dimensions + metrics
     - business_rules.json — set domain_context; add rules if needed
     - formulas.json     — leave as { "formulas": [] } unless you need computed metrics

4. Tune config/prompts/ files:
     - question_parser.json  — adjust instructions for your domain
     - nl_response.json      — adjust output formatting for your domain

5. (Optional) Override engine capabilities in capabilities.py:
     Only needed if your DB requires non-standard SQL construction.
     See CAPABILITY OVERRIDES section below for details.

6. Run it:
     cd /path/to/prismage-data-chat-engine
     DATABASE_URL=postgresql://... python -c "
     from api.chatbot import build_plugin_engine
     engine = build_plugin_engine('my-plugin')
     r = engine.answer('your test question here')
     print(r.summary)
     print(r.detail)
     "


KEY CONCEPTS
------------
Dimensions
  Columns used in GROUP BY. Each has a list of user-facing aliases so the
  parser LLM can map natural language to the right column. Set table_affinity
  to the tables that contain this column.

Metrics
  Numeric columns to aggregate. Use aggregate_fn = "SUM" for totals and
  "AVG" for ratios/percentages. Set table_affinity to the tables that
  contain this column.

Table affinity routing
  The engine intersects the table_affinity sets of all requested dimensions
  and metrics to pick the right tables automatically. A query only runs
  against tables where all requested columns exist.

Channels
  Set "channel" on each table entry (e.g. "primary", "secondary"). When the
  user does not specify a channel, all matching tables run. Use a business
  rule with set_channel action to restrict routing based on keywords.

Snapshot date mode
  Set "date_mode": "snapshot" on a table to automatically filter to the
  latest loaded date (WHERE time_key = SELECT MAX(time_key) FROM table).
  Without this, queries with no date filter aggregate across all dates.
  When a date range is given, default engine behaviour is BETWEEN. Override
  build_date_filter() in capabilities.py to use MAX within range instead
  (needed when your table stores periodic snapshots, not incremental rows).

Business rules
  Rules fire between Stage 1 (parse) and Stage 2 (build SQL). They can:
    ensure_metrics    — add columns that should always appear together
    set_sort          — override sort direction (e.g. for "lowest performing")
    set_channel       — restrict to primary or secondary tables
    set_output_hint   — signal the display layer (e.g. side_by_side_table)

Prompt templates (Jinja2)
  Variables available in question_parser.json:
    {{ domain_context }}   — from business_rules.json
    {{ dimensions_list }}  — rendered from dimensions.json
    {{ metrics_list }}     — rendered from metrics.json
    {{ formulas_list }}    — rendered from formulas.json
    {{ metric_hints }}     — from business_rules.json metric_hints
    {{ few_shot_examples }} — from engine/prompts/prompt_builder.py

  Variables available in nl_response.json:
    {{ enumeration }}      — tabular query results as text
    {{ domain_context }}   — from business_rules.json


CAPABILITY OVERRIDES
--------------------
capabilities.py lets a plugin change how the engine builds specific SQL
clauses without modifying the shared engine code.

How it works:
  1. Create capabilities.py in your plugin root (already present in this template).
  2. Uncomment and rename the example class, which extends EngineCapabilities.
  3. Override only the methods you need; all others use the engine defaults.
  4. The loader auto-detects capabilities.py and injects your class into the
     query builder at startup.

Currently overridable methods (defined in engine/capabilities/base.py):

  build_date_filter(table, date_col, date_range, table_meta) -> str
    When to override: your table stores periodic snapshots (one complete
    picture per load date). A BETWEEN filter would span multiple loads and
    inflate totals. Override to use MAX(date_col) within the range instead.

    Default:  WHERE date_col BETWEEN 'start' AND 'end'
    Override: WHERE date_col = (SELECT MAX(date_col) FROM table
                                WHERE date_col >= start AND date_col <= end)

  build_snapshot_filter(table, date_col, table_meta) -> str
    When to override: you need a different "latest row" strategy when no
    date range is given and the table has date_mode = "snapshot".

    Default:  WHERE date_col = (SELECT MAX(date_col) FROM table)

Adding new overridable behaviours (engine change required):
  1. Add a method with a sensible default to engine/capabilities/base.py
  2. Replace the hardcoded logic in the relevant engine file with
     self.capabilities.<method>(...)
  3. Override the new method in your plugin's capabilities.py


CHATRESPONSE FIELDS
-------------------
engine.answer(question) returns a ChatResponse with:
  success         bool    — True if a result was produced
  answer          str     — full LLM response
  summary         str     — QUICK SUMMARY line (extracted from answer)
  detail          str     — ANALYSIS section (extracted from answer)
  tabular         str     — programmatic row enumeration (all rows)
  sql_queries     list    — [{ "table": ..., "sql": ... }, ...]
  total_rows      int     — total rows across all result sets
  applied_rules   list    — names of business rules that fired
  used_fallback   bool    — True if LangChain SQL fallback was used
  error           str     — error message when success=False


REFERENCE PLUGIN
----------------
See plugins/haldiram-sales/ for a fully worked example with:
  - 13 dimensions across a 3-tier sales hierarchy
  - 19 metrics (absolute, cumulative, percentage)
  - 6 DB tables across 3 granularity levels × 2 channels
  - 7 business rules
  - 121-question test suite
  - Full documentation in docs/
