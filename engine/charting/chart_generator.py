"""
LLM-based Vega-Lite v5 chart spec generator.

Flow:
  1. Resolve vega_types per column — from Trino type strings via vega_types.yaml,
     falling back to value-based inference for columns with no type hint.
  2. Render tabular data as markdown table for LLM context.
  3. Call LLM in JSON mode: prompt + column types + table → spec skeleton
     (LLM must NOT include $schema or data — both are injected programmatically).
  4. Parse + validate response via Pydantic model_validate_json() in one step —
     no separate json.loads; partial or invalid JSON raises ValidationError cleanly.
  5. Inject $schema and data.values into validated skeleton → complete Vega-Lite v5 spec.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError

logger = logging.getLogger(__name__)

# Path to the vega_types.yaml file in the same directory
_VEGA_TYPES_YAML = Path(__file__).parent / "vega_types.yaml"

VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"

VALID_MARKS = {
    "arc", "area", "bar", "boxplot", "circle", "errorband", "errorbar",
    "geoshape", "image", "line", "point", "rect", "rule", "square",
    "text", "tick", "trail",
}

DEFAULT_PROMPT = """\
You are a data visualization expert. Given the user's question and the tabular \
output below, produce a Vega-Lite v5 spec skeleton that best communicates the \
answer visually.

Requirements:
- Output ONLY a valid JSON object — no markdown, no explanation, no code fences.
- Do NOT include a "$schema" or "data" field; both will be injected separately.
- Always include "mark" and "encoding" fields.
- Assign correct Vega-Lite types using the vega_types hint provided: \
quantitative, ordinal, nominal, temporal.
- Include a descriptive "title" derived from the question.
- Keep it simple — one mark layer, no transforms unless essential.
- Always place the legend at the bottom: add "legend": {{"orient": "bottom"}} on any color/size/shape encoding.

Chart selection rules (use vega_types to decide):
- line: x-axis is temporal (time series, trend over time).
- arc (pie/donut): ≤6 nominal categories AND question asks for share/proportion/breakdown. \
Use theta for the measure, color for the category.
- bar horizontal (mark {{"type": "bar", "orient": "horizontal"}}, x=quantitative, y=nominal): category names are long OR >6 categories.
- bar vertical (mark {{"type": "bar", "orient": "vertical"}}, x=nominal, y=quantitative): ≤6 short category names.
- point/scatter: both x and y are quantitative (correlation or distribution).
- grouped bar: two nominal dimensions + one quantitative — put the nominal with more distinct \
values on y, the other on color, add yOffset=color field for side-by-side bars.
- period comparison (when both period1_value and period2_value columns are present): use a fold \
transform to show both side by side. Example:
  "transform": [{{"fold": ["period1_value", "period2_value"], "as": ["period", "value"]}}],
  "mark": {{"type": "bar", "orient": "horizontal"}},
  "encoding": {{
    "y": {{"field": "<dimension>", "type": "nominal"}},
    "x": {{"field": "value", "type": "quantitative"}},
    "yOffset": {{"field": "period", "type": "nominal"}},
    "color": {{"field": "period", "type": "nominal"}}
  }}
- multiple nominal dimensions: when there are 2+ nominal columns, pick the one that varies \
most within groups of the others for color+yOffset (side-by-side), and the one with the most \
distinct values for y. All nominals must be visible — never drop one silently.
Default to horizontal bar when none of the above clearly applies.

User question: {question}

Column vega_types:
{vega_types}

Tabular data:
{table}

Output the Vega-Lite v5 spec skeleton as a valid JSON object:"""


# ── Pydantic model ────────────────────────────────────────────────────────────

class VegaLiteSkeleton(BaseModel):
    """
    Pydantic model for the LLM-generated Vega-Lite spec skeleton.

    The LLM is instructed to omit $schema and data; both are injected
    programmatically via to_full_spec(). Extra fields (width, height,
    layer, transform, etc.) are allowed and passed through unchanged.
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    title: Optional[str] = None
    mark: Union[str, Dict[str, Any]]
    encoding: Dict[str, Any]

    @field_validator("mark")
    @classmethod
    def validate_mark(cls, v: Any) -> Any:
        mark_type = v if isinstance(v, str) else v.get("type", "")
        if mark_type not in VALID_MARKS:
            raise ValueError(
                f"Invalid mark type {mark_type!r}. Must be one of: {sorted(VALID_MARKS)}"
            )
        return v

    @field_validator("encoding")
    @classmethod
    def validate_encoding(cls, v: Any) -> Any:
        if not isinstance(v, dict) or not v:
            raise ValueError("encoding must be a non-empty dict")
        return v

    def to_full_spec(self, data_values: List[Dict], schema: str = VEGA_LITE_SCHEMA) -> Dict:
        """Inject $schema and data.values to produce a complete Vega-Lite v5 spec."""
        spec = self.model_dump(exclude_none=True)
        spec["$schema"] = schema
        spec["data"] = {"values": data_values}
        return spec


# ── Type map loading ──────────────────────────────────────────────────────────

_TYPE_MAP_CACHE: Optional[Dict[str, str]] = None


def load_vega_type_map(yaml_path: str | Path = _VEGA_TYPES_YAML) -> Dict[str, str]:
    """
    Load the Trino-type → Vega-Lite-type mapping from a YAML file.
    Cached after first load. Returns {} on any error.
    """
    global _TYPE_MAP_CACHE
    if _TYPE_MAP_CACHE is not None:
        return _TYPE_MAP_CACHE

    try:
        import yaml
        with open(yaml_path) as fh:
            raw = yaml.safe_load(fh) or {}
        _TYPE_MAP_CACHE = {str(k).lower(): str(v).lower() for k, v in raw.items()}
    except Exception as exc:
        logger.warning("Could not load vega_types.yaml: %s — defaulting all types to 'nominal'", exc)
        _TYPE_MAP_CACHE = {}

    return _TYPE_MAP_CACHE


def resolve_vega_type(col_type_str: str, type_map: Optional[Dict[str, str]] = None) -> str:
    """
    Map a Trino column type string to a Vega-Lite type using prefix matching.

    Examples:
        "decimal(10,2)"  →  "quantitative"
        "varchar(255)"   →  "nominal"
        "timestamp"      →  "temporal"
        "unknown_type"   →  "nominal"  (default)
    """
    if type_map is None:
        type_map = load_vega_type_map()

    key = col_type_str.lower().strip()
    if key in type_map:
        return type_map[key]
    for prefix, vega_type in type_map.items():
        if key.startswith(prefix):
            return vega_type
    return "nominal"


def infer_vega_type_from_values(values: List[Any]) -> str:
    """
    Fallback: infer Vega-Lite type by inspecting actual column values.
    Used when no Trino type information is available (e.g. embedding engine).
    """
    import re

    non_null = [v for v in values if v is not None and v != ""]
    if not non_null:
        return "nominal"

    _DATE_RE = re.compile(
        r"^\d{4}-\d{2}-\d{2}"
        r"|^[A-Za-z]{3}\s+\d{1,2},?\s+\d{4}"
        r"|^\d{1,2}/\d{1,2}/\d{4}$"
    )
    if all(isinstance(v, str) and _DATE_RE.match(str(v)) for v in non_null):
        return "temporal"

    if all(isinstance(v, (int, float)) for v in non_null):
        return "quantitative"

    try:
        for v in non_null:
            float(str(v).replace(",", "").replace("₹", "").replace("%", "").strip())
        return "quantitative"
    except (ValueError, TypeError):
        pass

    return "nominal"


# ── Table formatting ──────────────────────────────────────────────────────────

def _build_table_text(columns: List[str], rows: List[Dict], max_rows: int = 50) -> str:
    """Format rows as a pipe-delimited markdown table for the LLM context."""
    if not rows or not columns:
        return "(no data)"

    display = rows[:max_rows]
    header = " | ".join(columns)
    sep = " | ".join("-" * max(len(c), 4) for c in columns)
    data_lines = [
        " | ".join(str(row.get(c, "")) for c in columns)
        for row in display
    ]
    suffix = [f"(... {len(rows) - max_rows} more rows not shown)"] if len(rows) > max_rows else []
    return "\n".join([header, sep] + data_lines + suffix)


# ── Data cleaning for injection ───────────────────────────────────────────────

def _clean_value(value: Any, vega_type: str) -> Any:
    """Strip currency/percent symbols from quantitative values before injection."""
    # Decimal (from SQLAlchemy/Trino) is not JSON-serializable — convert eagerly
    try:
        from decimal import Decimal
        if isinstance(value, Decimal):
            return float(value)
    except ImportError:
        pass
    if vega_type != "quantitative":
        return value
    # Replace None/null with 0 so all rows render (Vega-Lite skips null quantitative values)
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = value.replace("₹", "").replace(",", "").replace("%", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            pass
    return value


def _build_data_values(
    columns: List[str],
    rows: List[Dict],
    vega_types: Dict[str, str],
) -> List[Dict]:
    """Return cleaned row dicts suitable for Vega-Lite data.values injection."""
    return [
        {col: _clean_value(row.get(col), vega_types.get(col, "nominal")) for col in columns}
        for row in rows
    ]


# ── Period fold helpers ───────────────────────────────────────────────────────

_PERIOD_COLS = {"period1_value", "period2_value"}
_PERIOD_DATE_MAP = {"period1_value": "period1_date", "period2_value": "period2_date"}
_FOLDED_COL_TYPES: Dict[str, str] = {"period": "varchar", "value": "decimal"}


def _fold_period_rows(
    columns: List[str], rows: List[Dict]
) -> tuple[List[str], List[Dict]]:
    """
    Convert wide-format period rows into long format for grouped bar charting.

    {"dimension": "X", "period1_value": 14.51, "period2_value": 36.3,
     "period1_date": "Dec 31, 2025", "period2_date": "Jan 01, 2026"}
    becomes two rows:
    {"dimension": "X", "period": "Dec 31, 2025", "value": 14.51}
    {"dimension": "X", "period": "Jan 01, 2026", "value": 36.3}
    """
    skip = {"period1_value", "period2_value", "period1_date", "period2_date"}
    base_cols = [c for c in columns if c not in skip]
    folded_rows: List[Dict] = []
    for row in rows:
        for val_col, date_col in _PERIOD_DATE_MAP.items():
            new_row = {c: row.get(c) for c in base_cols}
            new_row["period"] = row.get(date_col) or val_col
            new_row["value"] = row.get(val_col)
            folded_rows.append(new_row)
    return base_cols + ["period", "value"], folded_rows


# ── Val/Vol fold helpers ──────────────────────────────────────────────────────

def _find_val_vol_pairs(columns: List[str]) -> List[tuple]:
    """
    Return list of (val_col, vol_col, base_name) for columns that have both
    a _val and a _vol counterpart, e.g. cymtd_val + cymtd_vol → ("cymtd_val", "cymtd_vol", "cymtd").
    """
    col_set = set(columns)
    pairs = []
    seen = set()
    for col in columns:
        if col.endswith("_val"):
            base = col[:-4]
            vol_col = base + "_vol"
            if vol_col in col_set and base not in seen:
                pairs.append((col, vol_col, base))
                seen.add(base)
    return pairs


def _fold_val_vol_rows(
    columns: List[str], rows: List[Dict], pairs: List[tuple]
) -> tuple[List[str], List[Dict]]:
    """
    Fold _val/_vol column pairs into long format for grouped bar charting.

    {"asm": "X", "cymtd_val": 2.5, "cymtd_vol": 150.0}
    becomes two rows:
    {"asm": "X", "metric": "cymtd (value)", "amount": 2.5}
    {"asm": "X", "metric": "cymtd (volume)", "amount": 150.0}
    """
    skip = {col for val_col, vol_col, _ in pairs for col in (val_col, vol_col)}
    base_cols = [c for c in columns if c not in skip]
    folded_rows: List[Dict] = []
    for row in rows:
        for val_col, vol_col, base in pairs:
            for col in (val_col, vol_col):
                new_row = {c: row.get(c) for c in base_cols}
                new_row["metric"] = col          # e.g. "cymtd_val", "cymtd_vol"
                new_row["amount"] = row.get(col)
                folded_rows.append(new_row)
    return base_cols + ["metric", "amount"], folded_rows


# ── Tooltip injection ─────────────────────────────────────────────────────────

def _add_tooltips(
    skeleton: "VegaLiteSkeleton", columns: List[str], vega_types: Dict[str, str]
) -> None:
    """
    Add a tooltip array for all columns if the spec has no tooltip encoding.
    Mutates skeleton in place.
    """
    enc = skeleton.encoding
    if "tooltip" not in enc:
        enc["tooltip"] = [
            {"field": col, "type": vega_types.get(col, "nominal")}
            for col in columns
        ]


# ── Generic bar encoding fix ──────────────────────────────────────────────────

def _fix_bar_encoding(
    skeleton: "VegaLiteSkeleton",
    rows: List[Dict],
    columns: List[str],
    vega_types: Dict[str, str],
) -> None:
    """
    For bar charts with 2 or more nominal dimensions, assign y/color/yOffset
    generically — no field names are hardcoded.

    Algorithm (purely data-driven):
      1. For each nominal field, compute the average number of distinct values it
         takes *within groups* defined by all other nominal fields.  The field
         with the highest within-group variation is the "inner" dimension — it
         changes most rapidly within every outer group, so it should drive
         color + yOffset (the side-by-side legend dimension).
      2. Among the remaining ("outer") nominal fields, the one with the most
         total distinct values becomes the y-axis (one bar-group per value).
      3. Any further outer nominal fields appear only in tooltip (handled later
         by _add_tooltips).
      4. The quantitative field with the most distinct values becomes x.

    Mutates skeleton in place.
    """
    mark = skeleton.mark
    mark_type = mark if isinstance(mark, str) else mark.get("type", "")
    if mark_type != "bar":
        return

    nominal_fields = [col for col in columns if vega_types.get(col) == "nominal"]
    if len(nominal_fields) < 2:
        return  # nothing to rearrange

    # Maximum distinct values a color field can have and still produce a readable legend.
    # Beyond this, the field is too granular for color and should go on the y-axis instead.
    LEGEND_THRESHOLD = 10

    def within_group_avg_distinct(field: str) -> float:
        """Average distinct values of `field` per group of all other nominals."""
        others = [f for f in nominal_fields if f != field]
        groups: Dict[tuple, set] = {}
        for row in rows:
            key = tuple(row.get(f) for f in others)
            groups.setdefault(key, set()).add(row.get(field))
        return sum(len(v) for v in groups.values()) / max(len(groups), 1)

    def total_distinct(field: str) -> int:
        return len({row.get(field) for row in rows})

    # Step 1: find the "inner" dimension — the one that varies most within
    # groups formed by the other nominal fields.
    inner_field = max(nominal_fields, key=within_group_avg_distinct)
    inner_distinct = total_distinct(inner_field)
    outer_fields = [f for f in nominal_fields if f != inner_field]

    if inner_distinct <= LEGEND_THRESHOLD:
        # Normal case: inner field is compact enough for a color legend.
        #   y      = outer field with most distinct values (granular rows)
        #   color  = inner field
        y_field = max(outer_fields, key=total_distinct)
        color_field = inner_field
    else:
        # Inner field has too many distinct values for a readable legend
        # (e.g. 40+ towns).  Put it on the y-axis instead and find the best
        # color candidate among the outer fields — the one with the most
        # distinct values that still fits within the legend threshold.
        y_field = inner_field
        legend_candidates = [f for f in outer_fields if total_distinct(f) <= LEGEND_THRESHOLD]
        if legend_candidates:
            # Prefer the most informative field that still fits in a legend
            color_field = max(legend_candidates, key=total_distinct)
        else:
            # All remaining fields are also too large — just use the smallest
            color_field = min(outer_fields, key=total_distinct)

    enc = skeleton.encoding
    enc["y"] = {"field": y_field, "type": "nominal"}
    enc["color"] = {"field": color_field, "type": "nominal", "legend": {"orient": "bottom"}}
    enc["yOffset"] = {"field": color_field, "type": "nominal"}

    # Ensure x encodes the quantitative measure
    quant_fields = [col for col in columns if vega_types.get(col) == "quantitative"]
    if quant_fields:
        best_quant = max(quant_fields, key=total_distinct)
        enc["x"] = {"field": best_quant, "type": "quantitative"}

    # Force horizontal orientation so y=nominal / x=quantitative is valid
    skeleton.mark = {"type": "bar", "orient": "horizontal"}

    logger.debug(
        "Bar encoding (generic): y=%s (distinct=%d), color/yOffset=%s (distinct=%d), x=%s",
        y_field, total_distinct(y_field),
        color_field, total_distinct(color_field),
        quant_fields[0] if quant_fields else "?",
    )


# ── Bar orientation fix ───────────────────────────────────────────────────────

def _flip_to_horizontal_if_needed(skeleton: "VegaLiteSkeleton", rows: List[Dict]) -> None:
    """
    If the LLM chose a vertical bar chart (x=nominal, y=quantitative) and any
    label in the nominal field exceeds 10 characters, flip to horizontal by
    swapping x/y and setting mark to {"type": "bar", "orient": "horizontal"}.

    Already-horizontal bars are left untouched. Mutates skeleton in place.
    """
    # Resolve mark type regardless of whether mark is a string or a dict
    mark = skeleton.mark
    mark_type = mark if isinstance(mark, str) else mark.get("type", "")
    if mark_type != "bar":
        return

    enc = skeleton.encoding
    x_enc = enc.get("x", {})
    y_enc = enc.get("y", {})

    # Only applies to vertical bars (x=nominal, y=quantitative)
    if x_enc.get("type") == "nominal" and y_enc.get("type") == "quantitative":
        nominal_field = x_enc.get("field")
        if nominal_field:
            max_label_len = max(
                (len(str(row.get(nominal_field, ""))) for row in rows),
                default=0,
            )
            if max_label_len > 10:
                # Swap axes and set mark to dict with horizontal orient
                enc["x"], enc["y"] = enc["y"], enc["x"]
                skeleton.mark = {"type": "bar", "orient": "horizontal"}

    # Add color encoding if absent — use whichever axis is nominal after any flip
    if "color" not in enc:
        nominal_enc = enc.get("y", {}) if enc.get("y", {}).get("type") == "nominal" else enc.get("x", {})
        nominal_field = nominal_enc.get("field")
        if nominal_field:
            enc["color"] = {"field": nominal_field, "type": "nominal", "legend": {"orient": "bottom"}}

    # Enforce legend at bottom for all encoding channels that have a legend
    for channel in ("color", "size", "shape", "opacity"):
        ch_enc = enc.get(channel)
        if isinstance(ch_enc, dict):
            legend = ch_enc.get("legend")
            if legend is None:
                ch_enc["legend"] = {"orient": "bottom"}
            elif isinstance(legend, dict):
                legend["orient"] = "bottom"


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_vega_spec(
    question: str,
    columns: List[str],
    rows: List[Dict],
    generate_fn: Callable[[str], str],
    col_types: Optional[Dict[str, str]] = None,
    prompt_template: Optional[str] = None,
    schema: str = VEGA_LITE_SCHEMA,
) -> Optional[Dict]:
    """
    Generate a complete Vega-Lite v5 spec dict with $schema and data injected.

    Args:
        question:        The user's original question.
        columns:         Column names in display order.
        rows:            List of row dicts (values may have ₹/% symbols).
        generate_fn:     LLM callable — takes the full prompt string, returns JSON string.
                         The caller should bind the LLM to JSON output mode so the
                         response is always a valid JSON object without markdown fences.
        col_types:       Optional dict mapping column name → Trino type string
                         (e.g. {"revenue": "decimal(10,2)", "date": "date"}).
                         When omitted, Vega types are inferred from row values.
        prompt_template: Override the default prompt. Must contain {question},
                         {vega_types}, and {table} placeholders.

    Returns:
        Complete Vega-Lite v5 spec dict (with $schema and data.values injected),
        or None on LLM failure or Pydantic validation failure.
    """
    if not columns or not rows:
        return None

    type_map = load_vega_type_map()
    template = prompt_template or DEFAULT_PROMPT

    # 1. Resolve vega_types per column
    vega_types: Dict[str, str] = {}
    for col in columns:
        if col_types and col in col_types:
            vega_types[col] = resolve_vega_type(col_types[col], type_map)
        else:
            vega_types[col] = infer_vega_type_from_values([row.get(col) for row in rows])

    vega_types_text = "\n".join(f"  {col}: {vt}" for col, vt in vega_types.items())

    # 2. Build markdown table for LLM context
    table_text = _build_table_text(columns, rows)

    # 3. Build prompt
    prompt = template.format(
        question=question,
        vega_types=vega_types_text,
        table=table_text,
    )

    # 4. Call LLM
    try:
        response_text = generate_fn(prompt)
    except Exception as exc:
        logger.error("Chart LLM call failed: %s", exc)
        return None

    if not response_text or not response_text.strip():
        return None

    # 5. Parse + validate with Pydantic in one step.
    #    model_validate_json() handles both JSON parsing and schema validation —
    #    partial or malformed LLM output raises ValidationError, not json.JSONDecodeError.
    try:
        skeleton = VegaLiteSkeleton.model_validate_json(response_text.strip())
    except ValidationError as exc:
        logger.warning("Chart spec validation failed:\n%s", exc)
        return None

    # 5a. Period comparison fallback (Option 1): if both period columns exist but
    #     the LLM did not emit a fold transform, pre-fold and re-run with simpler data.
    has_period_cols = _PERIOD_COLS.issubset(set(columns))
    if has_period_cols:
        # Use model_dump() — always returns a dict including extra fields like transform.
        # model_extra can be None in some Pydantic v2 versions when no extras are present.
        spec_dict = skeleton.model_dump()
        transforms = spec_dict.get("transform") or []
        has_fold = any(
            isinstance(t, dict) and "fold" in t for t in transforms
        )
        if not has_fold:
            logger.info("Chart: LLM skipped fold transform — falling back to pre-folded rows.")
            folded_columns, folded_rows = _fold_period_rows(columns, rows)
            folded_col_types = {**(_FOLDED_COL_TYPES), **(col_types or {})}
            return generate_vega_spec(
                question=question,
                columns=folded_columns,
                rows=folded_rows,
                generate_fn=generate_fn,
                col_types=folded_col_types,
                prompt_template=prompt_template,
                schema=schema,
            )

    # 5b. Val/Vol fold fallback: if _val/_vol pairs exist but the LLM only encoded
    #     one of them, fold into metric/amount long format and re-run.
    val_vol_pairs = _find_val_vol_pairs(columns)
    if val_vol_pairs:
        all_encoding_fields = {
            v.get("field") for v in skeleton.encoding.values()
            if isinstance(v, dict) and v.get("field")
        }
        val_cols = {p[0] for p in val_vol_pairs}
        vol_cols = {p[1] for p in val_vol_pairs}
        uses_val = bool(val_cols & all_encoding_fields)
        uses_vol = bool(vol_cols & all_encoding_fields)
        if not (uses_val and uses_vol):
            logger.info("Chart: LLM used only one of _val/_vol — folding to long format.")
            folded_columns, folded_rows = _fold_val_vol_rows(columns, rows, val_vol_pairs)
            folded_col_types = {"metric": "varchar", "amount": "decimal", **(col_types or {})}
            # Build a hint for the remaining nominal dimensions so the LLM uses them all
            remaining_nominals = [c for c in folded_columns if c not in ("metric", "amount")]
            nom_hint = (
                f"\nIMPORTANT: The nominal dimension columns are: {remaining_nominals}. "
                "Use ALL of them in the chart. Put the one with the most distinct values on y, "
                "the next on color. Do NOT leave any nominal column unencoded."
            ) if len(remaining_nominals) > 1 else ""
            augmented_question = question + nom_hint
            return generate_vega_spec(
                question=augmented_question,
                columns=folded_columns,
                rows=folded_rows,
                generate_fn=generate_fn,
                col_types=folded_col_types,
                prompt_template=prompt_template,
                schema=schema,
            )

    # 5c. Flip vertical bar to horizontal if any label exceeds 10 characters.
    _flip_to_horizontal_if_needed(skeleton, rows)

    # 5d. Generic bar encoding: assign y/color/yOffset by data-driven heuristic
    #     (inner vs outer dimensions). Applies whenever 2+ nominal fields exist.
    _fix_bar_encoding(skeleton, rows, columns, vega_types)

    # 5e. Add tooltip for all columns if not already present.
    _add_tooltips(skeleton, columns, vega_types)

    # 6. Inject $schema and data.values
    data_values = _build_data_values(columns, rows, vega_types)
    return skeleton.to_full_spec(data_values, schema=schema)
