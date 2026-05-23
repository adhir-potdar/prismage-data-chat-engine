"""
create_tables.py — E-Commerce Example

Creates the 5 PostgreSQL tables required by the generic e-commerce example config.
Table definitions are driven directly from examples/ecommerce/config/metadata/ so
this script stays in sync automatically when config changes.

Tables created:
  orders       — one row per order (revenue, quantity, MTD/YTD metrics)
  order_items  — one row per line item (product-level revenue, cost, quantity)
  invoices     — invoiced orders with settlement metrics
  customers    — customer master (segment, location, satisfaction)
  products     — product master (category, price, cost)

Usage:
    cd /path/to/prismage-data-chat-engine
    python examples/ecommerce/scripts/create_tables.py
    python examples/ecommerce/scripts/create_tables.py --drop-existing
    python examples/ecommerce/scripts/create_tables.py --verify-only
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_EXAMPLE_DIR  = _SCRIPT_DIR.parent
_ENGINE_ROOT  = _EXAMPLE_DIR.parents[1]
_CONFIG_DIR   = _EXAMPLE_DIR / "config" / "metadata"

load_dotenv(_ENGINE_ROOT / ".env")

# Column type mapping: metric db_column → SQL type
# Metrics with integer semantics use INTEGER, rest NUMERIC
_INT_COLS = {"orders_count", "quantity", "delivery_days"}


def _get_conn():
    url = os.getenv("DATABASE_URL")
    if url:
        return psycopg2.connect(url)
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        database=os.getenv("POSTGRES_DB", "vectordb"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )


def _load_config():
    with open(_CONFIG_DIR / "tables.json")  as f: tables  = json.load(f)["tables"]
    with open(_CONFIG_DIR / "metrics.json") as f: metrics = json.load(f)["metrics"]
    # Only metrics with a db_column are stored columns
    stored = {m["name"]: m["db_column"] for m in metrics if m.get("db_column")}
    return tables, stored


def _create_sql(table: dict, stored: dict[str, str]) -> str:
    name     = table["name"]
    dims     = table["dimensions"]
    date_col = table.get("date_column") or "created_at"
    # Metrics used by this table
    table_metrics = [m for m in table.get("metrics", []) if m in stored]

    cols = [f"{date_col} DATE NOT NULL"]
    cols += [f"{d} TEXT" for d in dims]
    for m_name in table_metrics:
        col = stored[m_name]
        sql_type = "INTEGER" if col in _INT_COLS else "NUMERIC(15,2)"
        cols.append(f"{col} {sql_type}")

    return (
        f"CREATE TABLE IF NOT EXISTS {name} (\n"
        f"    id SERIAL PRIMARY KEY,\n"
        f"    " + ",\n    ".join(cols) + "\n);"
    )


def _index_sqls(table: dict) -> list[str]:
    name     = table["name"]
    date_col = table.get("date_column") or "created_at"
    dims     = table["dimensions"]
    idxs = [f"CREATE INDEX IF NOT EXISTS idx_{name}_{date_col} ON {name}({date_col});"]
    for d in dims[:3]:   # index first 3 dimensions
        idxs.append(f"CREATE INDEX IF NOT EXISTS idx_{name}_{d} ON {name}({d});")
    return idxs


def create_tables(drop_existing: bool = False):
    tables, stored = _load_config()
    conn = _get_conn()
    cur  = conn.cursor()

    print(f"\nE-Commerce Example — create_tables.py")
    print(f"Database : {os.getenv('DATABASE_URL', os.getenv('POSTGRES_DB', 'vectordb'))}")
    print(f"Tables   : {len(tables)}\n")

    try:
        for t in tables:
            name = t["name"]
            print(f"  {name}")

            if drop_existing:
                cur.execute(f"DROP TABLE IF EXISTS {name} CASCADE;")
                conn.commit()
                print(f"    dropped")

            cur.execute(_create_sql(t, stored))
            conn.commit()
            print(f"    created")

            for idx in _index_sqls(t):
                cur.execute(idx)
            conn.commit()
            print(f"    {len(_index_sqls(t))} indexes created")

        print(f"\nAll {len(tables)} tables created successfully.")
    except Exception as e:
        conn.rollback()
        print(f"\nERROR: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def verify_tables():
    tables, _ = _load_config()
    conn = _get_conn()
    cur  = conn.cursor()
    print(f"\nVerifying tables:")
    try:
        for t in tables:
            name = t["name"]
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=%s;", (name,)
            )
            col_count = cur.fetchone()[0]
            if col_count:
                cur.execute(f"SELECT COUNT(*) FROM {name};")
                row_count = cur.fetchone()[0]
                print(f"  OK  {name}  ({col_count} cols, {row_count} rows)")
            else:
                print(f"  MISSING  {name}")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create e-commerce example tables")
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--verify-only",   action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        verify_tables()
    else:
        create_tables(drop_existing=args.drop_existing)
        verify_tables()
