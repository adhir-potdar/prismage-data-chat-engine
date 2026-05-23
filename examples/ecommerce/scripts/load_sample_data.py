"""
load_sample_data.py — E-Commerce Example

Generates and loads realistic sample data into the 5 e-commerce tables.
Produces enough data to meaningfully answer the questions in demo.py.

Data generated:
  products     : 30 products across 5 categories
  customers    : 200 customers across 4 segments and 3 regions
  orders       : ~1200 orders over the past 12 months
  order_items  : ~3000 line items linked to orders
  invoices     : ~1000 invoices from the orders set

Usage:
    cd /path/to/prismage-data-chat-engine
    python examples/ecommerce/scripts/load_sample_data.py
    python examples/ecommerce/scripts/load_sample_data.py --truncate-first
    python examples/ecommerce/scripts/load_sample_data.py --verify-only
"""
from __future__ import annotations
import argparse
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

_SCRIPT_DIR  = Path(__file__).resolve().parent
_ENGINE_ROOT = _SCRIPT_DIR.parents[3]

load_dotenv(_ENGINE_ROOT / ".env")

random.seed(42)

# ── Reference data ─────────────────────────────────────────────────────────────
REGIONS    = ["North", "South", "East", "West", "Central"]
COUNTRIES  = {"North": "India", "South": "India", "East": "India",
              "West": "India", "Central": "India"}
CITIES     = {"North": ["Delhi", "Chandigarh", "Amritsar"],
              "South": ["Chennai", "Bangalore", "Hyderabad"],
              "East":  ["Kolkata", "Bhubaneswar", "Patna"],
              "West":  ["Mumbai", "Pune", "Ahmedabad"],
              "Central": ["Nagpur", "Indore", "Bhopal"]}
SEGMENTS   = ["Enterprise", "SMB", "Retail", "Online"]
CHANNELS   = ["Direct", "Partner", "Online"]
SALES_REPS = ["Ravi Kumar", "Priya Singh", "Amit Sharma", "Neha Patel",
              "Suresh Gupta", "Anjali Rao", "Vikram Nair", "Pooja Mehta"]
MANAGERS   = {"Ravi Kumar": "Anil Verma", "Priya Singh": "Anil Verma",
              "Amit Sharma": "Sunita Joshi", "Neha Patel": "Sunita Joshi",
              "Suresh Gupta": "Anil Verma", "Anjali Rao": "Sunita Joshi",
              "Vikram Nair": "Raj Kapoor", "Pooja Mehta": "Raj Kapoor"}
CATEGORIES = ["Electronics", "Clothing", "Home & Kitchen", "Sports", "Books"]
PRODUCTS   = {
    "Electronics":    ["Laptop Pro 15", "Wireless Earbuds", "Smart Watch", "Tablet 10",
                       "USB-C Hub", "Bluetooth Speaker"],
    "Clothing":       ["Cotton T-Shirt", "Formal Trousers", "Winter Jacket",
                       "Sports Shorts", "Casual Dress"],
    "Home & Kitchen": ["Coffee Maker", "Air Fryer", "Knife Set",
                       "Dinner Set", "Bed Sheets"],
    "Sports":         ["Yoga Mat", "Dumbbells 5kg", "Running Shoes",
                       "Cricket Bat", "Football"],
    "Books":          ["Python Programming", "Business Strategy", "Fiction Novel",
                       "Cook Book", "Travel Guide"],
}
UNIT_PRICES = {
    "Laptop Pro 15": 65000, "Wireless Earbuds": 3500, "Smart Watch": 8000,
    "Tablet 10": 25000, "USB-C Hub": 1500, "Bluetooth Speaker": 4500,
    "Cotton T-Shirt": 800, "Formal Trousers": 2500, "Winter Jacket": 5000,
    "Sports Shorts": 600, "Casual Dress": 1800,
    "Coffee Maker": 4000, "Air Fryer": 6500, "Knife Set": 1200,
    "Dinner Set": 3000, "Bed Sheets": 1500,
    "Yoga Mat": 900, "Dumbbells 5kg": 1200, "Running Shoes": 3500,
    "Cricket Bat": 2000, "Football": 800,
    "Python Programming": 600, "Business Strategy": 500,
    "Fiction Novel": 350, "Cook Book": 450, "Travel Guide": 400,
}


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


def _rand_date(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))


def load_sample_data(truncate_first: bool = False):
    conn = _get_conn()
    cur  = conn.cursor()

    if truncate_first:
        for tbl in ["order_items", "invoices", "orders", "customers", "products"]:
            cur.execute(f"TRUNCATE TABLE {tbl} CASCADE;")
        conn.commit()
        print("Truncated all tables.")

    today     = date.today()
    year_ago  = today.replace(year=today.year - 1)
    month_start = today.replace(day=1)

    # ── products ──────────────────────────────────────────────────────────────
    print("Loading products...")
    all_products = []
    for cat, prods in PRODUCTS.items():
        for p in prods:
            price = UNIT_PRICES[p]
            cost  = round(price * random.uniform(0.45, 0.65), 2)
            launch = _rand_date(date(2020, 1, 1), date(2023, 1, 1))
            cur.execute(
                "INSERT INTO products (created_at, product_category, product_name, unit_price) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;",
                (launch, cat, p, price)
            )
            all_products.append((cat, p, price, cost))
    conn.commit()
    print(f"  {len(all_products)} products")

    # ── customers ─────────────────────────────────────────────────────────────
    print("Loading customers...")
    all_customers = []
    for i in range(200):
        region  = random.choice(REGIONS)
        city    = random.choice(CITIES[region])
        segment = random.choice(SEGMENTS)
        name    = f"Customer {i+1:03d}"
        satisfaction = round(random.uniform(3.0, 5.0), 1)
        created = _rand_date(date(2021, 1, 1), year_ago)
        cur.execute(
            "INSERT INTO customers (created_at, country, city, customer_segment, "
            "customer_name, satisfaction_score, order_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING;",
            (created, COUNTRIES[region], city, segment, name, satisfaction, 0)
        )
        all_customers.append((region, city, segment, name))
    conn.commit()
    print(f"  {len(all_customers)} customers")

    # ── orders ────────────────────────────────────────────────────────────────
    print("Loading orders...")
    order_rows = []
    for i in range(1200):
        region, city, segment, cname = random.choice(all_customers)
        rep     = random.choice(SALES_REPS)
        manager = MANAGERS[rep]
        channel = random.choice(CHANNELS)
        odate   = _rand_date(year_ago, today)

        base_rev   = round(random.uniform(5000, 200000), 2)
        prev_rev   = round(base_rev * random.uniform(0.7, 1.3), 2)
        tgt_rev    = round(base_rev * random.uniform(0.9, 1.2), 2)
        discount   = round(base_rev * random.uniform(0, 0.15), 2)
        returns    = round(base_rev * random.uniform(0, 0.05), 2) if random.random() < 0.2 else 0
        qty        = random.randint(1, 50)
        delivery   = random.randint(1, 10)
        satisfaction = round(random.uniform(3.0, 5.0), 1)

        # MTD/YTD — approximate based on date
        is_mtd = odate >= month_start
        mtd_rev      = base_rev if is_mtd else 0
        prev_mtd_rev = round(prev_rev * 0.03, 2)
        ytd_rev      = round(base_rev * (odate.timetuple().tm_yday / 365), 2)
        prev_ytd_rev = round(ytd_rev * random.uniform(0.8, 1.2), 2)
        ytd_tgt      = round(tgt_rev * (odate.timetuple().tm_yday / 365), 2)
        qtd_rev      = round(base_rev * 0.25, 2)

        cur.execute(
            "INSERT INTO orders (order_date, region, country, city, customer_segment, "
            "customer_name, sales_rep, sales_manager, channel, revenue, quantity, "
            "order_id, target_revenue, prev_revenue, returns_value, discount, "
            "satisfaction_score, delivery_days, mtd_revenue, prev_mtd_revenue, "
            "ytd_revenue, prev_ytd_revenue, ytd_target, qtd_revenue) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT DO NOTHING;",
            (odate, region, COUNTRIES[region], city, segment, cname, rep, manager, channel,
             base_rev, qty, i+1, tgt_rev, prev_rev, returns, discount,
             satisfaction, delivery, mtd_rev, prev_mtd_rev, ytd_rev, prev_ytd_rev,
             ytd_tgt, qtd_rev)
        )
        order_rows.append((i+1, odate, region, rep, channel))
    conn.commit()
    print(f"  {len(order_rows)} orders")

    # ── order_items ───────────────────────────────────────────────────────────
    print("Loading order_items...")
    item_count = 0
    for order_id, odate, region, rep, channel in order_rows:
        n_items = random.randint(1, 4)
        for _ in range(n_items):
            cat, pname, unit_price, cost = random.choice(all_products)
            qty      = random.randint(1, 10)
            discount = round(unit_price * qty * random.uniform(0, 0.1), 2)
            revenue  = round(unit_price * qty - discount, 2)
            prev_rev = round(revenue * random.uniform(0.7, 1.3), 2)
            cur.execute(
                "INSERT INTO order_items (order_date, region, country, product_category, "
                "product_name, channel, revenue, quantity, cost, discount) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING;",
                (odate, region, COUNTRIES[region], cat, pname, channel,
                 revenue, qty, round(cost * qty, 2), discount)
            )
            item_count += 1
    conn.commit()
    print(f"  {item_count} order items")

    # ── invoices ──────────────────────────────────────────────────────────────
    print("Loading invoices...")
    inv_count = 0
    for order_id, odate, region, rep, channel in random.sample(order_rows, 1000):
        inv_date = odate + timedelta(days=random.randint(1, 7))
        region2, city, segment, cname = random.choice(all_customers)
        manager = MANAGERS[rep]
        revenue  = round(random.uniform(5000, 200000), 2)
        tgt_rev  = round(revenue * random.uniform(0.9, 1.2), 2)
        returns  = round(revenue * random.uniform(0, 0.05), 2) if random.random() < 0.15 else 0
        mtd_rev  = revenue if inv_date >= month_start else 0
        prev_mtd = round(revenue * 0.03, 2)
        ytd_rev  = round(revenue * (inv_date.timetuple().tm_yday / 365), 2)
        ytd_tgt  = round(tgt_rev * (inv_date.timetuple().tm_yday / 365), 2)
        cur.execute(
            "INSERT INTO invoices (invoice_date, region, country, "
            "customer_name, sales_rep, sales_manager, revenue, target_revenue, prev_revenue) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING;",
            (inv_date, region, COUNTRIES[region], cname, rep, manager,
             revenue, tgt_rev, round(revenue * random.uniform(0.7, 1.3), 2))
        )
        inv_count += 1
    conn.commit()
    print(f"  {inv_count} invoices")

    print("\nSample data loaded successfully.")


def verify():
    conn = _get_conn()
    cur  = conn.cursor()
    print("\nRow counts:")
    for tbl in ["products", "customers", "orders", "order_items", "invoices"]:
        cur.execute(f"SELECT COUNT(*) FROM {tbl};")
        print(f"  {tbl}: {cur.fetchone()[0]}")
    cur.close()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load sample data into e-commerce tables")
    parser.add_argument("--truncate-first", action="store_true", help="Truncate tables before loading")
    parser.add_argument("--verify-only",    action="store_true", help="Show row counts only")
    args = parser.parse_args()

    if args.verify_only:
        verify()
    else:
        load_sample_data(truncate_first=args.truncate_first)
        verify()
