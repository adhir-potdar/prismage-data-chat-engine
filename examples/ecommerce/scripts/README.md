# E-Commerce Example — Initial Setup

Run these steps once to create tables and load sample data before starting
the chatbot engine with the generic e-commerce config.
All commands are run from the **repository root** (`prismage-data-chat-engine/`).

---

## Prerequisites

1. **Python environment** — create and activate the venv:
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Environment variables** — copy the template and fill in your values:
   ```bash
   cp .env.example .env
   # Edit .env — set DATABASE_URL (or POSTGRES_* vars) and OPENAI_API_KEY
   ```

3. **PostgreSQL** — the target database must exist and be reachable.

---

## Step 1 — Create database tables

Creates the 5 e-commerce tables driven by the config in
`examples/ecommerce/config/metadata/`:

```bash
python examples/ecommerce/scripts/create_tables.py
```

To drop and recreate from scratch:

```bash
python examples/ecommerce/scripts/create_tables.py --drop-existing
```

To verify tables exist without creating:

```bash
python examples/ecommerce/scripts/create_tables.py --verify-only
```

**Tables created:**
- `orders` — one row per order (revenue, quantity, MTD/YTD metrics)
- `order_items` — one row per line item (product-level revenue, cost, quantity)
- `invoices` — invoiced orders with settlement metrics
- `customers` — customer master (segment, location, satisfaction)
- `products` — product master (category, price, cost)

---

## Step 2 — Load sample data

Generates and loads realistic synthetic data (reproducible via `random.seed(42)`):

```bash
python examples/ecommerce/scripts/load_sample_data.py
```

To truncate existing data before loading:

```bash
python examples/ecommerce/scripts/load_sample_data.py --truncate-first
```

To verify row counts without loading:

```bash
python examples/ecommerce/scripts/load_sample_data.py --verify-only
```

**Data generated:**
| Table | Rows |
|---|---|
| products | 30 (5 categories × 5–6 products) |
| customers | 200 (4 segments, 5 regions, 15 cities) |
| orders | ~1 200 (past 12 months) |
| order_items | ~3 000 (1–4 items per order) |
| invoices | ~1 000 (sampled from orders) |

---

## Step 3 — Start the chatbot

The generic e-commerce config is loaded by default (no `--plugin` flag):

```bash
python -m api.chatbot
```

You will see:
```
Prismage Data Chat Engine [generic] — type 'exit' to quit.

You:
```

---

## Sample questions

```
What is total revenue this month?
Which product category has the highest revenue?
Show me top 5 customers by order value.
What is the revenue split by region?
Which sales rep has the most orders?
What is the average satisfaction score by segment?
```

---

## Verify row counts

At any time:

```bash
python examples/ecommerce/scripts/load_sample_data.py --verify-only
```
