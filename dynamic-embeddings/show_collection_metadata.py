#!/usr/bin/env python3
"""
Show Collection Metadata Table Contents

This script displays the contents of the collection metadata table for a specified namespace.

Usage:
    cd dynamic-embeddings
    python show_collection_metadata.py --namespace revenue_mgmt
    python show_collection_metadata.py --namespace revenue_mgmt --dimension device
    python show_collection_metadata.py --namespace revenue_mgmt --granularity wow
    python show_collection_metadata.py --namespace revenue_mgmt --detailed
"""

import sys
import argparse
import psycopg
from pathlib import Path


def load_db_config():
    """Load database configuration from .env file."""
    env_file = Path('.env')
    if not env_file.exists():
        print(f"❌ .env file not found: {env_file}")
        print(f"   Expected location: {Path('.env').resolve()}")
        sys.exit(1)

    db_config = {}
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                db_config[key] = value.strip()

    return db_config


def connect_to_database(db_config):
    """Connect to PostgreSQL database."""
    try:
        print(f"\n🔌 Connecting to PostgreSQL database...")
        conn = psycopg.connect(
            host=db_config.get('POSTGRES_HOST', 'localhost'),
            port=int(db_config.get('POSTGRES_PORT', 5432)),
            dbname=db_config.get('POSTGRES_DB', 'vectordb'),
            user=db_config.get('POSTGRES_USER', 'postgres'),
            password=db_config.get('POSTGRES_PASSWORD', '')
        )
        print(f"✅ Connected successfully")
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)


def show_summary(cursor, table_name, dimension_filter=None, granularity_filter=None):
    """Display summary statistics of the metadata table."""
    # Total count
    where_clause = ""
    params = []

    if dimension_filter:
        where_clause += " WHERE dimension = %s"
        params.append(dimension_filter)

    if granularity_filter:
        where_clause += " AND " if where_clause else " WHERE "
        where_clause += "time_granularity = %s"
        params.append(granularity_filter.upper())

    query = f"SELECT COUNT(*) FROM {table_name}{where_clause}"
    cursor.execute(query, params)
    total = cursor.fetchone()[0]

    print(f'\n📊 METADATA TABLE SUMMARY')
    print('='*100)
    print(f'Table: {table_name}')
    if dimension_filter:
        print(f'Filtered by dimension: {dimension_filter}')
    if granularity_filter:
        print(f'Filtered by granularity: {granularity_filter.upper()}')
    print(f'Total collections: {total}\n')

    if total == 0:
        print("ℹ️  No collections found")
        return False

    # By dimension and granularity
    group_query = f"""
        SELECT
            dimension,
            time_granularity,
            COUNT(*) as count,
            MIN(period2_end_date) as earliest,
            MAX(period2_end_date) as latest,
            SUM(total_embeddings) as total_emb
        FROM {table_name}
        {where_clause}
        GROUP BY dimension, time_granularity
        ORDER BY dimension, time_granularity
    """

    cursor.execute(group_query, params)

    print('📋 COLLECTIONS BY DIMENSION & TIME GRANULARITY:')
    print('-'*100)
    print(f"{'Dimension':<25} {'Granularity':<12} {'Count':>6}   {'Total Emb':>10}   {'Date Range'}")
    print('-'*100)

    for dim, gran, count, earliest, latest, total_emb in cursor.fetchall():
        print(f'{dim:<25} {gran:<12} {count:>6}   {total_emb:>10,}   {earliest} to {latest}')

    print()
    return True


def show_detailed_collections(cursor, table_name, dimension_filter=None, granularity_filter=None, limit=50):
    """Display detailed collection information."""
    where_clause = ""
    params = []

    if dimension_filter:
        where_clause += " WHERE dimension = %s"
        params.append(dimension_filter)

    if granularity_filter:
        where_clause += " AND " if where_clause else " WHERE "
        where_clause += "time_granularity = %s"
        params.append(granularity_filter.upper())

    query = f"""
        SELECT
            collection_name,
            dimension,
            time_granularity,
            period1_start_date,
            period1_end_date,
            period2_start_date,
            period2_end_date,
            total_embeddings,
            created_at
        FROM {table_name}
        {where_clause}
        ORDER BY period2_end_date DESC, dimension, time_granularity
        LIMIT %s
    """

    params.append(limit)
    cursor.execute(query, params)

    print(f'📋 DETAILED COLLECTION LIST (Top {limit}):')
    print('-'*120)
    print(f"{'Date':<10} | {'Dim':<20} | {'Gran':<5} | {'Period1':>17} | {'Period2':>17} | {'Emb':>5} | {'Collection Name'}")
    print('-'*120)

    for row in cursor.fetchall():
        cname, dim, gran, p1_start, p1_end, p2_start, p2_end, emb_count, created = row

        # Format periods
        period1 = f"{p1_start}-{p1_end}" if p1_start != p1_end else p1_start
        period2 = f"{p2_start}-{p2_end}" if p2_start != p2_end else p2_start

        # Shorten collection name if needed
        max_name_len = 40
        short_name = cname if len(cname) <= max_name_len else cname[:max_name_len-3] + '...'

        print(f'{p2_end:<10} | {dim:<20} | {gran:<5} | {period1:>17} | {period2:>17} | {emb_count:>5} | {short_name}')

    print()


def show_date_coverage(cursor, table_name, dimension_filter=None):
    """Display date coverage for each time granularity."""
    where_clause = ""
    params = []

    if dimension_filter:
        where_clause = " WHERE dimension = %s"
        params.append(dimension_filter)

    query = f"""
        SELECT
            time_granularity,
            MIN(period1_start_date) as earliest_period1,
            MAX(period2_end_date) as latest_period2,
            COUNT(DISTINCT dimension) as dimension_count
        FROM {table_name}
        {where_clause}
        GROUP BY time_granularity
        ORDER BY time_granularity
    """

    cursor.execute(query, params)

    print(f'📅 DATE COVERAGE BY TIME GRANULARITY:')
    print('-'*80)
    print(f"{'Granularity':<15} {'Earliest Date':<15} {'Latest Date':<15} {'Dimensions'}")
    print('-'*80)

    for gran, earliest, latest, dim_count in cursor.fetchall():
        print(f'{gran:<15} {earliest:<15} {latest:<15} {dim_count}')

    print()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Show collection metadata table contents for a namespace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all collections in revenue_mgmt namespace
  python show_collection_metadata.py --namespace revenue_mgmt

  # Show only device dimension collections
  python show_collection_metadata.py --namespace revenue_mgmt --dimension device

  # Show only WOW collections
  python show_collection_metadata.py --namespace revenue_mgmt --granularity wow

  # Show detailed collection information
  python show_collection_metadata.py --namespace revenue_mgmt --detailed

  # Show only device WOW collections with details
  python show_collection_metadata.py --namespace revenue_mgmt --dimension device --granularity wow --detailed
        """
    )

    parser.add_argument(
        '--namespace', '-n',
        required=True,
        help='Namespace (e.g., revenue_mgmt, campaign_mgmt)'
    )

    parser.add_argument(
        '--dimension', '-d',
        help='Filter by dimension (e.g., device, geo, property)'
    )

    parser.add_argument(
        '--granularity', '-g',
        help='Filter by time granularity (e.g., dod, wow, mom)'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed collection information'
    )

    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=50,
        help='Maximum number of detailed collections to show (default: 50)'
    )

    parser.add_argument(
        '--date-coverage',
        action='store_true',
        help='Show date coverage summary'
    )

    args = parser.parse_args()

    print("="*100)
    print("📊 COLLECTION METADATA VIEWER")
    print("="*100)
    print(f"Namespace: {args.namespace}")
    if args.dimension:
        print(f"Dimension filter: {args.dimension}")
    if args.granularity:
        print(f"Granularity filter: {args.granularity.upper()}")
    print("="*100)

    # Load database configuration
    db_config = load_db_config()

    # Connect to database
    conn = connect_to_database(db_config)
    cursor = conn.cursor()

    # Generate table name
    table_name = f'embeddings_collection_metadata_{args.namespace}'

    # Check if table exists
    try:
        cursor.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
            (table_name,)
        )
        exists = cursor.fetchone()[0]

        if not exists:
            print(f"\n❌ Metadata table '{table_name}' does not exist")
            print(f"\n💡 To create it, run:")
            print(f"   python create_collection_metadata_table.py --namespace {args.namespace}")
            print(f"   python build_collection_metadata.py --namespace {args.namespace}")
            cursor.close()
            conn.close()
            sys.exit(1)

        # Show summary
        has_data = show_summary(cursor, table_name, args.dimension, args.granularity)

        if not has_data:
            cursor.close()
            conn.close()
            sys.exit(0)

        # Show date coverage if requested
        if args.date_coverage:
            show_date_coverage(cursor, table_name, args.dimension)

        # Show detailed collections if requested
        if args.detailed:
            show_detailed_collections(cursor, table_name, args.dimension, args.granularity, args.limit)

        print("="*100)
        print(f"✅ Metadata display complete")
        print("="*100)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        conn.rollback()
        sys.exit(1)

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
