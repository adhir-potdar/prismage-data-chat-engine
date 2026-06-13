#!/usr/bin/env python
"""
Collection Metadata Builder

Rebuilds the collection_metadata table by:
1. Listing all collections from vector store
2. Parsing collection names to extract metadata
3. Updating metadata table with parsed information

Run daily via cron job to keep metadata fresh.

Usage:
    python build_collection_metadata.py [--namespace NAMESPACE]
"""

import re
import os
from pathlib import Path
from datetime import datetime
from sqlalchemy import text

# Load .env from the same directory as this script
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

from dynamic_embeddings.database.connection import DatabaseConnection
from dynamic_embeddings.services.vector_store import VectorStore


def parse_collection_name(collection_name: str) -> dict:
    """
    Parse collection name to extract metadata.

    Format: {dimension}_{granularity}_{start1}_{end1}_vs_{start2}_{end2}
    Example: property_geo_device_qoq_20250601_20250831_vs_20250901_20251130

    Returns:
        dict with dimension, time_granularity, period dates
    """
    # Known time granularities
    time_granularities = ['qoq', 'qtd', 'mom', 'mtd', 'wow', 'wtd', 'dod']

    parts = collection_name.split('_')

    # Find granularity position
    gran_idx = -1
    granularity = None
    for i, part in enumerate(parts):
        if part.lower() in time_granularities:
            gran_idx = i
            granularity = part.lower()
            break

    if gran_idx == -1:
        raise ValueError(f"No time granularity found in collection: {collection_name}")

    # Extract dimension (everything before granularity)
    dimension = '_'.join(parts[:gran_idx])

    # Extract dates after granularity
    # Format: {gran}_{start1}_{end1}_vs_{start2}_{end2}
    date_parts = parts[gran_idx + 1:]

    # Find 'vs' separator
    try:
        vs_idx = date_parts.index('vs')
        period1_dates = date_parts[:vs_idx]
        period2_dates = date_parts[vs_idx + 1:]

        # Validate date format (YYYYMMDD = 8 digits)
        if len(period1_dates) >= 2 and len(period2_dates) >= 2:
            return {
                'dimension': dimension,
                'time_granularity': granularity,
                'period1_start_date': int(period1_dates[0]),
                'period1_end_date': int(period1_dates[1]),
                'period2_start_date': int(period2_dates[0]),
                'period2_end_date': int(period2_dates[1])
            }
    except (ValueError, IndexError):
        pass

    raise ValueError(f"Cannot parse date format from collection: {collection_name}")


def build_metadata_table(namespace: str = 'default'):
    """Rebuild collection metadata table."""

    print(f"🔄 Starting collection metadata rebuild for namespace: {namespace}")
    print("="*60)

    # Generate table name
    table_name = f'embeddings_collection_metadata_{namespace}'
    print(f"   Target table: {table_name}")

    # Initialize services
    db_conn = DatabaseConnection()
    vector_store = VectorStore(db_conn)

    # Clear model cache to ensure we get the latest schema
    vector_store.table_factory.clear_cache()

    # Step 1: Get all collections (lightweight - only names)
    print("\n📋 Fetching collection names...")

    with db_conn.get_session() as session:
        # Get distinct collection names (no expensive stats)
        RecordModel = vector_store.table_factory.get_or_create_model(namespace)
        collections = session.query(RecordModel.collection_name).distinct().all()
        collection_names = [col[0] for col in collections]

    print(f"   Found {len(collection_names)} collections")

    # Step 2: Parse collection names
    print("\n🔍 Parsing collection names...")

    parsed_metadata = []
    parse_errors = []

    # Namespace prefix to strip from dimensions
    namespace_prefix = f"{namespace}_"

    for collection_name in collection_names:
        try:
            metadata = parse_collection_name(collection_name)
            metadata['collection_name'] = collection_name

            # Strip namespace prefix from dimension
            # Example: "revenue_mgmt_property" -> "property"
            if metadata['dimension'].startswith(namespace_prefix):
                metadata['dimension'] = metadata['dimension'][len(namespace_prefix):]

            # Get embedding count and dimension values (single query)
            with db_conn.get_session() as session:
                count = session.query(RecordModel).filter(
                    RecordModel.collection_name == collection_name
                ).count()
                metadata['total_embeddings'] = count

                # Extract distinct dimension_values from this collection
                # Check if dimension_value attribute exists (may not be present in cached models)
                if hasattr(RecordModel, 'dimension_value'):
                    dimension_values_query = session.query(
                        RecordModel.dimension_value
                    ).filter(
                        RecordModel.collection_name == collection_name,
                        RecordModel.dimension_value.isnot(None)
                    ).distinct().all()

                    # Convert to list, filtering out None values
                    dimension_values = [dv[0] for dv in dimension_values_query if dv[0]]
                    metadata['dimension_values'] = dimension_values if dimension_values else None
                else:
                    # If dimension_value column doesn't exist in model, use raw SQL
                    from sqlalchemy import text
                    try:
                        dimension_values_query = session.execute(text(f"""
                            SELECT DISTINCT dimension_value
                            FROM embeddings_{namespace}
                            WHERE collection_name = :collection_name
                            AND dimension_value IS NOT NULL
                        """), {"collection_name": collection_name}).fetchall()
                        dimension_values = [dv[0] for dv in dimension_values_query if dv[0]]
                        metadata['dimension_values'] = dimension_values if dimension_values else None
                    except Exception as e:
                        # Column might not exist in database either
                        print(f"   ⚠️  Could not fetch dimension_values: {e}")
                        metadata['dimension_values'] = None

            metadata['last_updated_at'] = datetime.utcnow()
            parsed_metadata.append(metadata)

        except ValueError as e:
            parse_errors.append({'collection': collection_name, 'error': str(e)})

    print(f"   Successfully parsed: {len(parsed_metadata)}")
    if parse_errors:
        print(f"   ⚠️  Parse errors: {len(parse_errors)}")
        for err in parse_errors[:5]:  # Show first 5 errors
            print(f"      - {err['collection']}: {err['error']}")

    # Step 3: Update metadata table (bulk insert/update)
    print("\n💾 Updating metadata table...")

    with db_conn.get_session() as session:
        # Clear existing metadata
        session.execute(text(f"DELETE FROM {table_name}"))

        # Bulk insert new metadata
        # Convert dimension_values to JSON string for JSONB column
        import json
        for metadata in parsed_metadata:
            if metadata.get('dimension_values'):
                metadata['dimension_values'] = json.dumps(metadata['dimension_values'])

        # Insert using bindparam style for bulk insert
        stmt = text(f"""
            INSERT INTO {table_name}
            (collection_name, dimension, time_granularity, dimension_values,
             period1_start_date, period1_end_date,
             period2_start_date, period2_end_date,
             total_embeddings, last_updated_at)
            VALUES
            (:collection_name, :dimension, :time_granularity, CAST(:dimension_values AS JSONB),
             :period1_start_date, :period1_end_date,
             :period2_start_date, :period2_end_date,
             :total_embeddings, :last_updated_at)
        """)

        session.execute(stmt, parsed_metadata)

        session.commit()

    print(f"   ✅ Inserted {len(parsed_metadata)} metadata records")

    # Step 4: Summary
    print(f"\n{'='*60}")
    print(f"✅ Metadata rebuild complete!")
    print(f"   Collections processed: {len(collection_names)}")
    print(f"   Metadata records: {len(parsed_metadata)}")
    print(f"   Parse errors: {len(parse_errors)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build collection metadata table')
    parser.add_argument('--namespace', default='default', help='Namespace to process')
    args = parser.parse_args()

    try:
        build_metadata_table(namespace=args.namespace)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
