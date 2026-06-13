#!/usr/bin/env python3
"""
Drop Collection Metadata Table

This script drops the collection_metadata table for a specified namespace.

Usage:
    cd dynamic-embeddings
    python drop_collection_metadata_table.py --namespace revenue_mgmt
"""

import sys
import argparse
import psycopg2
from pathlib import Path


def drop_metadata_table(namespace: str, force: bool = False):
    """Drop the collection metadata table for the specified namespace.

    Args:
        namespace: The namespace identifier
        force: If True, skip confirmation prompt
    """
    
    # Load database credentials from .env file
    env_file = Path('.env')
    if not env_file.exists():
        print(f"❌ .env file not found: {env_file}")
        print(f"   Expected location: {Path('.env').resolve()}")
        sys.exit(1)
    
    # Parse .env file
    db_config = {}
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                db_config[key] = value.strip()
    
    # Connect to database
    try:
        print(f"\n🔌 Connecting to PostgreSQL database...")
        conn = psycopg2.connect(
            host=db_config.get('POSTGRES_HOST', 'localhost'),
            port=int(db_config.get('POSTGRES_PORT', 5432)),
            database=db_config.get('POSTGRES_DB', 'vectordb'),
            user=db_config.get('POSTGRES_USER', 'postgres'),
            password=db_config.get('POSTGRES_PASSWORD', '')
        )
        print(f"✅ Connected successfully")
        
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)
    
    cursor = conn.cursor()
    
    # Generate table name
    table_name = f'embeddings_collection_metadata_{namespace}'
    
    print(f"\n🔍 Checking if table '{table_name}' exists...")
    
    try:
        cursor.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
            (table_name,)
        )
        exists = cursor.fetchone()[0]
        
        if exists:
            print(f"✓ Table '{table_name}' found")

            # Ask for confirmation (unless force flag is set)
            if not force:
                print(f"\n⚠️  WARNING: This will permanently delete the table '{table_name}'")
                response = input("Are you sure you want to proceed? (yes/no): ")

                if response.lower() not in ['yes', 'y']:
                    print("❌ Operation cancelled by user")
                    cursor.close()
                    conn.close()
                    sys.exit(0)
            
            # Drop the table
            print(f"\n🗑️  Dropping table '{table_name}'...")
            cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
            conn.commit()
            
            print(f"✅ Table '{table_name}' dropped successfully!")
            print(f"\n💡 To recreate the table, run:")
            print(f"   python create_collection_metadata_table.py --namespace {namespace}")
            print(f"   python build_collection_metadata.py --namespace {namespace}")
            
        else:
            print(f"ℹ️  Table '{table_name}' does not exist")
            print(f"   Nothing to drop")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
        sys.exit(1)
    
    finally:
        cursor.close()
        conn.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Drop collection metadata table for a namespace'
    )
    parser.add_argument(
        '--namespace', '-n',
        required=True,
        help='Namespace (e.g., revenue_mgmt, campaign_mgmt)'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("🗑️  DROP COLLECTION METADATA TABLE")
    print("="*70)
    print(f"Namespace: {args.namespace}")
    print(f"Table: embeddings_collection_metadata_{args.namespace}")
    print("="*70)
    
    # If force flag is set, modify the function to skip confirmation
    if args.force:
        print("⚠️  Force mode enabled - skipping confirmation")

    drop_metadata_table(args.namespace, force=args.force)


if __name__ == "__main__":
    main()
