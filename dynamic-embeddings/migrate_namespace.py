#!/usr/bin/env python3
"""
Namespace Migration Tool

This script provides utilities for migrating data between namespaces,
including migrating from the legacy 'embeddings' table to namespace-based tables.

Usage:
    python migrate_namespace.py --check
    python migrate_namespace.py --legacy
    python migrate_namespace.py --source prod --target staging
    python migrate_namespace.py --source dev --target prod --move
    python migrate_namespace.py --list
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, try manual loading
    env_path = project_root / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

from dynamic_embeddings.database.connection import DatabaseConnection
from dynamic_embeddings.database.migration import NamespaceMigration
from dynamic_embeddings.database.schema import EmbeddingSchema


def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    log_file = f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    if not verbose:
        print(f"📝 Detailed logs: {log_file}")


def create_migration_manager():
    """Create and return migration manager."""
    db_connection = DatabaseConnection(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        database=os.getenv('POSTGRES_DB', 'vectordb'),
        username=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', ''),
    )

    return NamespaceMigration(db_connection.database_url)


def check_migration_needed(args):
    """Check if legacy migration is needed."""
    print("🔍 CHECKING MIGRATION STATUS")
    print("=" * 60)

    try:
        migration = create_migration_manager()

        # Check for legacy table
        has_legacy = migration.table_exists('embeddings')
        has_default = migration.table_exists('embeddings_default')

        print(f"📊 Current State:")
        print(f"   Legacy 'embeddings' table: {'✓ EXISTS' if has_legacy else '✗ Not found'}")
        print(f"   'embeddings_default' table: {'✓ EXISTS' if has_default else '✗ Not found'}")

        if has_legacy and not has_default:
            row_count = migration.get_table_row_count('embeddings')
            print(f"\n⚠️  MIGRATION NEEDED")
            print(f"   The legacy 'embeddings' table contains {row_count:,} rows")
            print(f"   Run: python migrate_namespace.py --legacy")
        elif has_legacy and has_default:
            legacy_count = migration.get_table_row_count('embeddings')
            default_count = migration.get_table_row_count('embeddings_default')
            print(f"\n⚠️  WARNING: Both tables exist")
            print(f"   'embeddings': {legacy_count:,} rows")
            print(f"   'embeddings_default': {default_count:,} rows")
            print(f"   Consider backing up and removing the legacy table")
        else:
            print(f"\n✓ No migration needed")
            print(f"   System is using namespace-based tables")

        return True

    except Exception as e:
        print(f"❌ Error checking migration status: {e}")
        return False


def list_namespaces(args):
    """List all namespaces."""
    print("📚 LISTING ALL NAMESPACES")
    print("=" * 60)

    try:
        db_connection = DatabaseConnection(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', '5432')),
            database=os.getenv('POSTGRES_DB', 'vectordb'),
            username=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', ''),
        )

        schema = EmbeddingSchema(db_connection.database_url)
        namespaces = schema.list_namespaces()

        if not namespaces:
            print("❌ No namespace tables found in the database.")
            print("💡 Create a namespace with: python cli_demo.py --namespace <name> --setup-db")
            return False

        print(f"\nFound {len(namespaces)} namespace(s):\n")

        for ns_info in namespaces:
            print(f"📦 Namespace: {ns_info['namespace']}")
            print(f"   Table: {ns_info['table_name']}")
            print(f"   Embeddings: {ns_info['embedding_count']:,}")
            print(f"   Vector Index: {'✓' if ns_info['vector_index_exists'] else '✗'}")
            print()

        return True

    except Exception as e:
        print(f"❌ Error listing namespaces: {e}")
        return False


def migrate_legacy(args):
    """Migrate legacy 'embeddings' table to 'embeddings_default'."""
    print("🚀 LEGACY TABLE MIGRATION")
    print("=" * 60)
    print(f"Source: embeddings (legacy table)")
    print(f"Target: embeddings_default")
    print()

    try:
        migration = create_migration_manager()

        # Check if migration is needed
        if not migration.table_exists('embeddings'):
            print("❌ Error: Legacy 'embeddings' table does not exist")
            return False

        if migration.table_exists('embeddings_default'):
            print("❌ Error: Target 'embeddings_default' already exists")
            print("💡 The migration may have already been performed")
            return False

        # Get row count
        row_count = migration.get_table_row_count('embeddings')
        print(f"📊 Legacy table contains {row_count:,} rows")

        # Confirm migration
        if not args.force:
            print(f"\n⚠️  This will:")
            print(f"   1. Create 'embeddings_default' table")
            print(f"   2. Copy all {row_count:,} rows")
            print(f"   3. Rename 'embeddings' to 'embeddings_backup_<timestamp>'")
            print(f"   4. This operation uses a transaction (safe)")

            response = input(f"\n❓ Proceed with migration? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Migration cancelled by user.")
                return False

        # Perform migration
        print(f"\n🔄 Starting migration...")
        print(f"⏳ This may take a while for large datasets...")

        result = migration.migrate_legacy_table(target_namespace='default')

        if result['success']:
            print(f"\n✅ MIGRATION SUCCESSFUL!")
            print(f"   Rows migrated: {result['rows_migrated']:,}")
            print(f"   Duration: {result['duration_seconds']:.2f} seconds")
            print(f"   Source action: {result['source_action']}")
            print(f"   Target table: {result['target_table']}")
            print(f"\n💡 You can now use --namespace default for all operations")
            return True
        else:
            print(f"\n❌ MIGRATION FAILED")
            print(f"   Error: {result['error']}")
            return False

    except Exception as e:
        print(f"❌ Migration error: {e}")
        return False


def migrate_namespace(args):
    """Migrate data between namespaces."""
    source = args.source
    target = args.target
    mode = 'move' if args.move else 'copy'

    print(f"🚀 NAMESPACE MIGRATION")
    print("=" * 60)
    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Mode: {mode.upper()}")
    print()

    try:
        migration = create_migration_manager()

        # Determine source table name
        if source == 'embeddings':
            source_table = 'embeddings'
        else:
            source_table = f'embeddings_{source}'

        target_table = f'embeddings_{target}'

        # Validate
        if not migration.table_exists(source_table):
            print(f"❌ Error: Source table '{source_table}' does not exist")
            return False

        if migration.table_exists(target_table):
            print(f"❌ Error: Target table '{target_table}' already exists")
            print(f"💡 Choose a different target namespace or delete the existing one")
            return False

        # Get row count
        row_count = migration.get_table_row_count(source_table)
        print(f"📊 Source contains {row_count:,} rows")

        # Confirm migration
        if not args.force:
            print(f"\n⚠️  This will:")
            print(f"   1. Create '{target_table}' table")
            print(f"   2. Copy all {row_count:,} rows from '{source_table}'")
            if mode == 'move':
                print(f"   3. Rename '{source_table}' to backup")
            else:
                print(f"   3. Keep '{source_table}' unchanged")
            print(f"   4. This operation uses a transaction (safe)")

            response = input(f"\n❓ Proceed with migration? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Migration cancelled by user.")
                return False

        # Perform migration
        print(f"\n🔄 Starting migration...")
        print(f"⏳ This may take a while for large datasets...")

        result = migration.migrate_namespace(source, target, mode=mode)

        if result['success']:
            print(f"\n✅ MIGRATION SUCCESSFUL!")
            print(f"   Rows migrated: {result['rows_migrated']:,}")
            print(f"   Duration: {result['duration_seconds']:.2f} seconds")
            print(f"   Source action: {result['source_action']}")
            print(f"   Target table: {result['target_table']}")
            return True
        else:
            print(f"\n❌ MIGRATION FAILED")
            print(f"   Error: {result['error']}")
            return False

    except Exception as e:
        print(f"❌ Migration error: {e}")
        return False


def dry_run(args):
    """Show what would happen without executing."""
    print("🔍 DRY RUN MODE - No changes will be made")
    print("=" * 60)

    try:
        migration = create_migration_manager()

        if args.legacy:
            source_table = 'embeddings'
            target_table = 'embeddings_default'
            mode = 'move'
        else:
            source = args.source
            target = args.target
            source_table = 'embeddings' if source == 'embeddings' else f'embeddings_{source}'
            target_table = f'embeddings_{target}'
            mode = 'move' if args.move else 'copy'

        # Check existence
        source_exists = migration.table_exists(source_table)
        target_exists = migration.table_exists(target_table)

        print(f"📋 Migration Plan:")
        print(f"   Source table: {source_table} ({'EXISTS' if source_exists else 'NOT FOUND'})")
        print(f"   Target table: {target_table} ({'EXISTS' if target_exists else 'WILL CREATE'})")
        print(f"   Mode: {mode.upper()}")

        if source_exists:
            row_count = migration.get_table_row_count(source_table)
            print(f"   Rows to migrate: {row_count:,}")

        print(f"\n📝 Steps that would be executed:")
        print(f"   1. ✓ Validate source table exists")
        print(f"   2. ✓ Validate target table doesn't exist")
        print(f"   3. ⚙️  Create '{target_table}' with same schema")
        print(f"   4. ⚙️  Copy all data to '{target_table}'")
        print(f"   5. ⚙️  Create vector indexes on '{target_table}'")
        print(f"   6. ⚙️  Verify row counts match")

        if mode == 'move':
            backup_name = f"{source_table}_backup_<timestamp>"
            print(f"   7. ⚙️  Rename '{source_table}' to '{backup_name}'")
        else:
            print(f"   7. ⚙️  Keep '{source_table}' unchanged")

        if not source_exists:
            print(f"\n❌ Cannot proceed: Source table does not exist")
        elif target_exists:
            print(f"\n❌ Cannot proceed: Target table already exists")
        else:
            print(f"\n✓ Dry run complete - migration is feasible")
            print(f"💡 Remove --dry-run to execute the migration")

        return True

    except Exception as e:
        print(f"❌ Dry run error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Namespace Migration Tool for Dynamic Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if legacy migration is needed
  python migrate_namespace.py --check

  # List all namespaces
  python migrate_namespace.py --list

  # Migrate legacy 'embeddings' table to 'embeddings_default'
  python migrate_namespace.py --legacy

  # Copy data from one namespace to another
  python migrate_namespace.py --source prod --target staging

  # Move data (delete source after copy)
  python migrate_namespace.py --source dev --target prod --move

  # Dry run (preview what would happen)
  python migrate_namespace.py --source prod --target staging --dry-run

  # Skip confirmation prompts
  python migrate_namespace.py --legacy --force
        """
    )

    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--check',
                             action='store_true',
                             help='Check if legacy migration is needed')
    action_group.add_argument('--list',
                             action='store_true',
                             help='List all namespace tables')
    action_group.add_argument('--legacy',
                             action='store_true',
                             help='Migrate legacy "embeddings" table to "embeddings_default"')
    action_group.add_argument('--source',
                             metavar='NAMESPACE',
                             help='Source namespace or "embeddings" for legacy table')

    # Target (required when source is specified)
    parser.add_argument('--target',
                       metavar='NAMESPACE',
                       help='Target namespace (required with --source)')

    # Mode
    parser.add_argument('--move',
                       action='store_true',
                       help='Move data (delete source after migration)')

    # Options
    parser.add_argument('--dry-run',
                       action='store_true',
                       help='Show what would happen without executing')
    parser.add_argument('--force', '-f',
                       action='store_true',
                       help='Skip confirmation prompts')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate arguments
    if args.source and not args.target:
        print("❌ ERROR: --target is required when --source is specified")
        parser.print_help()
        sys.exit(1)

    if args.target and not args.source:
        print("❌ ERROR: --source is required when --target is specified")
        parser.print_help()
        sys.exit(1)

    # Execute the requested action
    success = False

    try:
        if args.check:
            success = check_migration_needed(args)

        elif args.list:
            success = list_namespaces(args)

        elif args.dry_run:
            success = dry_run(args)

        elif args.legacy:
            success = migrate_legacy(args)

        elif args.source:
            success = migrate_namespace(args)

    except KeyboardInterrupt:
        print("\n\n⚠️  Migration interrupted by user")
        success = False

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        success = False

    # Exit with appropriate code
    if success:
        print("\n✓ Operation completed successfully")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
