#!/usr/bin/env python3
"""
Collection Management Tool

This script provides utilities for managing collections in the vector database,
including emptying/clearing collections and viewing collection statistics.

Usage:
    python test_collection_manager.py --list
    python test_collection_manager.py --stats <collection_name>
    python test_collection_manager.py --empty <collection_name>
    python test_collection_manager.py --namespace prod --list
    python test_collection_manager.py --namespace dev --stats <collection_name>
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

from dynamic_embeddings.pipelines.embedding_pipeline import EmbeddingPipeline
from dynamic_embeddings.database.connection import DatabaseConnection


def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_pipeline(namespace='default'):
    """Create and return configured pipeline.

    Args:
        namespace: Namespace for embeddings (default: 'default')
    """
    db_connection = DatabaseConnection(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        database=os.getenv('POSTGRES_DB', 'vectordb'),
        username=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', ''),
    )

    return EmbeddingPipeline(
        database_connection=db_connection,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        embedding_model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large'),
        namespace=namespace
    )


def list_all_collections(pipeline):
    """List all collections and their statistics."""
    print("📚 LISTING ALL COLLECTIONS")
    print("="*60)

    try:
        collections = pipeline.list_collections()

        if not collections:
            print("❌ No collections found in the database.")
            print("💡 Use the JSON loader script to create embeddings first.")
            return True

        total_embeddings = sum(col['total_embeddings'] for col in collections)
        print(f"🗃️  Total Collections: {len(collections)}")
        print(f"🧠 Total Embeddings: {total_embeddings:,}")
        print("-" * 60)

        # Table header
        print(f"{'Collection Name':<25} {'Embeddings':<12} {'Strategies':<20} {'Avg Density':<12}")
        print("-" * 70)

        for collection in collections:
            name = collection['collection_name']
            count = collection['total_embeddings']
            strategies = ', '.join(list(collection['strategies'].keys())[:3])  # Show first 3
            if len(collection['strategies']) > 3:
                strategies += "..."
            density = collection.get('avg_semantic_density', 0)

            print(f"{name:<25} {count:<12,} {strategies:<20} {density:<12.3f}")

        print("-" * 70)
        return True

    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return False


def show_collection_stats(pipeline, collection_name):
    """Show detailed statistics for a specific collection."""
    print(f"📊 COLLECTION STATISTICS: {collection_name}")
    print("="*60)

    try:
        # Get collection stats
        stats = pipeline.get_collection_stats(collection_name)

        if 'error' in stats:
            print(f"❌ Error: {stats['error']}")
            return False

        if stats['total_embeddings'] == 0:
            print(f"❌ Collection '{collection_name}' not found or is empty.")
            return False

        # Display comprehensive stats
        print(f"🏷️  Collection Name: {stats['collection_name']}")
        print(f"🧠 Total Embeddings: {stats['total_embeddings']:,}")
        print(f"📊 Average Semantic Density: {stats['avg_semantic_density']:.3f}")
        print(f"🎯 Average Confidence: {stats.get('avg_confidence', 0):.3f}")

        # Size statistics
        size_stats = stats.get('size_stats', {})
        if size_stats:
            print(f"\n📏 TEXT SIZE STATISTICS:")
            print(f"   Average Length: {size_stats.get('avg_text_length', 0):.1f} chars")
            print(f"   Minimum Length: {size_stats.get('min_text_length', 0):,} chars")
            print(f"   Maximum Length: {size_stats.get('max_text_length', 0):,} chars")

        # Strategy breakdown
        strategies = stats.get('strategies', {})
        if strategies:
            print(f"\n🔧 CHUNKING STRATEGIES:")
            total_chunks = sum(strategies.values())
            for strategy, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_chunks) * 100
                print(f"   {strategy:15} {count:6,} chunks ({percentage:5.1f}%)")

        # Content type breakdown
        content_types = stats.get('content_types', {})
        if content_types:
            print(f"\n📝 CONTENT TYPES:")
            total_content = sum(content_types.values())
            for content_type, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_content) * 100
                print(f"   {content_type:15} {count:6,} chunks ({percentage:5.1f}%)")

        return True

    except Exception as e:
        print(f"❌ Error getting collection stats: {e}")
        return False


def empty_collection(pipeline, collection_name, force=False):
    """Empty/clear a collection by deleting all its embeddings."""
    print(f"🗑️  EMPTYING COLLECTION: {collection_name}")
    print("="*60)

    try:
        # First, get collection stats to show what will be deleted
        stats = pipeline.get_collection_stats(collection_name)

        if 'error' in stats:
            print(f"❌ Error: {stats['error']}")
            return False

        if stats['total_embeddings'] == 0:
            print(f"❌ Collection '{collection_name}' not found or is already empty.")
            return False

        # Show what will be deleted
        print(f"📊 Collection found:")
        print(f"   🏷️  Name: {collection_name}")
        print(f"   🧠 Embeddings: {stats['total_embeddings']:,}")
        print(f"   🔧 Strategies: {list(stats.get('strategies', {}).keys())}")
        print(f"   📝 Content Types: {list(stats.get('content_types', {}).keys())}")

        # Confirmation prompt (unless forced)
        if not force:
            print(f"\n⚠️  WARNING: This will permanently delete all {stats['total_embeddings']:,} embeddings!")
            print(f"⚠️  This action cannot be undone!")

            while True:
                response = input(f"\n❓ Are you sure you want to empty '{collection_name}'? (yes/no): ").strip().lower()
                if response in ['yes', 'y']:
                    break
                elif response in ['no', 'n']:
                    print("❌ Operation cancelled by user.")
                    return False
                else:
                    print("💡 Please enter 'yes' or 'no'")

        # Perform the deletion
        print(f"\n🗑️  Deleting embeddings from '{collection_name}'...")

        deleted_count = pipeline.delete_collection(collection_name)

        if deleted_count > 0:
            print(f"✅ SUCCESS: Deleted {deleted_count:,} embeddings from '{collection_name}'")
            print(f"🧹 Collection '{collection_name}' is now empty.")

            # Verify deletion
            verification_stats = pipeline.get_collection_stats(collection_name)
            if verification_stats['total_embeddings'] == 0:
                print("✅ Deletion verified: Collection is empty.")
            else:
                print(f"⚠️  Warning: {verification_stats['total_embeddings']} embeddings still remain.")

        else:
            print("❌ No embeddings were deleted. Collection may have been empty already.")

        return True

    except Exception as e:
        print(f"❌ Error emptying collection: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Collection Management Tool for Vector Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all collections
  python test_collection_manager.py --list

  # Show statistics for a collection
  python test_collection_manager.py --stats reasoning_output

  # Empty a collection (with confirmation)
  python test_collection_manager.py --empty reasoning_output

  # Empty a collection (skip confirmation)
  python test_collection_manager.py --empty reasoning_output --force

  # Verbose output
  python test_collection_manager.py --list --verbose
        """
    )

    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--list', '-l',
                             action='store_true',
                             help='List all collections and their statistics')
    action_group.add_argument('--empty', '-e',
                             metavar='COLLECTION_NAME',
                             help='Empty/clear the specified collection')
    action_group.add_argument('--stats', '-s',
                             metavar='COLLECTION_NAME',
                             help='Show detailed statistics for the specified collection')

    # Optional arguments
    parser.add_argument('--namespace', '-ns', type=str,
                       default=os.getenv('EMBEDDINGS_NAMESPACE', 'default'),
                       help='Namespace for embeddings (default: %(default)s)')
    parser.add_argument('--force', '-f',
                       action='store_true',
                       help='Skip confirmation prompt when emptying collections')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Check environment (OpenAI key not needed for collection management)
    # But we'll check database connection is possible

    try:
        # Create pipeline
        print(f"Using namespace: {args.namespace}")
        pipeline = create_pipeline(args.namespace)

        # Test database connection
        collections = pipeline.list_collections()

    except Exception as e:
        print(f"❌ ERROR: Cannot connect to database: {e}")
        print("💡 Please check your database connection settings in .env file")
        sys.exit(1)

    # Execute the requested action
    success = False

    try:
        if args.list:
            success = list_all_collections(pipeline)

        elif args.stats:
            success = show_collection_stats(pipeline, args.stats)

        elif args.empty:
            success = empty_collection(pipeline, args.empty, args.force)

        # Cleanup
        pipeline.close()

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        success = False

    # Exit with appropriate code
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()