#!/usr/bin/env python3
"""
CLI Demo Script for Dynamic JSON Embeddings Pipeline

Quick start script to test the complete pipeline with sample data.

Usage:
    python cli_demo.py --setup-db                        # Setup database only
    python cli_demo.py --process-sample                  # Process sample data
    python cli_demo.py --search "query"                  # Search embeddings
    python cli_demo.py --stats                           # Show collection stats
    python cli_demo.py --full-demo                       # Run complete demo
    python cli_demo.py --namespace prod --setup-db       # Setup with custom namespace
    python cli_demo.py --namespace dev --process-sample  # Process in dev namespace
    python cli_demo.py --list-namespaces                 # List all namespaces
    python cli_demo.py --namespace-stats prod            # Show namespace stats
    python cli_demo.py --drop-namespace test_ns          # Delete namespace
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, try manual loading
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

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


def create_sample_data():
    """Create sample Ad-Tech data."""
    return {
        "campaign_id": "demo_camp_001",
        "campaign_name": "CLI Demo Campaign",
        "campaign_type": "search",
        "status": "active",
        "performance": {
            "impressions": 50000,
            "clicks": 1200,
            "conversions": 48,
            "ctr": 2.4,
            "conversion_rate": 4.0,
            "cost_per_click": 1.50
        },
        "targeting": {
            "keywords": ["tech gadgets", "electronics", "mobile phones"],
            "demographics": {"age": "25-45", "income": "middle_high"},
            "locations": ["US", "CA", "UK"]
        },
        "budget": {
            "daily_budget": 500.0,
            "total_spent": 3600.0,
            "remaining": 11400.0
        }
    }


def setup_database(args):
    """Setup database schema and extensions."""
    namespace = getattr(args, 'namespace', 'default')
    print(f"Setting up database for namespace '{namespace}'...")

    try:
        pipeline = create_pipeline(namespace)
        result = pipeline.setup_database()

        if result['success']:
            print("✓ Database setup completed successfully!")
            print(f"✓ Namespace: {result['namespace']}")
            connection_info = result['connection_info']
            print(f"✓ PostgreSQL: {connection_info['postgresql_version']}")
            print(f"✓ PGVector: {'Installed' if connection_info['pgvector_installed'] else 'Not installed'}")
        else:
            print(f"✗ Database setup failed: {result['error']}")
            return False

        pipeline.close()
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def process_sample(args):
    """Process sample data through the pipeline."""
    namespace = getattr(args, 'namespace', 'default')
    print(f"Processing sample Ad-Tech data in namespace '{namespace}'...")

    try:
        pipeline = create_pipeline(namespace)
        sample_data = create_sample_data()

        result = pipeline.process_json_data(
            json_data=sample_data,
            collection_name="cli_demo",
            document_id="demo_campaign_001"
        )

        if result['success']:
            print("✓ Sample data processed successfully!")
            print(f"✓ Namespace: {namespace}")
            print(f"✓ Generated {result['total_embeddings']} embeddings")
            print(f"✓ Collection: {result['collection_name']}")
            print(f"✓ Document ID: {result['document_id']}")

            # Show some statistics
            chunking_stats = result['chunking_stats']
            embedding_stats = result['embedding_stats']
            print(f"✓ Chunks created: {chunking_stats['total_chunks']}")
            print(f"✓ API tokens used: {embedding_stats['api_usage']['total_tokens']}")

        else:
            print(f"✗ Processing failed: {result['error']}")

        pipeline.close()

    except Exception as e:
        print(f"✗ Error: {e}")


def search_embeddings(args):
    """Search for similar embeddings."""
    namespace = getattr(args, 'namespace', 'default')
    query = args.query
    print(f"🔍 Searching for: '{query}' in namespace '{namespace}'")
    print("=" * 60)

    try:
        pipeline = create_pipeline(namespace)

        # Use configurable threshold from environment, default to 0.0 for demo
        default_threshold = float(os.getenv('DEFAULT_SIMILARITY_THRESHOLD', '0.0'))
        limit = int(os.getenv('SEARCH_RESULT_LIMIT', '5'))

        search_info = pipeline.search_similar(
            query_text=query,
            collection_name="cli_demo",
            limit=limit,
            similarity_threshold=default_threshold
        )

        if 'error' in search_info:
            print(f"✗ Search Error: {search_info['error']}")
            return

        results = search_info['results']
        total_embeddings = search_info['total_embeddings']
        matched_embeddings = search_info['matched_embeddings']

        print(f"📊 SEARCH STATISTICS:")
        print(f"   Total Embeddings in Collection: {total_embeddings}")
        print(f"   Matched Embeddings Returned: {matched_embeddings}")
        print(f"   Collection: {search_info['collection_name']}")
        print(f"   Search Algorithm: Cosine Similarity")
        print()

        if results:
            print(f"🎯 SIMILARITY RESULTS (Top {len(results)} matches):")
            print("-" * 60)
            for i, (record, similarity) in enumerate(results, 1):
                print(f"\n📋 Result {i}:")
                print(f"   📈 Similarity Score: {similarity:.4f}")
                print(f"   🏷️  Chunk ID: {record.chunk_id}")
                print(f"   🔧 Strategy: {record.strategy}")
                print(f"   📝 Content Type: {record.content_type}")
                print(f"   📍 Path: {record.path}")
                print(f"   📊 Semantic Density: {record.semantic_density:.3f}")
                print(f"   🕒 Created: {record.created_at}")
                print(f"   📄 FULL TEXT:")
                print(f"      {record.text}")

                if i < len(results):
                    print("   " + "-" * 55)
        else:
            print("❌ No similar embeddings found.")

        pipeline.close()

    except Exception as e:
        print(f"✗ Error: {e}")


def show_stats(args):
    """Show collection statistics."""
    namespace = getattr(args, 'namespace', 'default')
    print(f"Collection Statistics for namespace '{namespace}':")
    print("=" * 60)

    try:
        pipeline = create_pipeline(namespace)

        collections = pipeline.list_collections()

        if collections:
            # Show summary
            total_embeddings = sum(col['total_embeddings'] for col in collections)
            print(f"\n📊 SUMMARY:")
            print(f"   Total Collections: {len(collections)}")
            print(f"   Total Embeddings: {total_embeddings:,}")
            print(f"\n{'='*60}\n")

            # Show individual collections
            for i, collection in enumerate(collections, 1):
                print(f"📁 Collection {i}: {collection['collection_name']}")
                print(f"   Embeddings: {collection['total_embeddings']:,}")
                print(f"   Strategies: {list(collection['strategies'].keys())}")
                print(f"   Content Types: {list(collection['content_types'].keys())}")
                print(f"   Avg Semantic Density: {collection.get('avg_semantic_density', 0):.3f}")
                if i < len(collections):
                    print()
        else:
            print("\n❌ No collections found.")

        pipeline.close()

    except Exception as e:
        print(f"✗ Error: {e}")


def run_full_demo(args):
    """Run complete demonstration."""
    namespace = getattr(args, 'namespace', 'default')
    print(f"Running Full Demo in namespace '{namespace}'...")
    print("=" * 50)

    # Setup database
    print("\n1. Setting up database...")
    if not setup_database(args):
        return

    # Process sample data
    print("\n2. Processing sample data...")
    process_sample(args)

    # Run some searches
    print("\n3. Running similarity searches...")

    test_queries = [
        "campaign performance metrics",
        "budget and spending",
        "targeting demographics"
    ]

    for query in test_queries:
        print(f"\nSearching: '{query}'")
        try:
            pipeline = create_pipeline(namespace)

            # Use configurable threshold from environment
            demo_threshold = float(os.getenv('DEFAULT_SIMILARITY_THRESHOLD', '0.4'))
            demo_limit = int(os.getenv('SEARCH_RESULT_LIMIT', '2'))

            search_info = pipeline.search_similar(
                query_text=query,
                collection_name="cli_demo",
                limit=demo_limit,
                similarity_threshold=demo_threshold
            )

            if 'error' in search_info:
                print(f"  Error: {search_info['error']}")
            else:
                results = search_info['results']
                matched_embeddings = search_info['matched_embeddings']
                total_embeddings = search_info['total_embeddings']

                print(f"  Found {matched_embeddings}/{total_embeddings} matches:")

                if results:
                    for i, (record, similarity) in enumerate(results, 1):
                        print(f"    {i}. Similarity: {similarity:.4f} - {record.chunk_id}")
                        print(f"       Strategy: {record.strategy} | Type: {record.content_type}")
                        print(f"       Text: {record.text[:80]}...")
                else:
                    print("    No results found above threshold.")

            pipeline.close()
        except Exception as e:
            print(f"  Error: {e}")

    # Show statistics
    print("\n4. Collection statistics...")
    show_stats(args)

    print("\n" + "=" * 50)
    print("Full demo completed!")


def list_namespaces(args):
    """List all namespaces with statistics."""
    print("📚 LISTING ALL NAMESPACES")
    print("=" * 60)

    try:
        # Create a pipeline (namespace doesn't matter for listing)
        pipeline = create_pipeline()

        namespaces = pipeline.list_namespaces()

        if not namespaces:
            print("❌ No namespaces found in the database.")
            print("💡 Create a namespace with: python cli_demo.py --namespace <name> --setup-db")
            return

        print(f"\nFound {len(namespaces)} namespace(s):\n")

        for ns_info in namespaces:
            print(f"📦 Namespace: {ns_info['namespace']}")
            print(f"   Table: {ns_info['table_name']}")
            print(f"   Embeddings: {ns_info['embedding_count']:,}")
            print(f"   Vector Index: {'✓' if ns_info.get('vector_index_exists', False) else '✗'}")
            print()

        pipeline.close()

    except Exception as e:
        print(f"✗ Error: {e}")


def namespace_stats(args):
    """Show detailed statistics for a specific namespace."""
    namespace = args.namespace_stats
    print(f"📊 NAMESPACE STATISTICS: {namespace}")
    print("=" * 60)

    try:
        # Create a pipeline (any namespace works for getting stats)
        pipeline = create_pipeline()

        stats = pipeline.get_namespace_stats(namespace)

        if not stats.get('exists', False):
            print(f"❌ Namespace '{namespace}' does not exist")
            print("💡 List available namespaces with: python cli_demo.py --list-namespaces")
            pipeline.close()
            return

        print(f"\n✓ Namespace: {stats['namespace']}")
        print(f"  Table: {stats['table_name']}")
        print(f"  Embeddings: {stats['embedding_count']:,}")
        print(f"  Collections: {stats['collection_count']}")
        print(f"  Table Size: {stats['table_size']}")
        print(f"  Vector Index: {'✓ Yes' if stats['vector_index_exists'] else '✗ No'}")

        pipeline.close()

    except Exception as e:
        print(f"✗ Error: {e}")


def drop_namespace(args):
    """Delete a namespace."""
    namespace = args.drop_namespace
    print(f"🗑️  DROP NAMESPACE: {namespace}")
    print("=" * 60)

    try:
        # Create a pipeline
        pipeline = create_pipeline()

        # Check if namespace exists
        stats = pipeline.get_namespace_stats(namespace)

        if not stats.get('exists', False):
            print(f"❌ Namespace '{namespace}' does not exist")
            print("💡 List available namespaces with: python cli_demo.py --list-namespaces")
            pipeline.close()
            return

        # Show what will be deleted
        print(f"\n⚠️  WARNING: This will permanently delete:")
        print(f"   Namespace: {namespace}")
        print(f"   Table: {stats['table_name']}")
        print(f"   Embeddings: {stats['embedding_count']:,}")
        print(f"   Collections: {stats['collection_count']}")
        print(f"\n⚠️  This action cannot be undone!")

        # Confirm deletion
        if not args.force:
            response = input(f"\n❓ Are you sure you want to delete namespace '{namespace}'? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Operation cancelled by user.")
                pipeline.close()
                return

        # Perform deletion
        print(f"\n🗑️  Deleting namespace '{namespace}'...")
        result = pipeline.drop_namespace(namespace, confirm=True)

        if result:
            print(f"✅ Namespace '{namespace}' deleted successfully")
        else:
            print(f"❌ Failed to delete namespace '{namespace}'")

        pipeline.close()

    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI Demo for Dynamic JSON Embeddings Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--setup-db', action='store_true',
                       help='Setup database schema and extensions')
    parser.add_argument('--process-sample', action='store_true',
                       help='Process sample Ad-Tech data')
    parser.add_argument('--search', type=str, metavar='QUERY',
                       help='Search for similar embeddings')
    parser.add_argument('--stats', action='store_true',
                       help='Show collection statistics')
    parser.add_argument('--full-demo', action='store_true',
                       help='Run complete demonstration')
    parser.add_argument('--namespace', '-ns', type=str,
                       default=os.getenv('EMBEDDINGS_NAMESPACE', 'default'),
                       help='Namespace for embeddings (default: %(default)s)')
    parser.add_argument('--list-namespaces', action='store_true',
                       help='List all namespaces with statistics')
    parser.add_argument('--namespace-stats', type=str, metavar='NAMESPACE',
                       help='Show detailed statistics for a namespace')
    parser.add_argument('--drop-namespace', type=str, metavar='NAMESPACE',
                       help='Delete a namespace (requires confirmation)')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Skip confirmation prompts')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print("✗ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in .env file or environment")
        sys.exit(1)

    # Execute commands
    if args.setup_db:
        setup_database(args)
    elif args.process_sample:
        process_sample(args)
    elif args.search:
        args.query = args.search
        search_embeddings(args)
    elif args.stats:
        show_stats(args)
    elif args.full_demo:
        run_full_demo(args)
    elif args.list_namespaces:
        list_namespaces(args)
    elif args.namespace_stats:
        namespace_stats(args)
    elif args.drop_namespace:
        drop_namespace(args)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python cli_demo.py --full-demo")


if __name__ == "__main__":
    main()