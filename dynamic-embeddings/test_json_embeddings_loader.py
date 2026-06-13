#!/usr/bin/env python3
"""
JSON Folder Parser and Embedding Loader

This script processes all JSON files in a folder, chunks them,
and loads embeddings into the vector database.

Usage:
    python test_json_embeddings_loader.py <path_to_folder>
    python test_json_embeddings_loader.py /Users/adhirpotdar/Work/YuktaMediaLLP/yieldmgmt-v2/reasoning_output/
"""

import os
import sys
import json
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
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'json_loader_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def create_pipeline(namespace='default', skip_setup=False):
    """Create and return configured pipeline.

    Args:
        namespace: Namespace for embeddings (default: 'default')
        skip_setup: Skip database setup if namespace already exists (default: False)
    """
    db_connection = DatabaseConnection(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        database=os.getenv('POSTGRES_DB', 'vectordb'),
        username=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', ''),
    )

    pipeline = EmbeddingPipeline(
        database_connection=db_connection,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        embedding_model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large'),
        namespace=namespace
    )

    # Mark setup as done if skipping (prevents setup_database() calls)
    if skip_setup:
        pipeline._setup_done = True

    return pipeline


def load_and_validate_json(json_path):
    """Load and validate JSON file."""
    json_file = Path(json_path)

    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    if not json_file.suffix.lower() == '.json':
        raise ValueError(f"File must have .json extension: {json_path}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logging.info(f"Successfully loaded JSON file: {json_file.name}")
        logging.info(f"File size: {json_file.stat().st_size / 1024:.2f} KB")

        # Basic structure validation
        if isinstance(data, dict):
            logging.info(f"Top-level keys: {list(data.keys())}")
        elif isinstance(data, list):
            logging.info(f"Array with {len(data)} items")

        return data, json_file.stem

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_path}: {e}")
    except Exception as e:
        raise Exception(f"Error reading {json_path}: {e}")


def process_json_to_embeddings(json_path, collection_name, pipeline, replace_existing=True):
    """Process JSON file through the embedding pipeline."""

    logging.info(f"Starting processing of: {json_path}")

    try:
        # Load and validate JSON
        json_data, file_stem = load_and_validate_json(json_path)

        # Determine document ID
        document_id = file_stem

        # Check if document already exists and delete if replace_existing is True
        if replace_existing:
            try:
                # First, check if document exists in ANY collection and delete it
                existing_count = pipeline.delete_document(document_id)  # Delete from all collections
                if existing_count > 0:
                    logging.info(f"🔄 Replaced existing document: deleted {existing_count} embeddings for '{document_id}' from all collections")
            except Exception as e:
                # Table might not exist yet - that's okay, nothing to delete
                logging.debug(f"Could not delete existing document (table may not exist yet): {e}")

        # Process JSON data
        logging.info(f"Processing JSON data for document: {document_id}")

        result = pipeline.process_json_data(
            json_data=json_data,
            collection_name=collection_name,
            document_id=document_id
        )

        if result['success']:
            # Document Chunking Statistics
            chunking_stats = result['chunking_stats']
            embedding_stats = result['embedding_stats']

            # Extract detailed chunking strategy information
            strategies_used = chunking_stats.get('strategies_used', {})

            file_stats = {
                'file_name': Path(json_path).name,
                'file_size_kb': Path(json_path).stat().st_size / 1024,
                'document_id': document_id,
                'collection_name': collection_name,
                'success': True,
                'total_chunks': chunking_stats['total_chunks'],
                'total_embeddings': result['total_embeddings'],
                'strategies_used': strategies_used,
                'primary_strategy': max(strategies_used.items(), key=lambda x: x[1])[0] if strategies_used else 'unknown',
                'api_tokens': embedding_stats['api_usage']['total_tokens'],
                'embedding_model': embedding_stats['embedding_model'],
                'vector_dimensions': embedding_stats['vector_dimensions']
            }

            logging.info("✓ JSON processing successful!")
            logging.info(f"✓ Total Chunks: {chunking_stats['total_chunks']}")
            logging.info(f"✓ Strategies Used: {list(strategies_used.keys())}")
            logging.info(f"✓ API Tokens Used: {embedding_stats['api_usage']['total_tokens']}")

            return file_stats

        else:
            logging.error(f"✗ Processing failed: {result['error']}")
            return {
                'file_name': Path(json_path).name,
                'file_size_kb': Path(json_path).stat().st_size / 1024,
                'document_id': file_stem,
                'collection_name': collection_name,
                'success': False,
                'error': result['error']
            }

    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return {
            'file_name': Path(json_path).name,
            'file_size_kb': 0,
            'document_id': Path(json_path).stem,
            'collection_name': collection_name,
            'success': False,
            'error': str(e)
        }


def find_json_files(folder_path):
    """Find all JSON files in the given folder."""
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    json_files = list(folder.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in: {folder_path}")

    # Sort by file size (smallest first) for better processing order
    json_files.sort(key=lambda x: x.stat().st_size)

    return json_files


def display_summary_statistics(all_results):
    """Display comprehensive summary statistics."""

    successful_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]

    print("\n" + "="*80)
    print("📊 COMPREHENSIVE PROCESSING SUMMARY")
    print("="*80)

    # Overall Statistics
    print(f"📁 Total Files Processed: {len(all_results)}")
    print(f"✅ Successful: {len(successful_results)}")
    print(f"❌ Failed: {len(failed_results)}")

    if successful_results:
        total_chunks = sum(r['total_chunks'] for r in successful_results)
        total_embeddings = sum(r['total_embeddings'] for r in successful_results)
        total_tokens = sum(r['api_tokens'] for r in successful_results)
        total_size_mb = sum(r['file_size_kb'] for r in successful_results) / 1024

        print(f"📦 Total Chunks Created: {total_chunks:,}")
        print(f"🧠 Total Embeddings Generated: {total_embeddings:,}")
        print(f"🔤 Total API Tokens Used: {total_tokens:,}")
        print(f"📏 Total Data Processed: {total_size_mb:.2f} MB")

        # Strategy Analysis Across All Files
        print(f"\n🔧 CHUNKING STRATEGY ANALYSIS:")
        print("-" * 60)

        all_strategies = {}
        strategy_by_file = {}

        for result in successful_results:
            for strategy, count in result['strategies_used'].items():
                all_strategies[strategy] = all_strategies.get(strategy, 0) + count

            primary = result['primary_strategy']
            strategy_by_file[primary] = strategy_by_file.get(primary, 0) + 1

        print(f"📈 Total Chunks by Strategy:")
        for strategy, count in sorted(all_strategies.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_chunks) * 100
            print(f"   {strategy:15} {count:6,} chunks ({percentage:5.1f}%)")

        print(f"\n📋 Files by Primary Strategy:")
        for strategy, count in sorted(strategy_by_file.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(successful_results)) * 100
            print(f"   {strategy:15} {count:3} files ({percentage:5.1f}%)")

    # Individual File Details
    print(f"\n📄 INDIVIDUAL FILE DETAILS:")
    print("-" * 95)
    print(f"{'File Name':<35} {'Collection':<18} {'Size(KB)':<10} {'Chunks':<8} {'Primary Strategy':<15} {'Status':<10}")
    print("-" * 95)

    for result in all_results:
        if result['success']:
            status = "✅ SUCCESS"
            chunks = f"{result['total_chunks']:,}"
            strategy = result['primary_strategy']
        else:
            status = "❌ FAILED"
            chunks = "-"
            strategy = "-"

        collection = result['collection_name']
        print(f"{result['file_name']:<35} {collection:<18} {result['file_size_kb']:<10.1f} {chunks:<8} {strategy:<15} {status:<10}")

    # Detailed Strategy Breakdown per File
    if successful_results:
        print(f"\n🔍 DETAILED STRATEGY BREAKDOWN PER FILE:")
        print("-" * 80)

        for result in successful_results:
            print(f"\n📁 {result['file_name']} → Collection: {result['collection_name']} (Total: {result['total_chunks']} chunks)")
            strategies = result['strategies_used']
            for strategy, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / result['total_chunks']) * 100
                print(f"   {strategy:15} {count:4} chunks ({percentage:5.1f}%)")

    # Failed Files Details
    if failed_results:
        print(f"\n❌ FAILED FILES DETAILS:")
        print("-" * 60)
        for result in failed_results:
            print(f"📁 {result['file_name']}")
            print(f"   Error: {result['error']}")

    print("\n" + "="*80)


def process_folder(folder_path, collection_name, namespace='default'):
    """Process all JSON files in a folder."""

    print(f"🚀 PROCESSING JSON FILES IN FOLDER")
    print("="*60)
    print(f"📁 Folder: {folder_path}")
    print(f"📦 Collection: {collection_name}")
    print(f"📦 Namespace: {namespace}")

    try:
        # Find all JSON files
        json_files = find_json_files(folder_path)
        print(f"📄 Found {len(json_files)} JSON files")

        # Create single pipeline instance for all files
        pipeline = create_pipeline(namespace)

        # Setup database once
        print(f"\n⚙️  Setting up database...")
        setup_result = pipeline.setup_database()
        if not setup_result['success']:
            raise Exception(f"Database setup failed: {setup_result.get('error', 'Unknown error')}")
        print("✅ Database setup successful")

        print(f"\n📊 PROCESSING FILES:")
        print("-" * 60)

        all_results = []
        processed_count = 0

        for i, json_file in enumerate(json_files, 1):
            print(f"\n[{i}/{len(json_files)}] Processing: {json_file.name}")
            print(f"   Size: {json_file.stat().st_size / 1024:.1f} KB")

            # Process the file (with replacement enabled by default for batch processing)
            result = process_json_to_embeddings(str(json_file), collection_name, pipeline, replace_existing=True)
            all_results.append(result)

            if result['success']:
                print(f"   ✅ Success - {result['total_chunks']} chunks, Primary strategy: {result['primary_strategy']}")
                processed_count += 1
            else:
                print(f"   ❌ Failed - {result['error']}")

            # Show progress
            if processed_count > 0:
                avg_chunks = sum(r['total_chunks'] for r in all_results if r['success']) / processed_count
                print(f"   📈 Progress: {processed_count}/{len(json_files)} files, Avg chunks: {avg_chunks:.1f}")

        # Display comprehensive summary
        display_summary_statistics(all_results)

        # Cleanup
        pipeline.close()

        successful_count = len([r for r in all_results if r['success']])

        if successful_count == len(json_files):
            print(f"\n🎉 ALL FILES PROCESSED SUCCESSFULLY!")
        elif successful_count > 0:
            print(f"\n⚠️  PARTIAL SUCCESS: {successful_count}/{len(json_files)} files processed")
        else:
            print(f"\n❌ ALL FILES FAILED TO PROCESS")

        print("You can now use the interactive Q&A script to search these embeddings!")
        return successful_count > 0

    except Exception as e:
        print(f"\n❌ Error processing folder: {e}")
        return False


def process_single_file(file_path, collection_name, replace_existing=True, namespace='default'):
    """Process a single JSON file."""

    print(f"🚀 PROCESSING SINGLE JSON FILE")
    print("="*60)
    print(f"📁 File: {file_path}")
    print(f"📦 Collection: {collection_name}")
    print(f"📦 Namespace: {namespace}")
    print(f"🔄 Replace existing: {'Yes' if replace_existing else 'No'}")

    try:
        # Create pipeline instance
        pipeline = create_pipeline(namespace)

        # Check if namespace exists before expensive setup
        namespace_exists = pipeline.schema.namespace_exists(namespace)

        if namespace_exists:
            # Namespace already exists, skip expensive setup_database() call
            print(f"\n⚙️  Namespace '{namespace}' already exists, skipping setup...")
            print("✅ Using existing database schema")
        else:
            # Setup database once for new namespace
            print(f"\n⚙️  Setting up database...")
            setup_result = pipeline.setup_database()
            if not setup_result['success']:
                raise Exception(f"Database setup failed: {setup_result.get('error', 'Unknown error')}")
            print("✅ Database setup successful")

        print(f"\n📊 PROCESSING FILE:")
        print("-" * 60)

        # Process the file
        result = process_json_to_embeddings(file_path, collection_name, pipeline, replace_existing)

        if result['success']:
            print(f"\n✅ SUCCESS - File processed successfully!")
            print(f"📦 Total Chunks: {result['total_chunks']:,}")
            print(f"🧠 Total Embeddings: {result['total_embeddings']:,}")
            print(f"🔧 Primary Strategy: {result['primary_strategy']}")
            print(f"🔤 API Tokens Used: {result['api_tokens']:,}")
            print(f"📏 Vector Dimensions: {result['vector_dimensions']}")

            # Display strategy breakdown
            print(f"\n🔍 STRATEGY BREAKDOWN:")
            strategies = result['strategies_used']
            for strategy, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / result['total_chunks']) * 100
                print(f"   {strategy:15} {count:4} chunks ({percentage:5.1f}%)")

        else:
            print(f"\n❌ FAILED - {result['error']}")

        # Cleanup
        pipeline.close()

        return result['success']

    except Exception as e:
        print(f"\n❌ Error processing file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Process JSON files into embeddings - supports both folder and single file processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire folder
  python test_json_embeddings_loader.py /path/to/json_folder
  python test_json_embeddings_loader.py /path/to/json_folder --collection my_collection

  # Process single file
  python test_json_embeddings_loader.py --file /path/to/single_file.json --collection my_collection

  # Process single file without replacing existing (will fail on duplicates)
  python test_json_embeddings_loader.py --file /path/to/single_file.json --collection my_collection --no-replace

  # Verbose logging
  python test_json_embeddings_loader.py /path/to/folder --verbose
        """
    )

    parser.add_argument('folder_path', nargs='?',
                       help='Path to folder containing JSON files (optional if using --file)')
    parser.add_argument('--file', '-f',
                       help='Process a single JSON file instead of a folder')
    parser.add_argument('--collection', '-c',
                       default='reasoning_output',
                       help='Collection name for embeddings (default: reasoning_output)')
    parser.add_argument('--namespace', '-ns', type=str,
                       default=os.getenv('EMBEDDINGS_NAMESPACE', 'default'),
                       help='Namespace for embeddings (default: %(default)s)')
    parser.add_argument('--no-replace',
                       action='store_true',
                       help='Do not replace existing documents (will fail on duplicates)')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print("✗ ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in .env file or environment")
        sys.exit(1)

    # Determine processing mode
    if args.file:
        # Single file processing
        file_path = Path(args.file).resolve()

        if not file_path.exists():
            print(f"❌ ERROR: File does not exist: {file_path}")
            sys.exit(1)

        if not file_path.suffix.lower() == '.json':
            print(f"❌ ERROR: File must have .json extension: {file_path}")
            sys.exit(1)

        replace_existing = not args.no_replace
        success = process_single_file(str(file_path), args.collection, replace_existing, args.namespace)

    elif args.folder_path:
        # Folder processing (existing functionality)
        folder_path = Path(args.folder_path).resolve()

        if not folder_path.exists():
            print(f"❌ ERROR: Path does not exist: {folder_path}")
            sys.exit(1)

        success = process_folder(str(folder_path), args.collection, args.namespace)

    else:
        print("❌ ERROR: Must provide either folder_path or --file argument")
        parser.print_help()
        sys.exit(1)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()