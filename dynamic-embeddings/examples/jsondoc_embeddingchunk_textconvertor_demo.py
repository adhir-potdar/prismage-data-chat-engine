"""
Document Chunking Demo: Complete JSON-to-Embeddings Pipeline

This example demonstrates the complete document chunking implementation:
- Loading JSON documents
- Automatic strategy selection
- Text conversion for embeddings
- Quality validation
- Export capabilities
"""

import json
import logging
from pathlib import Path

from dynamic_embeddings import DocumentProcessor, EmbeddingChunk


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_sample_documents():
    """Create sample JSON documents for demonstration."""

    # Simple flat document
    simple_doc = {
        "user_id": "user123",
        "name": "John Doe",
        "email": "john.doe@example.com",
        "age": 30,
        "active": True,
        "score": 85.5
    }

    # Complex hierarchical document
    complex_doc = {
        "company": {
            "name": "TechCorp Inc",
            "employees": [
                {
                    "id": "emp001",
                    "profile": {
                        "name": "Alice Smith",
                        "department": "Engineering",
                        "skills": ["Python", "Machine Learning", "Data Science"],
                        "performance": {
                            "2023": {"rating": "excellent", "projects": 12},
                            "2022": {"rating": "good", "projects": 8}
                        }
                    }
                },
                {
                    "id": "emp002",
                    "profile": {
                        "name": "Bob Johnson",
                        "department": "Sales",
                        "skills": ["Negotiation", "CRM", "Analytics"],
                        "performance": {
                            "2023": {"rating": "good", "projects": 15},
                            "2022": {"rating": "excellent", "projects": 18}
                        }
                    }
                }
            ],
            "metrics": {
                "revenue": 2500000,
                "growth_rate": 0.15,
                "satisfaction_score": 4.2
            }
        }
    }

    # IoT sensor data document
    iot_doc = {
        "sensor_network": {
            "building_a": {
                "temperature_sensors": [
                    {"id": "temp_001", "location": "room_101", "reading": 22.5, "timestamp": "2023-12-13T10:00:00Z"},
                    {"id": "temp_002", "location": "room_102", "reading": 24.1, "timestamp": "2023-12-13T10:00:00Z"}
                ],
                "humidity_sensors": [
                    {"id": "hum_001", "location": "room_101", "reading": 45.2, "timestamp": "2023-12-13T10:00:00Z"},
                    {"id": "hum_002", "location": "room_102", "reading": 48.7, "timestamp": "2023-12-13T10:00:00Z"}
                ]
            },
            "building_b": {
                "motion_sensors": [
                    {"id": "mot_001", "location": "lobby", "status": "active", "last_trigger": "2023-12-13T09:45:00Z"},
                    {"id": "mot_002", "location": "corridor", "status": "inactive", "last_trigger": "2023-12-13T08:30:00Z"}
                ]
            }
        },
        "network_status": {
            "total_devices": 6,
            "online_devices": 6,
            "last_sync": "2023-12-13T10:01:00Z"
        }
    }

    return {
        "simple": simple_doc,
        "complex": complex_doc,
        "iot": iot_doc
    }


def demonstrate_transformation_steps():
    """Demonstrate each transformation step: JSON → DocumentChunk → EmbeddingChunk → Text."""
    print("=== TRANSFORMATION STEPS DEMO ===\n")

    from dynamic_embeddings.core.dynamic_engine import DynamicChunkingEngine
    from dynamic_embeddings.processors.text_converter import ChunkTextConverter

    # Get all sample documents
    documents = create_sample_documents()

    for doc_type, test_doc in documents.items():
        print(f"\n{'='*60}")
        print(f"🚀 PROCESSING {doc_type.upper()} DOCUMENT")
        print(f"{'='*60}")

        print(f"\n🔸 STEP 1: Original {doc_type.title()} JSON Input")
        print("=" * 50)
        print(json.dumps(test_doc, indent=2))

        # Step 1: JSON → DocumentChunk
        print(f"\n🔸 STEP 2: {doc_type.title()} JSON → DocumentChunk")
        print("=" * 50)
        engine = DynamicChunkingEngine(config_name="default")
        document_chunks, decision_metadata = engine.process_document(test_doc)

        strategy_used = decision_metadata.get('strategy_used', 'unknown')
        decision_details = decision_metadata.get('decision_details', {})
        confidence = decision_details.get('confidence', 0.0)

        print(f"Strategy selected: {strategy_used}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Number of chunks created: {len(document_chunks)}")

        for i, chunk in enumerate(document_chunks):
            print(f"\n--- DocumentChunk {i+1} ---")
            print(f"Chunk ID: {chunk.metadata.chunk_id}")
            print(f"Path: {chunk.metadata.source_path}")
            print(f"Level: {chunk.metadata.depth_level}")
            print(f"Content: {json.dumps(chunk.content, indent=2)}")
            print(f"Key count: {chunk.metadata.key_count}")
            print(f"Contains arrays: {chunk.metadata.contains_arrays}")

        # Step 2: DocumentChunk → EmbeddingChunk
        print(f"\n🔸 STEP 3: {doc_type.title()} DocumentChunk → EmbeddingChunk")
        print("=" * 50)
        text_converter = ChunkTextConverter(strategy="contextual_description")

        embedding_chunks = []
        for i, doc_chunk in enumerate(document_chunks):
            # Convert to text
            chunk_metadata = {
                'path': doc_chunk.metadata.source_path,
                'level': doc_chunk.metadata.depth_level
            }
            text = text_converter.convert_chunk_to_text(doc_chunk.content, chunk_metadata)
            semantic_density = text_converter.calculate_semantic_density(doc_chunk.content)

            # Create EmbeddingChunk
            embedding_chunk = EmbeddingChunk(
                text=text,
                chunk_id=doc_chunk.metadata.chunk_id,
                path=doc_chunk.metadata.source_path,
                level=doc_chunk.metadata.depth_level,
                content_type="mixed",
                key_count=len(doc_chunk.content) if isinstance(doc_chunk.content, dict) else 0,
                value_types=["string", "integer", "object", "array"],
                strategy=strategy_used,
                confidence=confidence,
                semantic_density=semantic_density,
                source_file=f"{doc_type}_document.json"
            )
            embedding_chunks.append(embedding_chunk)

            print(f"\n--- EmbeddingChunk {i+1} ---")
            print(f"Chunk ID: {embedding_chunk.chunk_id}")
            print(f"Path: {embedding_chunk.path}")
            print(f"Level: {embedding_chunk.level}")
            print(f"Content Type: {embedding_chunk.content_type}")
            print(f"Key Count: {embedding_chunk.key_count}")
            print(f"Value Types: {embedding_chunk.value_types}")
            print(f"Strategy: {embedding_chunk.strategy}")
            print(f"Confidence: {embedding_chunk.confidence:.2f}")
            print(f"Text Length: {embedding_chunk.text_length}")
            print(f"Semantic Density: {embedding_chunk.semantic_density:.2f}")
            print(f"Source File: {embedding_chunk.source_file}")

        # Step 3: EmbeddingChunk → Final Text
        print(f"\n🔸 STEP 4: {doc_type.title()} EmbeddingChunk → Final Text for Embeddings")
        print("=" * 50)

        for i, embedding_chunk in enumerate(embedding_chunks):
            print(f"\n--- Final Text Output {i+1} ---")
            print(f"Chunk ID: {embedding_chunk.chunk_id}")
            print(f"Text for Embeddings:")
            print(f'"{embedding_chunk.text}"')
            print(f"Character Count: {len(embedding_chunk.text)}")

        # Show different text conversion strategies for first chunk
        print(f"\n🔸 BONUS: Different Text Strategies for {doc_type.title()} (First Chunk)")
        print("=" * 50)

        if document_chunks:
            sample_chunk = document_chunks[0]
            sample_metadata = {
                'path': sample_chunk.metadata.source_path,
                'level': sample_chunk.metadata.depth_level
            }

            strategies = ["contextual_description", "key_value_pairs", "structured_narrative"]

            for strategy in strategies:
                converter = ChunkTextConverter(strategy=strategy)
                text_output = converter.convert_chunk_to_text(sample_chunk.content, sample_metadata)
                print(f"\n{strategy.upper()}:")
                print(f'"{text_output}"')


def demonstrate_basic_processing():
    """Demonstrate basic document processing."""
    print("\n=== Document Chunking Demo: Basic Processing ===\n")

    # Create processor with default configuration
    processor = DocumentProcessor(config_name="default")

    # Get sample documents
    documents = create_sample_documents()

    # Process each document type
    for doc_type, document in documents.items():
        print(f"Processing {doc_type} document...")

        # Process document
        chunks = processor.process_document(document, source_file=f"{doc_type}_sample.json")

        print(f"✅ Generated {len(chunks)} chunks")
        print(f"Strategy used: {chunks[0].strategy if chunks else 'N/A'}")
        print(f"Confidence: {chunks[0].confidence:.2f}" if chunks else "N/A")

        # Show first chunk as example
        if chunks:
            chunk = chunks[0]
            print(f"Sample chunk text: {chunk.text[:150]}...")
            print(f"Chunk metadata: path={chunk.path}, level={chunk.level}")
            print(f"Content type: {chunk.content_type}")
            print(f"Semantic density: {chunk.semantic_density:.2f}")

        print("-" * 50)


def demonstrate_configuration_comparison():
    """Demonstrate different configurations for IoT data."""
    print("\n=== Configuration Comparison for IoT Data ===\n")

    documents = create_sample_documents()
    iot_doc = documents["iot"]

    # Test different configurations
    configs = ["default", "iot", "analytics"]

    for config_name in configs:
        print(f"Testing with '{config_name}' configuration...")

        try:
            processor = DocumentProcessor(config_name=config_name)
            chunks = processor.process_document(iot_doc, source_file="iot_sample.json")

            print(f"✅ Chunks: {len(chunks)}")
            print(f"Strategy: {chunks[0].strategy if chunks else 'N/A'}")
            print(f"Avg semantic density: {sum(c.semantic_density for c in chunks)/len(chunks):.2f}" if chunks else "N/A")

        except Exception as e:
            print(f"❌ Failed with {config_name}: {e}")

        print("-" * 30)


def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n=== Batch Processing Demo ===\n")

    # Create temporary JSON files
    temp_dir = Path("temp_examples")
    temp_dir.mkdir(exist_ok=True)

    documents = create_sample_documents()
    file_paths = []

    # Save documents to files
    for doc_type, document in documents.items():
        file_path = temp_dir / f"{doc_type}_sample.json"
        with open(file_path, 'w') as f:
            json.dump(document, f, indent=2)
        file_paths.append(file_path)

    # Batch process
    processor = DocumentProcessor(config_name="default")
    results = processor.process_batch(file_paths)

    print(f"✅ Batch processed {len(file_paths)} files")

    total_chunks = 0
    for file_path, chunks in results.items():
        filename = Path(file_path).name
        print(f"{filename}: {len(chunks)} chunks")
        total_chunks += len(chunks)

    print(f"Total chunks across all files: {total_chunks}")

    # Cleanup
    for file_path in file_paths:
        file_path.unlink()
    temp_dir.rmdir()


def demonstrate_export_capabilities():
    """Demonstrate chunk export in different formats."""
    print("\n=== Export Capabilities Demo ===\n")

    processor = DocumentProcessor(config_name="default")
    documents = create_sample_documents()

    # Process complex document
    chunks = processor.process_document(documents["complex"], source_file="complex_sample.json")

    if chunks:
        # Export in different formats
        formats = ["json", "jsonl"]

        for format_type in formats:
            output_file = f"sample_chunks.{format_type}"
            processor.export_chunks(chunks, output_file, format=format_type)
            print(f"✅ Exported {len(chunks)} chunks to {output_file}")

            # Clean up
            Path(output_file).unlink()


def demonstrate_processing_statistics():
    """Demonstrate processing statistics."""
    print("\n=== Processing Statistics Demo ===\n")

    processor = DocumentProcessor(config_name="default")
    documents = create_sample_documents()

    all_chunks = []

    # Process all documents
    for doc_type, document in documents.items():
        chunks = processor.process_document(document, source_file=f"{doc_type}_sample.json")
        all_chunks.extend(chunks)

    # Get statistics
    stats = processor.get_processing_stats(all_chunks)

    print(f"Total chunks processed: {stats['total_chunks']}")
    print(f"Average text length: {stats['avg_text_length']:.1f} characters")
    print(f"Average semantic density: {stats['avg_semantic_density']:.2f}")

    print("\nStrategy distribution:")
    for strategy, count in stats['strategy_distribution'].items():
        print(f"  {strategy}: {count} chunks")

    print("\nContent type distribution:")
    for content_type, count in stats['content_type_distribution'].items():
        print(f"  {content_type}: {count} chunks")

    print("\nQuality metrics:")
    quality = stats['quality_metrics']
    print(f"  Text length range: {quality['min_text_length']}-{quality['max_text_length']} chars")
    print(f"  Semantic density range: {quality['min_semantic_density']:.2f}-{quality['max_semantic_density']:.2f}")


def demonstrate_text_conversion_strategies():
    """Demonstrate different text conversion strategies."""
    print("\n=== Text Conversion Strategies Demo ===\n")

    from dynamic_embeddings import ChunkTextConverter

    # Sample chunk data
    sample_chunk = {
        "user_profile": {
            "name": "Alice Johnson",
            "role": "Data Scientist"
        },
        "performance_score": 92.5,
        "active": True
    }

    sample_metadata = {
        "path": "employees.0.details",
        "level": 2
    }

    # Test different strategies
    strategies = ["contextual_description", "key_value_pairs", "structured_narrative"]

    for strategy in strategies:
        converter = ChunkTextConverter(strategy=strategy)
        text = converter.convert_chunk_to_text(sample_chunk, sample_metadata)

        print(f"Strategy: {strategy}")
        print(f"Output: {text}")
        print("-" * 40)


def main():
    """Run the complete document chunking demonstration."""
    setup_logging()

    print("🚀 Dynamic JSON Embeddings - Document Chunking Pipeline Demo")
    print("=" * 65)

    try:
        # Run transformation steps demo first
        demonstrate_transformation_steps()

        # Run all demonstrations
        demonstrate_basic_processing()
        demonstrate_configuration_comparison()
        demonstrate_batch_processing()
        demonstrate_export_capabilities()
        demonstrate_processing_statistics()
        demonstrate_text_conversion_strategies()

        print("\n✅ Document chunking demo completed successfully!")
        print("\n📋 What was demonstrated:")
        print("- Automatic chunking strategy selection")
        print("- JSON-to-text conversion for embeddings")
        print("- Quality validation and filtering")
        print("- Batch processing capabilities")
        print("- Export in multiple formats")
        print("- Comprehensive statistics")
        print("- Different text conversion strategies")

        print("\n🔄 Next Steps:")
        print("- Next: Implement vector embedding generation with OpenAI")
        print("- Next: Vector storage with PGVector")
        print("- Next: Retrieval and similarity search system")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()