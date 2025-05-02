#!/usr/bin/env python3
"""
Script to strictly preprocess and import data into Milvus with aggressive field truncation.

This script ensures all fields are well below Milvus limits by:
1. Using very conservative truncation limits
2. Measuring byte length after encoding, not just character count
3. Using a simple schema with minimal metadata
"""

import json
import os
import sys
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
from pymilvus import connections, utility, Collection, DataType, FieldSchema, CollectionSchema

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import settings
from app.services.embeddings import embedding_service

# Very conservative field limits - well below Milvus limits
MAX_SECTION_LENGTH = 80  # Instead of 100
MAX_HEADING_LENGTH = 950  # Instead of 1000

def strictly_truncate(text, max_length):
    """
    Truncate text to ensure byte length is below limit.
    Measures actual UTF-8 encoded length.
    """
    if not text:
        return ""
        
    # First do a character-based truncation
    text = text[:max_length]
    
    # Then check byte length and truncate further if needed
    encoded = text.encode('utf-8')
    while len(encoded) > max_length:
        text = text[:-1]  # Remove one character
        encoded = text.encode('utf-8')
        
    return text

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load chunks from a JSONL file."""
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
    
    print(f"Loaded {len(chunks)} chunks from {file_path}")
    return chunks

def connect_to_milvus():
    """Connect to Milvus using credentials from settings."""
    print(f"Connecting to Milvus at {settings.MILVUS_URI}")
    
    try:
        # Try username/password if available
        if hasattr(settings, 'MILVUS_USERNAME') and settings.MILVUS_USERNAME and hasattr(settings, 'MILVUS_PASSWORD') and settings.MILVUS_PASSWORD:
            print("Using username/password authentication")
            connections.connect(
                alias="default",
                uri=settings.MILVUS_URI,
                user=settings.MILVUS_USERNAME,
                password=settings.MILVUS_PASSWORD
            )
        else:
            print("Using API key authentication")
            connections.connect(
                alias="default",
                uri=settings.MILVUS_URI,
                token=settings.MILVUS_API_KEY
            )
        
        print("Successfully connected to Milvus")
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False

def create_minimal_collection(collection_name: str):
    """Create a minimal collection schema to reduce issues."""
    if utility.has_collection(collection_name):
        print(f"Dropping existing collection: {collection_name}")
        utility.drop_collection(collection_name)
    
    print(f"Creating collection: {collection_name}")
    
    # Get vector dimension from embedding model
    dummy_vector = embedding_service.encode_query("test query")
    vector_size = len(dummy_vector)
    
    # Define a minimal collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="section_number", dtype=DataType.VARCHAR, max_length=MAX_SECTION_LENGTH),
        FieldSchema(name="heading", dtype=DataType.VARCHAR, max_length=MAX_HEADING_LENGTH),
        FieldSchema(name="doctype", dtype=DataType.VARCHAR, max_length=50),
    ]
    
    schema = CollectionSchema(fields, "Minimal legal document chunks collection")
    collection = Collection(name=collection_name, schema=schema)
    
    # Create index for vector field
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    
    collection.create_index(field_name="vector", index_params=index_params)
    print(f"Created collection and index: {collection_name}")
    return collection

def import_with_strict_truncation(jsonl_path: str, batch_size: int = 1):
    """Import data with strict field truncation."""
    # Connect to Milvus
    if not connect_to_milvus():
        print("Failed to connect to Milvus. Cannot proceed with import.")
        return 0
    
    # Create collection
    collection_name = settings.MILVUS_COLLECTION
    collection = create_minimal_collection(collection_name)
    collection.load()
    
    # Load chunks
    chunks = load_jsonl(jsonl_path)
    if not chunks:
        print("No chunks found in JSONL file. Exiting.")
        return 0
    
    # Process in batches of 1 to avoid batch issues
    success_count = 0
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing chunks"):
        batch = chunks[i:i + batch_size]
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in batch]
        
        try:
            # Generate embeddings
            embeddings = embedding_service.encode(texts, batch_size=batch_size)
            
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                # Create ID
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{chunk.get('section_number', '')}-{i+j}"))
                
                # Strictly truncate fields
                section_number = strictly_truncate(chunk.get("section_number", ""), MAX_SECTION_LENGTH)
                heading = strictly_truncate(chunk.get("heading", ""), MAX_HEADING_LENGTH)
                
                # Insert as a single entity to avoid batch issues
                try:
                    single_entity = [
                        [chunk_id],
                        [embedding.tolist()],
                        [chunk.get("text", "")],
                        [section_number],
                        [heading],
                        [chunk.get("doctype", "tax_law")],
                    ]
                    
                    collection.insert(single_entity)
                    success_count += 1
                    
                    if success_count % 10 == 0:
                        print(f"Successfully imported {success_count}/{len(chunks)} chunks")
                except Exception as e:
                    print(f"Error inserting chunk {i+j}: {e}")
                    # Continue with next chunk
        except Exception as e:
            print(f"Error processing batch {i // batch_size}: {e}")
    
    # Create index if it doesn't exist yet
    try:
        index_info = collection.index()
        if not index_info:
            print("Creating index on vector field...")
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
            collection.create_index(field_name="vector", index_params=index_params)
    except Exception as e:
        print(f"Error checking/creating index: {e}")
    
    # Flush and disconnect
    try:
        print("Flushing collection...")
        collection.flush()
    except Exception as e:
        print(f"Error flushing: {e}")
    
    # Disconnect from Milvus
    connections.disconnect("default")
    
    return success_count

def main():
    """Run the import script."""
    parser = argparse.ArgumentParser(description='Import tax law chunks with strict field truncation')
    parser.add_argument('--jsonl', type=str, default="data/output_chunks.jsonl",
                       help='Path to the JSONL file containing tax law chunks')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inserting (default: 1)')
    
    args = parser.parse_args()
    
    print(f"Starting import from {args.jsonl} to Milvus at {settings.MILVUS_URI}")
    print(f"Using collection: {settings.MILVUS_COLLECTION}")
    print(f"Using embedding model: {settings.EMBED_MODEL}")
    print(f"Using very strict field truncation")
    
    # Import with strict truncation
    success_count = import_with_strict_truncation(args.jsonl, args.batch_size)
    
    print(f"Import complete. Successfully imported {success_count}/{len(load_jsonl(args.jsonl))} chunks.")
    
    if success_count > 0:
        print("\nNext steps:")
        print("1. Verify the system: python scripts/verify_system.py")
        print("2. Start the API server: python scripts/run_api.py")

if __name__ == "__main__":
    main() 