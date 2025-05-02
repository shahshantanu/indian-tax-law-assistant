#!/usr/bin/env python3
"""
Script to import JSONL chunks into Milvus with BGE embeddings.

This script reads the existing JSONL file of tax law chunks, computes BGE
embeddings, and upserts them into the Milvus vector database.
"""

import os
import sys
import json
import uuid
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
from pymilvus import connections, utility, Collection, DataType, FieldSchema, CollectionSchema

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import settings
from app.services.embeddings import embedding_service


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


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load chunks from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of chunk objects
    """
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


def create_collection(collection_name: str):
    """Create the collection if it doesn't exist."""
    if utility.has_collection(collection_name):
        print(f"Collection {collection_name} already exists.")
        return Collection(name=collection_name)
    
    print(f"Creating collection: {collection_name}")
    
    # Get vector dimension from embedding model
    dummy_vector = embedding_service.encode_query("test query")
    vector_size = len(dummy_vector)
    
    # Define collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="section_number", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="heading", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="jurisdiction", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="doctype", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="ocr", dtype=DataType.BOOL),
    ]
    
    schema = CollectionSchema(fields, "Legal document chunks collection")
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


def embed_and_upsert(chunks: List[Dict[str, Any]], batch_size: int = 32) -> int:
    """
    Compute BGE embeddings for chunks and upsert them into Milvus.
    
    Args:
        chunks: List of chunk objects
        batch_size: Batch size for embedding and upserting
        
    Returns:
        Number of successfully upserted chunks
    """
    # Connect to Milvus
    if not connect_to_milvus():
        print("Failed to connect to Milvus. Cannot proceed with import.")
        return 0
    
    collection_name = settings.MILVUS_COLLECTION
    
    # Create or get collection
    collection = create_collection(collection_name)
    collection.load()
    
    total_chunks = len(chunks)
    success_count = 0
    
    # Process in batches to save memory
    for i in tqdm(range(0, total_chunks, batch_size), desc="Embedding batches"):
        batch = chunks[i:i + batch_size]
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in batch]
        
        # Generate embeddings
        try:
            embeddings = embedding_service.encode(texts, batch_size=batch_size)
            
            # Prepare data for Milvus
            ids = []
            vectors = []
            texts_list = []
            section_numbers = []
            headings = []
            jurisdictions = []
            doctypes = []
            ocrs = []
            
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                # Create a stable ID
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{chunk.get('section_number', '')}-{i+j}"))
                
                # Add embedding model info to payload
                chunk["embed_model"] = settings.EMBED_MODEL
                
                # Add doctype field for filtering if not present
                if "doctype" not in chunk:
                    chunk["doctype"] = "tax_law"
                
                # Truncate section_number to avoid exceeding field length limit
                section_number = chunk.get("section_number", "")
                if len(section_number) > 100:
                    section_number = section_number[:97] + "..."
                    print(f"Truncated section_number for chunk {j}: {len(chunk.get('section_number', ''))} -> 100 chars")
                
                # Truncate heading to avoid exceeding field length limit (1000 chars)
                heading = chunk.get("heading", "")
                if heading and len(heading) > 987:
                    heading = heading[:987] + "..."
                    print(f"Truncated heading for chunk {j}: {len(chunk.get('heading', ''))} -> 987 chars")
                
                ids.append(chunk_id)
                vectors.append(embedding.tolist())
                texts_list.append(chunk.get("text", ""))
                section_numbers.append(section_number)  # Using truncated section_number
                headings.append(heading)  # Using truncated heading
                jurisdictions.append(chunk.get("jurisdiction", "India"))
                doctypes.append(chunk.get("doctype", "tax_law"))
                ocrs.append(chunk.get("ocr", False))
            
            # Insert data into Milvus
            try:
                entities = [
                    ids,          # id
                    vectors,      # vector
                    texts_list,   # text
                    section_numbers,  # section_number
                    headings,     # heading
                    jurisdictions,  # jurisdiction
                    doctypes,     # doctype
                    ocrs          # ocr
                ]
                
                collection.insert(entities)
                success_count += len(batch)
                print(f"Inserted batch {i//batch_size+1}/{(total_chunks+batch_size-1)//batch_size}")
            except Exception as e:
                print(f"Error inserting batch to Milvus: {e}")
                
                # Try inserting one by one as fallback
                try:
                    print("Trying one-by-one insert as fallback...")
                    one_by_one_success = 0
                    for idx in range(len(ids)):
                        try:
                            single_entity = [
                                [ids[idx]],
                                [vectors[idx]],
                                [texts_list[idx]],
                                [section_numbers[idx]],  # Already truncated
                                [headings[idx]],  # Already truncated
                                [jurisdictions[idx]],
                                [doctypes[idx]],
                                [ocrs[idx]]
                            ]
                            collection.insert(single_entity)
                            one_by_one_success += 1
                        except Exception as e3:
                            print(f"Failed to insert entity {idx}: {e3}")
                            continue
                    
                    if one_by_one_success > 0:
                        success_count += one_by_one_success
                        print(f"Successfully inserted {one_by_one_success}/{len(ids)} entities using one-by-one method")
                    else:
                        print("One-by-one insert failed for all entities")
                except Exception as e2:
                    print(f"One-by-one insert method failed: {e2}")
            
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


def drop_collection_if_exists(collection_name: str) -> bool:
    """Drop the collection if it exists."""
    if utility.has_collection(collection_name):
        print(f"Dropping existing collection: {collection_name}")
        try:
            utility.drop_collection(collection_name)
            print(f"Successfully dropped collection: {collection_name}")
            return True
        except Exception as e:
            print(f"Error dropping collection: {e}")
            return False
    return True


def main():
    """Run the import script."""
    parser = argparse.ArgumentParser(description='Import tax law chunks to Milvus with BGE embeddings')
    parser.add_argument('--jsonl', type=str, default="data/output_chunks.jsonl",
                       help='Path to the JSONL file containing tax law chunks')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for embedding and upserting')
    parser.add_argument('--drop-collection', action='store_true',
                       help='Drop the collection if it exists before importing')
    
    args = parser.parse_args()
    
    print(f"Starting import from JSONL {args.jsonl} to Milvus at {settings.MILVUS_URI}")
    print(f"Using collection: {settings.MILVUS_COLLECTION}")
    print(f"Using embedding model: {settings.EMBED_MODEL}")
    
    # Connect to Milvus
    if not connect_to_milvus():
        print("Failed to connect to Milvus. Cannot proceed with import.")
        return
    
    # Drop collection if requested
    if args.drop_collection:
        if not drop_collection_if_exists(settings.MILVUS_COLLECTION):
            print("Failed to drop collection. Aborting import.")
            return
    
    # Load chunks
    chunks = load_jsonl(args.jsonl)
    if not chunks:
        print("No chunks found in JSONL file. Exiting.")
        return
    
    # Embed and upsert
    success_count = embed_and_upsert(chunks, args.batch_size)
    
    print(f"Import complete. Successfully upserted {success_count}/{len(chunks)} chunks.")


if __name__ == "__main__":
    main() 