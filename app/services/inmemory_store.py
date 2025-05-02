"""
In-memory vector store service for managing document embeddings.

This module provides functionality to store, search, and retrieve documents
using vector embeddings in memory without requiring an external database.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from app.services.vector_store import SearchHit
from app.services.embeddings import embedding_service


class InMemoryStore:
    """Service for in-memory vector storage and search."""
    
    _instance = None
    _vectors = None
    _chunks = None
    _loaded = False
    
    def __new__(cls):
        """Ensure only one instance of the vector store exists."""
        if cls._instance is None:
            cls._instance = super(InMemoryStore, cls).__new__(cls)
        return cls._instance
    
    def load_data(self, jsonl_path: str) -> bool:
        """
        Load data from a JSONL file and create embeddings.
        
        Args:
            jsonl_path: Path to the JSONL file
            
        Returns:
            Success status
        """
        if self._loaded:
            print("Data already loaded in memory.")
            return True
            
        try:
            chunks = []
            
            # Load chunks from JSONL
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            chunks.append(chunk)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line: {e}")
                            continue
            
            if not chunks:
                print(f"No chunks found in {jsonl_path}")
                return False
                
            print(f"Loaded {len(chunks)} chunks from {jsonl_path}")
            
            # Extract texts for embedding
            texts = [chunk["text"] for chunk in chunks]
            
            # Generate embeddings
            print("Generating embeddings...")
            embeddings = embedding_service.encode(texts)
            print(f"Generated {len(embeddings)} embeddings")
            
            # Store in memory
            self._chunks = chunks
            self._vectors = np.array(embeddings)
            self._loaded = True
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def search(
        self, 
        query: str, 
        k: int = 8, 
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchHit]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter_conditions: Optional filter to apply
            
        Returns:
            List of search hits
        """
        if not self._loaded or self._vectors is None or self._chunks is None:
            # Try to load from default location
            default_path = Path("data/output_chunks.jsonl")
            if default_path.exists():
                success = self.load_data(str(default_path))
                if not success:
                    print("Failed to load data from default location")
                    return []
            else:
                print("No data loaded and no default data found")
                return []
        
        # Encode query
        query_embedding = embedding_service.encode_query(query)
        
        # Calculate cosine similarities
        similarities = np.dot(self._vectors, query_embedding) / (
            np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get indices of top k results
        if filter_conditions:
            # Apply filters
            filtered_indices = []
            for i, chunk in enumerate(self._chunks):
                match = True
                for key, value in filter_conditions.items():
                    if key in chunk and chunk[key] != value:
                        match = False
                        break
                if match:
                    filtered_indices.append(i)
            
            if not filtered_indices:
                return []
                
            # Get top k from filtered indices
            filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
            top_indices = sorted(filtered_similarities, key=lambda x: x[1], reverse=True)[:k]
        else:
            # Get top k from all indices
            top_indices = [(i, similarities[i]) for i in np.argsort(-similarities)[:k]]
        
        # Convert to SearchHit format
        hits = []
        for idx, score in top_indices:
            chunk = self._chunks[idx]
            hits.append(SearchHit(
                id=str(idx),
                score=float(score),
                text=chunk["text"],
                payload=chunk
            ))
        
        return hits


# Export singleton instance
inmemory_store = InMemoryStore() 