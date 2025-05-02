"""
Embeddings service module for generating embeddings using sentence transformers.

This module provides a singleton loading implementation for SentenceTransformer models,
optimized for the BGE embedding model.
"""

import os
import numpy as np
from typing import List, Optional, Union
from app import settings

# Check if embeddings are disabled (for low-memory environments)
EMBEDDINGS_DISABLED = os.getenv("DISABLE_EMBEDDINGS", "false").lower() == "true"

class EmbeddingService:
    """Singleton service for generating embeddings."""
    
    _instance = None
    _model = None
    _dummy_dim = 1024  # Standard dimension for BGE models

    def __new__(cls):
        """Ensure only one instance of the embedding service exists."""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the embedding model."""
        # Skip model loading if embeddings are disabled
        if EMBEDDINGS_DISABLED:
            print("Embeddings are disabled. Using dummy embeddings for low-memory mode.")
            return
            
        print(f"Loading embedding model: {settings.EMBED_MODEL}")
        try:
            # Import here to allow environment to run without sentence_transformers
            from sentence_transformers import SentenceTransformer
            
            # First try with CUDA if available
            try:
                import torch
                if torch.cuda.is_available():
                    self._model = SentenceTransformer(settings.EMBED_MODEL, device="cuda")
                    print(f"Embedding model loaded on GPU. Output dimension: {self._model.get_sentence_embedding_dimension()}")
                    self._dummy_dim = self._model.get_sentence_embedding_dimension()
                else:
                    raise RuntimeError("CUDA not available")
            except (ImportError, RuntimeError):
                # Fallback to CPU
                self._model = SentenceTransformer(settings.EMBED_MODEL, device="cpu")
                print(f"Embedding model loaded on CPU. Output dimension: {self._model.get_sentence_embedding_dimension()}")
                self._dummy_dim = self._model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            # Last resort fallback with minimal parameters
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(settings.EMBED_MODEL)
                print(f"Fallback successful with default parameters. Output dimension: {self._model.get_sentence_embedding_dimension()}")
                self._dummy_dim = self._model.get_sentence_embedding_dimension()
            except Exception as e2:
                print(f"Critical error loading embedding model: {e2}")
                print("Using dummy embeddings instead")

    def _get_dummy_embedding(self, count: int = 1) -> np.ndarray:
        """Generate a consistent dummy embedding for testing without model."""
        if count == 1:
            # Return a single normalized vector
            dummy = np.ones(self._dummy_dim, dtype=np.float32)
            return dummy / np.linalg.norm(dummy)
        else:
            # Return multiple normalized vectors
            dummies = np.ones((count, self._dummy_dim), dtype=np.float32)
            # Add some variance to avoid all vectors being identical
            for i in range(count):
                dummies[i] = dummies[i] * (0.9 + 0.2 * (i / count))
            # Normalize each vector
            for i in range(count):
                dummies[i] = dummies[i] / np.linalg.norm(dummies[i])
            return dummies

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of L2-normalized embeddings
        """
        if not texts:
            return np.array([])
            
        if EMBEDDINGS_DISABLED or self._model is None:
            return self._get_dummy_embedding(len(texts))
            
        # BGE models need specific pre-processing for asymmetric retrieval
        processed_texts = [f"Represent this sentence for retrieval: {text}" for text in texts]
        
        # Generate embeddings, normalize to unit length
        embeddings = self._model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        return embeddings.astype(np.float32)
        
    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query with appropriate preprocessing.
        
        Args:
            query: Query text string
            
        Returns:
            Array of L2-normalized embedding (single vector)
        """
        if EMBEDDINGS_DISABLED or self._model is None:
            return self._get_dummy_embedding()
            
        # BGE models need specific pre-processing for asymmetric search
        processed_query = f"Represent this sentence for searching relevant passages: {query}"
        
        embedding = self._model.encode(
            processed_query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        return embedding.astype(np.float32)

# Export singleton instance
embedding_service = EmbeddingService() 