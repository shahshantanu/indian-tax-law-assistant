#!/usr/bin/env python3
"""
Unit tests for the legal assistant application.

This script contains pytest tests for various components of the legal assistant.
"""

import os
import sys
import re
import pytest
from typing import List, Dict, Any

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import settings
from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store
from app.agents.retriever_agent import retriever_agent
from app.agents.drafter_agent import drafter_agent


def test_embedding_service():
    """Test that the embedding service produces correct embeddings."""
    # Test query embedding
    query = "What are the deductions under section 80C?"
    query_embedding = embedding_service.encode_query(query)
    
    # Verify shape and normalization
    assert query_embedding.shape[0] > 0, "Embedding should have non-zero dimension"
    assert abs(1.0 - (query_embedding**2).sum()) < 1e-6, "Embedding should be L2-normalized"


def test_vector_store_connection():
    """Test that vector store connection works."""
    # Verify collection exists
    collections = vector_store._client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    # If collection doesn't exist, this is expected to fail in some environments
    # So we just check that the client is working
    assert vector_store._client is not None, "Qdrant client should be initialized"


def test_retriever_agent():
    """Test that retriever agent returns results."""
    # Only run if we know there's data
    collections = vector_store._client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if settings.QDRANT_COLLECTION not in collection_names:
        pytest.skip(f"Collection {settings.QDRANT_COLLECTION} not found in Qdrant")
    
    # Get collection info
    collection_info = vector_store._client.get_collection(settings.QDRANT_COLLECTION)
    if collection_info.vectors_count == 0:
        pytest.skip(f"Collection {settings.QDRANT_COLLECTION} is empty")
    
    # Test retrieval
    query = "What are deductions under section 80C?"
    chunks = retriever_agent.retrieve(query, k=3)
    
    # Verify we got some results
    assert len(chunks) > 0, "Should retrieve at least one chunk"
    
    # Verify each chunk has necessary fields
    for chunk in chunks:
        assert chunk.id, "Chunk should have an ID"
        assert chunk.text, "Chunk should have text content"
        assert chunk.citation, "Chunk should have citation"
        assert chunk.score >= 0, "Chunk should have a positive score"


def test_deduction_query_response():
    """
    Test a full query about deductions to verify citation correctness.
    
    This test ensures that:
    1. We get relevant chunks for a query about section 80C
    2. The LLM properly cites these sources
    3. Citations in the answer match chunks that mention section 80C
    """
    # Check if the collection exists and has data
    collections = vector_store._client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if settings.QDRANT_COLLECTION not in collection_names:
        pytest.skip(f"Collection {settings.QDRANT_COLLECTION} not found in Qdrant")
    
    # Get collection info
    collection_info = vector_store._client.get_collection(settings.QDRANT_COLLECTION)
    if collection_info.vectors_count == 0:
        pytest.skip(f"Collection {settings.QDRANT_COLLECTION} is empty")
    
    # Query about section 80C deductions
    query = "List deductions available under section 80C"
    
    # Retrieve context
    chunks = retriever_agent.retrieve(query, k=5)
    if not chunks:
        pytest.skip("No chunks retrieved, cannot test full query cycle")
    
    # Check if any chunks contain section 80C
    section_80c_chunks = [
        chunk for chunk in chunks 
        if "80C" in chunk.text or "80C" in chunk.metadata.get("section_number", "")
    ]
    
    if not section_80c_chunks:
        pytest.skip("No chunks about section 80C retrieved, cannot test specific citation")
    
    # Generate response
    response = drafter_agent.draft_response(query, chunks)
    
    # Requirements to verify
    has_citations = bool(response["citations"])
    
    # Extract citation numbers from analysis and answer
    citation_refs = set()
    for text in [response["analysis"], response["answer"]]:
        refs = re.findall(r'\[(\d+)\]', text)
        citation_refs.update(int(ref) for ref in refs if ref.isdigit())
    
    has_citation_references = bool(citation_refs)
    
    # Verify that all citation references have corresponding citations
    all_refs_have_citations = all(
        any(f"[{ref}]" in citation for citation in response["citations"])
        for ref in citation_refs
    )
    
    # Check that at least one citation references a chunk containing 80C
    section_80c_citation_exists = False
    for citation in response["citations"]:
        if "80C" in citation:
            section_80c_citation_exists = True
            break
    
    # Assertions
    assert has_citations, "Response should have citations"
    assert has_citation_references, "Response should reference citations in the text"
    assert all_refs_have_citations, "All citation references should have corresponding citations"
    assert section_80c_citation_exists, "At least one citation should reference Section 80C"


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__]) 