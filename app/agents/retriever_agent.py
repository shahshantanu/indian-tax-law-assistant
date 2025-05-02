"""
Retriever agent module for finding relevant content from the vector store.

This module provides functionality to search for and retrieve relevant
context chunks in response to user queries.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os

# Import Milvus store if available
try:
    from app.services.milvus_store import milvus_store, SearchHit
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    # Define SearchHit for type hints if not available
    class SearchHit:
        id: str
        score: float
        text: str
        payload: Dict[str, Any]

# Import in-memory store
try:
    from app.services.inmemory_store import inmemory_store
    INMEMORY_AVAILABLE = True
except ImportError:
    INMEMORY_AVAILABLE = False

@dataclass
class ContextChunk:
    """Class representing a processed context chunk with citation info."""
    
    id: str
    text: str
    citation: str
    metadata: Dict[str, Any]
    score: float
    
    def __str__(self) -> str:
        """Format the chunk for display, including citation tag."""
        return f"{self.text}\n[{self.citation}]"


class RetrieverAgent:
    """Agent for retrieving relevant document chunks."""
    
    def __init__(self):
        """Initialize the retriever agent."""
        # Determine which vector store to use
        self.use_milvus = os.getenv("USE_MILVUS", "true").lower() == "true"
        self.use_inmemory = os.getenv("USE_INMEMORY", "false").lower() == "true"
        
        # Initialize the data path for in-memory store
        self.data_path = os.getenv("DATA_PATH", "data/output_chunks.jsonl")
        
        if self.use_milvus and not MILVUS_AVAILABLE:
            print("Warning: USE_MILVUS is set but Milvus is not available. Falling back to in-memory search.")
            self.use_milvus = False
            
        if self.use_inmemory:
            print("Using in-memory vector search")
        else:
            print(f"Retriever initialized with Milvus vector store")
            if INMEMORY_AVAILABLE:
                print("In-memory search available as fallback")
    
    def retrieve(
        self, 
        query: str, 
        k: int = 8,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[ContextChunk]:
        """
        Retrieve relevant document chunks for a given query.
        
        Args:
            query: User query string
            k: Number of results to return
            filter_conditions: Optional filter to apply
            
        Returns:
            List of context chunks with citation information
        """
        # Try Milvus first if enabled
        if self.use_milvus and MILVUS_AVAILABLE:
            try:
                hits = milvus_store.search(query, k, filter_conditions)
                if hits:
                    return self._process_hits(hits)
                print("Milvus search returned no results, trying fallback...")
            except Exception as e:
                print(f"Error with Milvus search: {e}, trying fallback...")
        
        # Fall back to in-memory search if available
        if INMEMORY_AVAILABLE:
            try:
                # Ensure data is loaded
                if not inmemory_store._loaded:
                    inmemory_store.load_data(self.data_path)
                
                hits = inmemory_store.search(query, k, filter_conditions)
                return self._process_hits(hits)
            except Exception as e:
                print(f"Error with in-memory search: {e}")
                return []
        
        # If all options failed
        print("All vector store options failed")
        return []
    
    def _process_hits(self, hits: List[SearchHit]) -> List[ContextChunk]:
        """
        Process search hits into context chunks with citation information.
        
        Args:
            hits: List of search hits
            
        Returns:
            List of context chunks
        """
        chunks = []
        for i, hit in enumerate(hits):
            # Generate citation reference number
            citation_number = i + 1
            
            # Extract section/heading for citation
            section = hit.payload.get("section_number", "")
            heading = hit.payload.get("heading", "")
            
            # Format the citation based on available metadata
            if section:
                citation = f"{citation_number}: Section {section}"
                if heading and heading != section:
                    citation += f" ({heading})"
            else:
                citation = f"{citation_number}: {heading or 'Reference'}"
            
            # Add jurisdiction if available and not in citation yet
            jurisdiction = hit.payload.get("jurisdiction", "")
            if jurisdiction and jurisdiction not in citation:
                citation += f", {jurisdiction}"
            
            # Create context chunk
            chunk = ContextChunk(
                id=hit.id,
                text=hit.text,
                citation=citation,
                metadata=hit.payload,
                score=hit.score
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def format_context(self, chunks: List[ContextChunk]) -> str:
        """
        Format a list of context chunks into a string suitable for LLM input.
        
        Args:
            chunks: List of context chunks to format
            
        Returns:
            Formatted context string with citation markers
        """
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Format with citation tag [n]
            citation_number = i + 1
            formatted_chunk = f"[{citation_number}] {chunk.text}"
            formatted_chunks.append(formatted_chunk)
        
        return "\n\n".join(formatted_chunks)
    
    def get_citations_map(self, chunks: List[ContextChunk]) -> Dict[int, str]:
        """
        Create a mapping from citation numbers to full citation strings.
        
        Args:
            chunks: List of context chunks
            
        Returns:
            Dictionary mapping citation numbers to citation strings
        """
        citations = {}
        
        for i, chunk in enumerate(chunks):
            citation_number = i + 1
            
            # Format the full citation
            section = chunk.metadata.get("section_number", "")
            heading = chunk.metadata.get("heading", "")
            jurisdiction = chunk.metadata.get("jurisdiction", "")
            
            if section:
                full_citation = f"Section {section}"
                if heading and heading != section:
                    full_citation += f" ({heading})"
            else:
                full_citation = heading or "Reference"
            
            if jurisdiction and jurisdiction not in full_citation:
                full_citation += f", {jurisdiction}"
            
            citations[citation_number] = full_citation
        
        return citations


# Export agent instance
retriever_agent = RetrieverAgent() 