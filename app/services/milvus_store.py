"""
Vector store service module for managing document embeddings in Milvus.

This module provides functionality to store, search, and retrieve documents
using vector embeddings with Milvus vector database.
"""

import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pymilvus import (
    connections, 
    utility,
    Collection,
    FieldSchema, 
    CollectionSchema, 
    DataType
)
from app import settings
from app.services.embeddings import embedding_service


class SearchHit:
    """Class representing a single search result hit."""
    
    def __init__(
        self, 
        id: str, 
        score: float, 
        text: str,
        payload: Dict[str, Any],
    ):
        """Initialize a search hit."""
        self.id = id
        self.score = score
        self.text = text
        self.payload = payload
        
    def __repr__(self) -> str:
        """String representation of a search hit."""
        return f"SearchHit(id={self.id}, score={self.score:.4f}, payload={self.payload})"


class MilvusStore:
    """Service for interacting with Milvus vector database."""
    
    _instance = None
    _collection = None
    
    def __new__(cls):
        """Ensure only one instance of the vector store exists."""
        if cls._instance is None:
            cls._instance = super(MilvusStore, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the Milvus client and ensure collection exists."""
        try:
            # Connect to Milvus
            print(f"Connecting to Milvus cloud at {settings.MILVUS_URI}")
            
            # Determine which authentication method to use
            if hasattr(settings, 'MILVUS_USERNAME') and settings.MILVUS_USERNAME and hasattr(settings, 'MILVUS_PASSWORD') and settings.MILVUS_PASSWORD:
                print(f"Using username/password authentication for Milvus")
                connections.connect(
                    alias="default", 
                    uri=settings.MILVUS_URI,
                    user=settings.MILVUS_USERNAME,
                    password=settings.MILVUS_PASSWORD
                )
            elif hasattr(settings, 'MILVUS_API_KEY') and settings.MILVUS_API_KEY:
                print(f"Using API key authentication for Milvus")
                connections.connect(
                    alias="default", 
                    uri=settings.MILVUS_URI,
                    token=settings.MILVUS_API_KEY
                )
            else:
                print(f"No authentication credentials provided for Milvus")
                connections.connect(
                    alias="default", 
                    uri=settings.MILVUS_URI
                )
            
            # Check if collection exists
            collection_name = settings.MILVUS_COLLECTION
            
            if utility.has_collection(collection_name):
                print(f"Using existing collection: {collection_name}")
                self._collection = Collection(name=collection_name)
                self._collection.load()
            else:
                print(f"Creating collection: {collection_name}")
                # Get vector dimension from embedding model
                dummy_vector = embedding_service.encode_query("test query")
                vector_size = len(dummy_vector)
                
                # Define collection schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="section_number", dtype=DataType.VARCHAR, max_length=80),
                    FieldSchema(name="heading", dtype=DataType.VARCHAR, max_length=950),
                    FieldSchema(name="doctype", dtype=DataType.VARCHAR, max_length=50),
                ]
                
                schema = CollectionSchema(fields, "Legal document chunks collection")
                self._collection = Collection(name=collection_name, schema=schema)
                
                # Create index for vector field
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64}
                }
                self._collection.create_index(field_name="vector", index_params=index_params)
                self._collection.load()
                
        except Exception as e:
            print(f"Error initializing Milvus: {str(e)}")
            self._collection = None
            raise
    
    def upsert(
        self, 
        id: str, 
        vector: np.ndarray, 
        payload: Dict[str, Any],
    ) -> bool:
        """
        Insert or update a document in the vector store.
        
        Args:
            id: Unique identifier for the document
            vector: Embedding vector
            payload: Metadata and content
            
        Returns:
            Success status
        """
        if self._collection is None:
            print("Error: Milvus collection not initialized")
            return False
        
        try:
            # Truncate fields if needed
            section_number = payload.get("section_number", "")
            if len(section_number) > 75:
                section_number = section_number[:75]
            
            heading = payload.get("heading", "")
            if heading and len(heading) > 945:
                heading = heading[:945]
            
            # Extract fields from payload
            entities = [
                [id],  # id
                [vector.tolist()],  # vector
                [payload.get("text", "")],  # text
                [section_number],  # section_number
                [heading],  # heading
                [payload.get("doctype", "tax_law")],  # doctype
            ]
            
            # Insert data
            self._collection.insert(entities)
            return True
            
        except Exception as e:
            print(f"Error upserting document: {str(e)}")
            return False
    
    def bulk_upsert(
        self, 
        points: List[Tuple[str, np.ndarray, Dict[str, Any]]]
    ) -> bool:
        """
        Insert or update multiple documents in the vector store.
        
        Args:
            points: List of (id, vector, payload) tuples
            
        Returns:
            Success status
        """
        if not points:
            return True
            
        if self._collection is None:
            print("Error: Milvus collection not initialized")
            return False
            
        try:
            # Prepare data
            ids = []
            vectors = []
            texts = []
            section_numbers = []
            headings = []
            doctypes = []
            
            for id, vector, payload in points:
                ids.append(id)
                vectors.append(vector.tolist())
                texts.append(payload.get("text", ""))
                
                # Truncate section_number if needed
                section_number = payload.get("section_number", "")
                if len(section_number) > 75:
                    section_number = section_number[:75]
                section_numbers.append(section_number)
                
                # Truncate heading if needed
                heading = payload.get("heading", "")
                if heading and len(heading) > 945:
                    heading = heading[:945]
                headings.append(heading)
                
                doctypes.append(payload.get("doctype", "tax_law"))
            
            # Insert data
            entities = [
                ids,          # id
                vectors,      # vector
                texts,        # text
                section_numbers,  # section_number
                headings,     # heading
                doctypes,     # doctype
            ]
            
            self._collection.insert(entities)
            return True
            
        except Exception as e:
            print(f"Error bulk upserting documents: {str(e)}")
            
            # Try inserting one by one as fallback
            try:
                print("Trying one-by-one insert as fallback...")
                success = True
                for i, (id, vector, payload) in enumerate(points):
                    # Truncate fields if needed
                    section_number = payload.get("section_number", "")
                    if len(section_number) > 75:
                        section_number = section_number[:75]
                    
                    heading = payload.get("heading", "")
                    if heading and len(heading) > 945:
                        heading = heading[:945]
                    
                    single_entity = [
                        [id],
                        [vector.tolist()],
                        [payload.get("text", "")],
                        [section_number],
                        [heading],
                        [payload.get("doctype", "tax_law")],
                    ]
                    
                    try:
                        self._collection.insert(single_entity)
                    except Exception as e2:
                        print(f"Error inserting individual entity: {e2}")
                        success = False
                
                return success
            except Exception as e2:
                print(f"One-by-one fallback also failed: {e2}")
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
        if self._collection is None:
            print("Error: Milvus collection not initialized")
            return []
            
        # Encode query to embedding
        query_embedding = embedding_service.encode_query(query)
        
        # Prepare search expression if filter is provided
        expr = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                elif isinstance(value, bool):
                    conditions.append(f'{key} == {str(value).lower()}')
                else:
                    conditions.append(f'{key} == {value}')
            
            if conditions:
                expr = " && ".join(conditions)
        
        try:
            # Perform search
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
            
            results = self._collection.search(
                data=[query_embedding.tolist()],
                anns_field="vector",
                param=search_params,
                limit=k,
                expr=expr,
                output_fields=["text", "section_number", "heading", "doctype"]
            )
            
            # Convert to SearchHit format
            hits = []
            for hit in results[0]:
                # Build payload
                payload = {
                    "text": hit.entity.get("text", ""),
                    "section_number": hit.entity.get("section_number", ""),
                    "heading": hit.entity.get("heading", ""),
                    "doctype": hit.entity.get("doctype", "tax_law"),
                }
                
                hits.append(SearchHit(
                    id=hit.id,
                    score=hit.score,
                    text=hit.entity.get("text", ""),
                    payload=payload
                ))
            
            return hits
            
        except Exception as e:
            print(f"Error searching Milvus: {str(e)}")
            return []


# Export singleton instance
milvus_store = MilvusStore() 