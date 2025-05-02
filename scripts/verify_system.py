#!/usr/bin/env python3
"""
Script to verify that all components of the system are working.

This script checks:
1. Milvus connectivity
2. The retriever agent
3. The classifier agent 
4. The drafter agent
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import settings
from app.services.milvus_store import milvus_store
from app.agents.retriever_agent import retriever_agent
from app.agents.classifier_agent import classifier_agent
from app.agents.drafter_agent import drafter_agent

# Force using Milvus
os.environ["USE_MILVUS"] = "true"
os.environ["USE_INMEMORY"] = "false"

def test_milvus_connection():
    """Test connection to Milvus."""
    print("\n===== TESTING MILVUS CONNECTION =====")
    
    # Check if collection is initialized
    if milvus_store._collection is None:
        print("Error: Milvus collection not initialized")
        return False
    
    print(f"Successfully connected to Milvus at {settings.MILVUS_URI}")
    print(f"Using collection: {settings.MILVUS_COLLECTION}")
    
    # Try a simple search to verify functionality
    try:
        print("Testing search functionality...")
        results = milvus_store.search("What is income tax?", k=1)
        if results:
            print(f"Search successful. Found {len(results)} results.")
            return True
        else:
            print("Search returned no results. Collection may be empty.")
            return False
    except Exception as e:
        print(f"Error searching Milvus: {e}")
        return False

def test_retriever_agent():
    """Test the retrieval agent."""
    print("\n===== TESTING RETRIEVER AGENT =====")
    
    test_queries = [
        "What are the deductions under section 80C?",
        "What is the tax rate for senior citizens?",
        "How do I file an income tax return?"
    ]
    
    success = False
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        
        # Execute retrieval
        chunks = retriever_agent.retrieve(query, k=2)
        
        if chunks:
            print(f"Retrieved {len(chunks)} chunks:")
            for j, chunk in enumerate(chunks):
                print(f"  Result {j+1}:")
                print(f"    Score: {chunk.score:.4f}")
                print(f"    Citation: {chunk.citation}")
                print(f"    Text excerpt: {chunk.text[:100]}...")
            success = True
        else:
            print("No results found.")
    
    return success

def test_classifier_agent():
    """Test the classifier agent."""
    print("\n===== TESTING CLASSIFIER AGENT =====")
    
    try:
        # Test with sample metadata
        metadata = {
            "filename": "IncomeTaxAct_2025.pdf",
            "size": 1048576,
            "content_type": "application/pdf",
        }
        
        content_preview = "Chapter V INCOME TAX ACT Income from Salaries..."
        
        doc_type = classifier_agent.classify_document(
            metadata=metadata,
            content_preview=content_preview
        )
        
        print(f"Classification result: {doc_type}")
        return True
    except Exception as e:
        print(f"Error in classifier agent: {e}")
        return False

def test_drafter_agent():
    """Test the drafter agent."""
    print("\n===== TESTING DRAFTER AGENT =====")
    
    query = "What are the tax benefits for senior citizens?"
    print(f"Query: {query}")
    
    # Retrieve context
    chunks = retriever_agent.retrieve(query, k=3)
    if not chunks:
        print("No context chunks found for testing drafter")
        return False
    
    print(f"Retrieved {len(chunks)} context chunks")
    
    # Generate response
    try:
        print("Generating response...")
        response = drafter_agent.draft_response(query, chunks)
        
        print("\nAnswer:")
        print(response.answer[:200] + "..." if len(response.answer) > 200 else response.answer)
        print(f"\nCitations: {response.citations}")
        
        return True
    except Exception as e:
        print(f"Error in drafter agent: {e}")
        return False

def main():
    """Run all verification tests."""
    print("Starting system verification...")
    
    # Test Milvus connection
    milvus_ok = test_milvus_connection()
    if not milvus_ok:
        print("\nMilvus connection failed. Please check your Milvus configuration.")
        print("Make sure data has been imported with: python scripts/import_to_milvus.py")
        return 1
    
    # Test retrieval
    retrieval_ok = test_retriever_agent()
    if not retrieval_ok:
        print("\nRetriever agent test failed. Check your vector store configuration.")
    
    # Test classifier
    classifier_ok = test_classifier_agent()
    if not classifier_ok:
        print("\nClassifier agent test failed.")
    
    # Test drafter
    drafter_ok = test_drafter_agent()
    if not drafter_ok:
        print("\nDrafter agent test failed.")
    
    # Summary
    print("\n===== VERIFICATION SUMMARY =====")
    print(f"Milvus Connection: {'✓' if milvus_ok else '✗'}")
    print(f"Retriever Agent: {'✓' if retrieval_ok else '✗'}")
    print(f"Classifier Agent: {'✓' if classifier_ok else '✗'}")
    print(f"Drafter Agent: {'✓' if drafter_ok else '✗'}")
    
    if milvus_ok and retrieval_ok and classifier_ok and drafter_ok:
        print("\n✅ All systems working! Ready for demo.")
        print("\nTo start the API server, run: python scripts/run_api.py")
        return 0
    else:
        print("\n⚠️ Some components are not working properly.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 