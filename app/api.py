"""
API module for the legal assistant application.

This module defines the FastAPI routes and handlers for the application.
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from app import settings
from app.agents.classifier_agent import classifier_agent
from app.agents.retriever_agent import retriever_agent
from app.agents.drafter_agent import drafter_agent
import socket
from contextlib import closing

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Models
class QueryRequest(BaseModel):
    """Model for query request."""
    
    query: str = Field(..., description="The legal query to answer")
    k: int = Field(8, description="Number of context chunks to retrieve")
    document_type: Optional[str] = Field(None, description="Filter by document type")


class QueryResponse(BaseModel):
    """Model for query response."""
    
    analysis: str = Field(..., description="Detailed legal analysis")
    answer: str = Field(..., description="Concise final answer")
    citations: List[str] = Field(..., description="List of citations used")


class UploadResponse(BaseModel):
    """Model for document upload response."""
    
    filename: str = Field(..., description="Original filename")
    document_id: str = Field(..., description="Document ID")
    document_type: str = Field(..., description="Classified document type")
    message: str = Field(..., description="Status message")


class DocumentMetadata(BaseModel):
    """Model for document metadata."""
    
    filename: str = Field(..., description="Original filename")
    document_type: str = Field(..., description="Document type")
    upload_date: str = Field(..., description="Upload date")
    num_chunks: int = Field(..., description="Number of chunks")


# Routes
@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return {"message": "Indian Tax Law Assistant API", "version": settings.API_VERSION}


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    Process a legal query and return an answer with citations.
    
    Args:
        request: Query request with query text and options
        
    Returns:
        Analysis, answer, and citations
    """
    # Prepare filter conditions if document type is specified
    filter_conditions = None
    if request.document_type:
        filter_conditions = {"doctype": request.document_type}
    
    # Retrieve relevant context
    context_chunks = retriever_agent.retrieve(
        query=request.query,
        k=request.k,
        filter_conditions=filter_conditions
    )
    
    # Draft response
    response = drafter_agent.draft_response(
        query=request.query,
        context_chunks=context_chunks
    )
    
    return response


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload and process a new document.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded file
        
    Returns:
        Upload status information
    """
    # This is a stub - implementation would depend on actual document processing
    # logic from phase 1 PDF processor
    
    filename = file.filename
    document_id = f"doc_{hash(filename)}"
    
    # Read initial content for classification
    content_preview = await file.read(2048)
    content_preview_str = content_preview.decode("utf-8", errors="ignore")
    
    # Rewind file for further processing
    await file.seek(0)
    
    # Classify the document
    metadata = {
        "filename": filename,
        "size": len(content_preview),
        "content_type": file.content_type,
    }
    
    document_type = classifier_agent.classify_document(
        metadata=metadata,
        content_preview=content_preview_str
    )
    
    # Queue background processing
    # background_tasks.add_task(process_document, file, document_id, document_type)
    
    return {
        "filename": filename,
        "document_id": document_id,
        "document_type": document_type,
        "message": "Document queued for processing"
    }


@app.get("/documents", response_model=List[DocumentMetadata])
async def list_documents():
    """
    List all uploaded documents.
    
    Returns:
        List of document metadata
    """
    # This is a stub - implementation would depend on actual document storage
    # Would typically query the database or vector store for documents
    
    return [
        {
            "filename": "IncomeTaxAct_2025.pdf",
            "document_type": "tax_law",
            "upload_date": "2024-01-01",
            "num_chunks": 1500
        }
    ]


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    print(f"Starting {settings.API_TITLE} on {settings.LLM_API_BASE}")
    print(f"Using embedding model: {settings.EMBED_MODEL}")
    print(f"Reranking enabled: {settings.ENABLE_RERANK}")


# Find and move this function above the main section
def find_free_port(start_port=8000, max_port=8100):
    """Find a free port to use for the API server."""
    for port in range(start_port, max_port):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(('localhost', port)) != 0:
                return port
    return start_port  # Fallback to start_port if nothing else is available


# Run the application
if __name__ == "__main__":
    import uvicorn
    port = find_free_port()
    print(f"Starting API server on port {port}")
    uvicorn.run("app.api:app", host="0.0.0.0", port=port, reload=True) 