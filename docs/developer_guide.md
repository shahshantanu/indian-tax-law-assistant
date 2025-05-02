# Developer Guide: Indian Tax Law Assistant

This document provides detailed information for developers working on the Indian Tax Law Assistant project.

## Architecture Overview

The system follows a modular architecture with clear separation of concerns:

```
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│   PDF Ingestion │───▶│  Vector Storage │───▶│  Query Pipeline │
└────────────────┘    └────────────────┘    └────────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│ Text Processing │    │  Embeddings    │    │  API Endpoints │
└────────────────┘    └────────────────┘    └────────────────┘
```

### Core Components

1. **PDF Processing Pipeline** (`pdf_processor.py`)
   - Handles document ingestion and chunking
   - Creates semantic chunks with metadata
   - Outputs JSONL files ready for embedding

2. **Vector Services** (`app/services/`)
   - Embedding service for vector generation
   - Milvus integration for vector storage and retrieval
   - In-memory fallback for development

3. **Agent System** (`app/agents/`)
   - Retriever: Semantic search and context gathering
   - Classifier: Document type identification
   - Drafter: Response generation with citations

4. **API Layer** (`app/api.py`)
   - RESTful endpoints for interaction
   - Request/response models with validation
   - Static file serving for the frontend

## Development Environment Setup

### Prerequisites

1. Python 3.8+ with pip
2. Tesseract OCR installed
3. Together.ai API account
4. Milvus Cloud account (or local instance)

### Local Development Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd indian-tax-law-assistant
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file for local configuration:
   ```
   LLM_API_KEY=your_together_api_key
   MILVUS_URI=your_milvus_uri
   MILVUS_API_KEY=your_milvus_api_key
   MILVUS_USERNAME=your_milvus_username
   MILVUS_PASSWORD=your_milvus_password
   ```

5. Process a sample document:
   ```
   python pdf_processor.py path/to/sample.pdf --output data/test_chunks.jsonl
   ```

6. Start the development server:
   ```
   python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
   ```

## Code Organization

### PDF Processing

The PDF processing pipeline in `pdf_processor.py` follows these steps:

1. **Text Extraction**: Extract text from PDF pages with OCR fallback
2. **Text Cleaning**: Normalize text and handle encoding issues
3. **Section Identification**: Identify section boundaries and headings
4. **Chunk Creation**: Create semantic chunks with proper context
5. **Metadata Assignment**: Assign metadata to each chunk
6. **JSONL Export**: Export chunks as JSONL for vector embedding

Key functions:
- `extract_text_from_pdf()`: Extracts raw text with OCR support
- `clean_and_format_text()`: Cleans and formats extracted text
- `split_into_sections()`: Splits text into logical sections
- `create_chunks()`: Creates semantic chunks with overlap
- `process_pdf()`: Main entry point for PDF processing

### Vector Services

The vector services handle embedding generation and vector storage:

1. **Embedding Service** (`app/services/embeddings.py`):
   - Singleton service for embedding generation
   - Uses SentenceTransformers with BGE model
   - Handles asynchronous batched encoding

2. **Vector Store** (`app/services/milvus_store.py`):
   - Interfaces with Milvus for vector storage
   - Provides search and filtering capabilities
   - Handles collection creation and management

3. **In-memory Store** (`app/services/inmemory_store.py`):
   - Provides fallback when Milvus is unavailable
   - Uses numpy for vector operations
   - Useful for testing and development

### Agent System

The agent system is responsible for retrieval and response generation:

1. **Retriever Agent** (`app/agents/retriever_agent.py`):
   - Converts queries to vector embeddings
   - Searches vector store for relevant chunks
   - Applies filters and metadata-based ranking

2. **Classifier Agent** (`app/agents/classifier_agent.py`):
   - Identifies document types from content
   - Extracts metadata from uploaded documents
   - Supports classification for routing and filtering

3. **Drafter Agent** (`app/agents/drafter_agent.py`):
   - Formats context for LLM consumption
   - Generates structured legal responses
   - Ensures proper citation formatting

### API Layer

The API layer provides the interface for external interaction:

1. **API Endpoints** (`app/api.py`):
   - `/ask`: Main query endpoint
   - `/upload`: Document upload endpoint
   - `/documents`: Document listing endpoint

2. **Settings** (`app/settings.py`):
   - Configuration management
   - Environment variable handling
   - Default values for the application

## Extending the System

### Adding New Document Types

To add support for a new document type:

1. Update the `classifier_agent.py` to recognize the new document type
2. Add appropriate filters in the retriever agent
3. Consider specific prompting in the drafter agent for the document type

### Customizing the Frontend

The frontend is a single-page application in `static/index.html`:

1. UI components use Bootstrap 5 for styling
2. JavaScript handles API interaction and response rendering
3. Modify the HTML/CSS/JS directly to customize the interface

### Implementing New Features

When implementing new features:

1. Follow the existing module structure
2. Add appropriate tests in `scripts/unit_tests.py`
3. Update documentation to reflect changes
4. Consider backward compatibility

## API Reference

### `/ask` Endpoint

**Request:**
```json
{
  "query": "What are the deductions under section 80C?",
  "k": 8,
  "document_type": "tax_law"
}
```

**Response:**
```json
{
  "analysis": "Detailed legal analysis...",
  "answer": "Concise answer...",
  "citations": [
    "[1] Section 80C (Deductions)",
    "[2] Section 80CCD (Pension Scheme)"
  ]
}
```

### `/upload` Endpoint

**Request:**
```
POST /upload
Content-Type: multipart/form-data

file: <binary-pdf-data>
```

**Response:**
```json
{
  "filename": "uploaded.pdf",
  "document_id": "doc_12345",
  "document_type": "tax_law",
  "message": "Document queued for processing"
}
```

### `/documents` Endpoint

**Request:**
```
GET /documents
```

**Response:**
```json
[
  {
    "filename": "IncomeTaxAct_2025.pdf",
    "document_type": "tax_law",
    "upload_date": "2024-01-01",
    "num_chunks": 1500
  }
]
```

## Troubleshooting Development Issues

### Common Problems

1. **Embedding Model Issues**:
   - Check GPU availability with `torch.cuda.is_available()`
   - Use CPU fallback for development with `DISABLE_EMBEDDINGS=true`

2. **Milvus Connection**:
   - Verify credentials in settings.py or .env
   - Check network connectivity to Milvus instance
   - Use `--use-inmemory` flag for local testing

3. **LLM API Issues**:
   - Validate API key and base URL
   - Check response status codes and error messages
   - Use the Together.ai dashboard to debug rate limits

### Debugging Tips

1. Enable debug mode in settings.py: `DEBUG=true`
2. Check application logs for error messages
3. Use FastAPI's interactive docs at `/docs` for API testing
4. Inspect vector search results with `print_debug=True`

## Performance Optimization

### Embedding Generation

- Use batched processing for embedding generation
- Enable GPU acceleration when available
- Consider quantization for large embedding models

### Vector Search

- Tune vector search parameters for better recall
- Adjust HNSW index parameters: M=8, efConstruction=64
- Use proper filters to narrow search scope

### LLM Inference

- Optimize prompt design for token efficiency
- Use lower temperatures (0.1-0.2) for factual responses
- Consider context length when designing prompts

## Deployment Considerations

### Environment Variables

Set these environment variables in production:

```
LLM_API_KEY=<api-key>
MILVUS_URI=<milvus-uri>
MILVUS_API_KEY=<milvus-api-key>
MILVUS_USERNAME=<username>
MILVUS_PASSWORD=<password>
```

### Security Considerations

1. Never expose API keys in client-side code
2. Implement proper authentication for the API
3. Sanitize and validate all user inputs
4. Consider rate limiting for public endpoints

### Scaling

1. Use a proper ASGI server for production (Uvicorn behind Nginx)
2. Consider containerization with Docker
3. Implement caching for common queries
4. Scale Milvus collections for larger document sets

## Contributing

When contributing to the project:

1. Follow the existing code style and patterns
2. Write unit tests for new functionality
3. Update documentation to reflect changes
4. Create clear pull requests with descriptive titles

## License and Legal

This project is proprietary and confidential. All rights reserved. 