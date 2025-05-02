# Indian Tax Law Assistant

A legal document processing and retrieval system specialized for Indian tax law documents. This system processes PDF documents, extracts and chunks text with proper metadata, and enables semantic search with proper citations.

## Project Overview

This project consists of two main components:

1. **PDF Processing Pipeline** - Extracts text from PDFs, applies OCR for scanned pages, and creates semantically meaningful chunks
2. **Legal Assistant Backend** - Provides semantic search functionality with proper citations using vector embeddings

## Project Structure

```
project/
├── app/                  # Main application code
│   ├── agents/           # Specialized agents for retrieval and drafting
│   ├── services/         # Core services (embeddings, vector store, LLM)
│   ├── api.py            # FastAPI endpoints
│   └── settings.py       # Application settings
├── data/                 # Data files
│   ├── IncomeTaxAct_2025.pdf     # Source PDF document
│   └── output_chunks.jsonl       # Processed chunks from the PDF
├── docs/                 # Documentation and credentials
├── scripts/              # Utility scripts
│   ├── simple_pdf_search.py      # In-memory vector search script
│   ├── import_to_milvus.py       # Script to upload chunks to Milvus
│   ├── fix_strict_import.py      # Script to handle Milvus field limits
│   ├── unit_tests.py             # Unit tests for the components
├── static/               # Frontend assets
│   └── index.html        # Web interface
├── Makefile              # Build commands
├── pdf_processor.py      # Main PDF processing script
├── requirements.txt      # Python dependencies
├── run_demo.py           # Script to run the application
└── README.md             # This file
```

## Features

- **PDF Text Extraction**
  - Handles both regular and scanned PDF documents
  - Applies OCR for image-only pages using Tesseract
  - Preserves document structure as markdown

- **Semantic Chunking**
  - Creates semantically meaningful chunks with context preservation
  - Maintains section boundaries and hierarchical structure
  - Generates metadata for each chunk (section numbers, headings)

- **Vector Search**
  - Uses BAAI/bge-large-en-v1.5 embedding model for state-of-the-art semantic search
  - Integrates with Milvus vector database for scalable retrieval
  - Supports metadata filtering (document type, section, etc.)

- **Legal Response Generation**
  - Provides formal legal analysis with proper citations
  - Formats responses with analysis, answer, and citation sections
  - Uses Meta-Llama-3.3-70B-Instruct-Turbo for high-quality legal reasoning

- **Web Interface**
  - Clean, responsive UI built with Bootstrap 5
  - Interactive query interface with example questions
  - Structured display of legal responses

## Technical Stack

- **PDF Processing**: PyMuPDF, Tesseract OCR, tiktoken
- **Vector Database**: Milvus Cloud
- **Embeddings**: BAAI/bge-large-en-v1.5
- **LLM Integration**: Together.ai API with Meta-Llama-3.3-70B-Instruct-Turbo
- **Backend**: FastAPI, Pydantic, Uvicorn
- **Frontend**: HTML/CSS/JS with Bootstrap 5

## Requirements

- Python 3.8+
- PyMuPDF and dependencies (for PDF processing)
- SentenceTransformers (for embeddings)
- Pymilvus (for vector storage)
- Tesseract OCR (for scanned pages)
- Together.ai API key (for LLM access)

## Installation

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Install Tesseract OCR (only needed for OCR functionality):
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - macOS: `brew install tesseract`
   - Linux: `apt-get install tesseract-ocr`

3. Configure environment variables:
   - Create a `.env` file in the project root with your Together.ai API key:
     ```
     LLM_API_KEY=your_together_api_key_here
     ```

## Usage

### Processing a PDF Document

```bash
python pdf_processor.py data/IncomeTaxAct_2025.pdf --output data/output_chunks.jsonl
```

### Running the Application

The simplest way to run the system is to use the demo script:

```bash
python run_demo.py
```

This will:
1. Load the processed chunks
2. Connect to Milvus for vector storage
3. Start the API server on port 8000
4. Make the web interface available at http://localhost:8000

### Accessing the Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

The web interface provides a user-friendly way to ask tax law questions and view the responses with proper citations.

### API Usage

You can also interact with the system programmatically via the API:

```bash
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"query": "What are the deductions under section 80C?"}'
```

## Development

### Adding New Documents

To add new documents to the system:

1. Process the PDF:
   ```
   python pdf_processor.py path/to/new_document.pdf --output data/new_chunks.jsonl
   ```

2. Import to Milvus:
   ```
   python scripts/import_to_milvus.py --jsonl data/new_chunks.jsonl
   ```

### Customizing the LLM

You can customize the LLM model and parameters in `app/settings.py`:

```python
# LLM settings
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.together.xyz/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo")
```

## Troubleshooting

- **Missing API key errors**: Ensure your Together.ai API key is correctly set in `app/settings.py` or as an environment variable
- **Port already in use**: Change the port in the `python -m uvicorn app.api:app --host 0.0.0.0 --port 8000` command
- **Milvus connection issues**: Verify your Milvus credentials and connectivity

## License

This project is proprietary and confidential.

## Acknowledgments

- BAAI for the BGE embedding models
- Meta for the Llama 3.3 model
- Together.ai for providing API access to high-quality LLMs 