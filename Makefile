.PHONY: help run-api process-pdf import-to-milvus test clean

# Default target
help:
	@echo "Available targets:"
	@echo "  process-pdf         - Process PDF into chunks"
	@echo "  import-to-milvus    - Import chunks to Milvus vector database"
	@echo "  run-api             - Run API server with Together.ai backend"
	@echo "  test                - Run unit tests"
	@echo "  clean               - Clean temporary files"

# Process PDF into chunks
process-pdf:
	@echo "Processing PDF into chunks..."
	python pdf_processor.py data/IncomeTaxAct_2025.pdf --output data/output_chunks.jsonl

# Import chunks to Milvus
import-to-milvus:
	@echo "Importing chunks to Milvus..."
	python scripts/import_to_milvus.py --jsonl data/output_chunks.jsonl

# Run API with Together.ai backend
run-api:
	@echo "Starting API server with Together.ai backend..."
	python -m uvicorn app.api:app --host 0.0.0.0 --port 8000

# Run tests
test:
	@echo "Running unit tests..."
	python scripts/verify_system.py

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name ".eggs" -exec rm -rf {} + 