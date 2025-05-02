#!/usr/bin/env python3
"""
Demo script for the Indian Tax Law Assistant.
This script runs the legal assistant with Milvus vector store.
"""

import os
import sys
import argparse
import socket
from contextlib import closing
from pathlib import Path

def find_free_port(start_port=8000, max_port=8100):
    """Find a free port to use for the API server."""
    for port in range(start_port, max_port):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(('localhost', port)) != 0:
                return port
    return 8000  # Fallback to 8000 if nothing else is available

def main():
    """Run the demo script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Indian Tax Law Assistant demo")
    parser.add_argument("--data", type=str, default="data/output_chunks.jsonl", 
                      help="Path to the processed data file (default: data/output_chunks.jsonl)")
    parser.add_argument("--port", type=int, default=0,
                      help="Port to run the API server (default: auto-detect free port)")
    args = parser.parse_args()
    
    # Check if output_chunks.jsonl exists
    data_file = Path(args.data)
    if not data_file.exists():
        print(f"Error: Could not find processed data at {data_file}")
        print("Have you processed the PDF yet? If not, run:")
        print("  python pdf_processor.py data/IncomeTaxAct_2025.pdf --output data/output_chunks.jsonl")
        return 1
    
    # Set environment variables for Milvus
    os.environ["USE_MILVUS"] = "true"
    os.environ["USE_INMEMORY"] = "false"
    os.environ["DATA_PATH"] = str(data_file)
    
    # Determine port
    port = args.port if args.port > 0 else find_free_port()
    
    print("=" * 80)
    print(f"Starting Indian Tax Law Assistant on port {port}")
    print("=" * 80)
    print(f"• Using data file: {data_file}")
    print(f"• Vector store: Milvus")
    print(f"• Web interface: http://localhost:{port}")
    print(f"• API endpoint: http://localhost:{port}/ask")
    print("=" * 80)
    
    # Start the API server
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=port)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 