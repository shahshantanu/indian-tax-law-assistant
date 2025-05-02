"""
Settings module for the legal assistant application.

This module imports configurations from a separate config file and provides fallbacks where needed.
"""

import os
import sys
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base settings
BASE_DIR = Path(__file__).resolve().parent.parent

# Import settings from config file
try:
    from app.config.config import *
except ImportError:
    # If config.py doesn't exist, try to use environment variables with fallbacks
    print("Config file not found. Using environment variables with fallbacks.")
    
    # Debug setting
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Milvus settings
    MILVUS_URI = os.getenv("MILVUS_URI")
    MILVUS_API_KEY = os.getenv("MILVUS_API_KEY")
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "taxlaw_collection")
    MILVUS_USERNAME = os.getenv("MILVUS_USERNAME")
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
    USE_MILVUS = os.getenv("USE_MILVUS", "false").lower() == "true"
    
    # In-memory settings
    USE_INMEMORY = os.getenv("USE_INMEMORY", "true").lower() == "true"
    DATA_PATH = os.getenv("DATA_PATH", "data/output_chunks.jsonl")
    
    # Embedding model settings
    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    ENABLE_RERANK = os.getenv("ENABLE_RERANK", "false").lower() == "true"
    RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-large")
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "20"))
    
    # LLM settings
    LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.together.xyz/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    
    # Document processing settings
    CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Exit if required environment variables are missing
    if not all([MILVUS_URI, MILVUS_API_KEY, MILVUS_USERNAME, MILVUS_PASSWORD, LLM_API_KEY]) and not USE_INMEMORY:
        print("ERROR: Missing required environment variables.")
        print("Please either create a config.py file or set the required environment variables.")
        sys.exit(1)

# API settings
API_TITLE = "Legal Assistant API"
API_DESCRIPTION = "API for retrieving and analyzing legal documents."
API_VERSION = "0.1.0"

def get_llm_config() -> Dict[str, Any]:
    """Return the LLM configuration dictionary."""
    return {
        "api_base": LLM_API_BASE,
        "model": LLM_MODEL,
        "api_key": LLM_API_KEY,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
    } 