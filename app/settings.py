"""
Settings module for the legal assistant application.

This module reads environment variables and provides default configurations.
"""

import os
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base settings
BASE_DIR = Path(__file__).resolve().parent.parent
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Milvus settings
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-5c337293fb85b24.serverless.gcp-us-west1.cloud.zilliz.com")
MILVUS_API_KEY = os.getenv("MILVUS_API_KEY", "key-sphcnwtbarmyzthxhhilxzkey-sphcnwtbarmyzthxhhilxz")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "taxlaw_collection")
MILVUS_USERNAME = os.getenv("MILVUS_USERNAME", "db_5c337293fb85b24")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Hr6*)}6GD>VVRB5u")
USE_MILVUS = os.getenv("USE_MILVUS", "true").lower() == "true"

# In-memory settings
USE_INMEMORY = os.getenv("USE_INMEMORY", "false").lower() == "true"
DATA_PATH = os.getenv("DATA_PATH", "data/output_chunks.jsonl")

# Embedding model settings
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "false").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-large")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "20"))

# LLM settings
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.together.xyz/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
LLM_API_KEY = os.getenv("LLM_API_KEY", "5fa9b6ba9618f0eeca879a702d2ac0317b7ff033f5d467358d00d862e7ae19c5")  # Get your API key from https://api.together.xyz/settings/api-keys
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Document processing settings
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

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