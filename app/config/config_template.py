"""
Configuration template for the application.

Copy this file to config.py and fill in your actual values.
NEVER commit the actual config.py file with real credentials to version control.
"""

# Milvus settings
MILVUS_URI = "your-milvus-uri-here"
MILVUS_API_KEY = "your-milvus-api-key-here"
MILVUS_COLLECTION = "your-milvus-collection-name"
MILVUS_USERNAME = "your-milvus-username"
MILVUS_PASSWORD = "your-milvus-password"
USE_MILVUS = True

# LLM settings
LLM_API_BASE = "https://api.together.xyz/v1"
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
LLM_API_KEY = "your-together-api-key-here"

# OpenAI settings
OPENAI_API_KEY = "your-openai-api-key-here"
USE_OPENAI = False  # Set to True to use OpenAI instead of Together.ai

# Other settings that can be customized
DEBUG = False
USE_INMEMORY = False
DATA_PATH = "data/output_chunks.jsonl"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
ENABLE_RERANK = False
RERANK_MODEL = "BAAI/bge-reranker-large"
RERANK_TOP_K = 20
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.1
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP = 50 