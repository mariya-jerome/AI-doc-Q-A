import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Models — use stable production models from Groq
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking Configuration
try:
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
except ValueError:
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

# Retrieval Configuration
try:
    TOP_K = int(os.getenv("TOP_K", "4"))
except ValueError:
    TOP_K = 4
