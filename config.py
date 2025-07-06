# rag_chatbot/config.py

# Gemini API Key (leave empty string; Canvas will provide at runtime if needed)
# DO NOT paste your actual API key here for security.
GEMINI_API_KEY = "" # Leave as empty string as per instructions

# Embedding Model Configuration
# Using a local Sentence Transformers model for embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # A good balance of performance and size

# Vector Store Configuration
# Path to save/load the FAISS index
FAISS_INDEX_PATH = "faiss_index.bin"

# Document Chunking Parameters
# These values are crucial for effective retrieval.
# CHUNK_SIZE: The maximum number of characters in each text chunk.
CHUNK_SIZE = 1000
# CHUNK_OVERLAP: The number of characters to overlap between consecutive chunks.
# This helps maintain context across chunks.
CHUNK_OVERLAP = 200

# Large Language Model (LLM) Configuration
# Using Gemini 2.0 Flash model
LLM_MODEL_NAME = "gemini-2.0-flash" # <--- CHANGED TO GEMINI MODEL NAME
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent" # <--- ADDED GEMINI API URL

# Maximum number of retrieved chunks to pass to the LLM
TOP_K_RETRIEVED_CHUNKS = 4