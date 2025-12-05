import os
from dotenv import load_dotenv

load_dotenv()

# Ollama model for text generation
MODEL_NAME = "gemma:2b"   

# Embeddings model (local via Ollama)
EMBEDDING_MODEL = "nomic-embed-text"  

# Max tokens per response
MAX_NEW_TOKENS = 300

# Ollama handles CPU/GPU automatically
DEVICE = "cpu"

TEXTBOOKS = {
    "Math": "./textbooks/math.pdf",
    "English": "./textbooks/english.pdf",
    "EVS": "./textbooks/evs.pdf"
}
