import os
from types import SimpleNamespace

# Lightweight settings using environment variables. Populate a `.env` file in
# the project root if you want to persist local values (do NOT commit .env).

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma")
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


settings = SimpleNamespace(
    chroma_persist_dir=CHROMA_PERSIST_DIR,
    model_name=MODEL_NAME,
    ollama_host=OLLAMA_HOST,
    ollama_api_key=OLLAMA_API_KEY,
    openai_api_key=OPENAI_API_KEY,
    log_level=LOG_LEVEL,
)
