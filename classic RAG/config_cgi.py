# src/config_cgi.py
from pathlib import Path

# Racine du projet (dossier qui contient "src" et "data")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dossiers de données
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"
MARKDOWN_DIR = DATA_DIR / "markdown"
JSON_DIR = DATA_DIR / "json"
INDEX_DIR = DATA_DIR / "index"

# Fichiers utilisés par le RAG
CHUNKS_PATH = JSON_DIR / "cgi-2025_chunks.json"
FAISS_INDEX_PATH = INDEX_DIR / "cgi-2025_faiss.index"

# Fichier .env à la racine
ENV_PATH = PROJECT_ROOT / ".env"

# Modèles OpenAI
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4.1-mini"   # 

# Paramètres de recherche
FAISS_K = 20    # nombre de voisins récupérés dans FAISS
TOP_K = 3       # nombre de chunks envoyés au LLM
SOURCE_NAME = "CGI 2025"
