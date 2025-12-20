# src/config_graph.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENV_PATH = ROOT / ".env"

OPENAI_CHAT_MODEL = "gpt-4o-mini"          # ou gpt-4.1-mini
OPENAI_EMBED_MODEL = "text-embedding-3-small"

SOURCE_NAME_GRAPH = "CGI 2025 (GraphRAG Communities)"

# chemins index communities
GRAPH_INDEX_PATH = ROOT / "data" / "graph" / "communities.faiss"
GRAPH_META_PATH  = ROOT / "data" / "graph" / "communities_meta.json"

K_CANDIDATES = 20
TOP_K_COMMUNITIES = 3
