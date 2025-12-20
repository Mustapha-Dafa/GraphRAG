# src/retriever_graph.py
import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]

GRAPH_INDEX_PATH = ROOT / "data" / "graph" / "communities.faiss"
GRAPH_META_PATH = ROOT / "data" / "graph" / "communities_meta.json"
COMM_PROFILES_PATH = ROOT / "data" / "graph" / "communities" / "communities_profiles.json"

ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY manquant (mets-le dans .env).")

client = OpenAI(api_key=OPENAI_API_KEY)


def _embed(text: str) -> np.ndarray:
    text = (text or "").strip()
    if not text:
        return np.zeros((1536,), dtype=np.float32)
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def _load_meta_items() -> List[Dict[str, Any]]:
    # 1) meta alignée FAISS
    if GRAPH_META_PATH.exists():
        meta = json.load(open(GRAPH_META_PATH, "r", encoding="utf-8"))
        items = meta.get("items")
        if isinstance(items, list) and items:
            return items

    # 2) fallback: profiles dict {cid: {...}}
    if COMM_PROFILES_PATH.exists():
        d = json.load(open(COMM_PROFILES_PATH, "r", encoding="utf-8"))
        if isinstance(d, dict) and d:
            def _key(x):
                try:
                    return int(x)
                except Exception:
                    return x

            items = []
            for cid in sorted(d.keys(), key=_key):
                obj = d[cid]
                if isinstance(obj, dict):
                    obj = dict(obj)
                    obj.setdefault("community_id", str(cid))
                    items.append(obj)
            if items:
                return items

    raise FileNotFoundError(
        "Meta communautés introuvable. Attendu: data/graph/communities_meta.json "
        "ou data/graph/communities/communities_profiles.json"
    )


def search_communities(query: str, k_candidates: int = 10) -> List[Dict[str, Any]]:
    """
    IMPORTANT: format attendu par engine_graph.py:
      [{"community": {...}, "score": float, "rank": int}, ...]
    """
    if not GRAPH_INDEX_PATH.exists():
        raise FileNotFoundError(f"Index FAISS introuvable: {GRAPH_INDEX_PATH}")

    index = faiss.read_index(str(GRAPH_INDEX_PATH))
    meta_items = _load_meta_items()

    q = _embed(query).astype(np.float32).reshape(1, -1)
    D, I = index.search(q, k_candidates)

    out: List[Dict[str, Any]] = []
    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
        idx = int(idx)
        if idx < 0:
            continue
        if idx >= len(meta_items):
            continue  # sécurité

        c = dict(meta_items[idx])
        # normaliser community_id
        c.setdefault("community_id", str(c.get("id", idx)))
        c["community_id"] = str(c["community_id"])

        out.append({
            "community": c,
            "score": float(dist),
            "rank": rank
        })

    return out
