# src/retriever_faiss.py

import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

from config_cgi import (
    CHUNKS_PATH,
    FAISS_INDEX_PATH,
    OPENAI_EMBED_MODEL,
    ENV_PATH,
)

# =========================
# 1) Chargement config & clients
# =========================

# Charger .env (clé OpenAI, etc.)
load_dotenv(ENV_PATH)
client = OpenAI()

# Charger les chunks JSON (liste de dict)
CHUNKS_PATH = Path(CHUNKS_PATH)
with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    CHUNKS: List[Dict[str, Any]] = json.load(f)

# Charger l’index FAISS
FAISS_INDEX_PATH = Path(FAISS_INDEX_PATH)
faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

# Cross-encoder (reranker) – lazy load
_CROSS_ENCODER: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    """
    Charge le modèle de rerank une seule fois (lazy).
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        # Modèle standard pour le rerank, rapide et efficace
        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CROSS_ENCODER


# =========================
# 2) Embeddings OpenAI
# =========================

def _embed_texts(texts: List[str]) -> np.ndarray:
    """
    Calcule les embeddings OpenAI pour une liste de textes.
    Retourne un array numpy [n, d].
    """
    resp = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=texts,
    )
    vectors = [d.embedding for d in resp.data]
    return np.array(vectors, dtype="float32")


# =========================
# 3) Recherche FAISS + rerank
# =========================

def search_chunks(
    question: str,
    k: int = 3,
    use_rerank: bool = True,
    faiss_top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Recherche des chunks pertinents avec FAISS + rerank (cross-encoder).

    Logique :
      - FAISS → récupère `faiss_top_k` candidats (ex : 20).
      - Cross-encoder → rerank ces candidats.
      - On renvoie les `k` meilleurs au moteur (ex : 3).

    Args
    ----
    question : str
        Question de l’utilisateur.
    k : int
        Nombre final de chunks retournés (top-k après rerank).
    use_rerank : bool
        Si False : on ne fait que FAISS.
    faiss_top_k : int
        Nombre de candidats à récupérer d’abord via FAISS.

    Returns
    -------
    List[dict]
        Liste de résultats :
        [
          {
            "rank_faiss": int,
            "score_faiss": float,
            "score_rerank": float | None,
            "chunk": { ... }  # dict du chunk complet
          },
          ...
        ]
    """
    if k <= 0:
        return []

    # 1) Embedding de la question
    q_vec = _embed_texts([question])[0].reshape(1, -1)

    # 2) Recherche FAISS
    k_faiss = faiss_top_k if use_rerank else k
    distances, indices = faiss_index.search(q_vec, k_faiss)

    distances = distances[0]
    indices = indices[0]

    candidates: List[Dict[str, Any]] = []
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        if idx < 0:
            continue  # FAISS peut renvoyer -1 si pas assez de résultats

        chunk = CHUNKS[idx]
        candidates.append(
            {
                "rank_faiss": rank,
                "score_faiss": float(dist),
                "score_rerank": None,
                "chunk": chunk,
            }
        )

    if not candidates:
        return []

    # 3) Si pas de rerank → on garde juste les k premiers FAISS
    if not use_rerank:
        return candidates[:k]

    # 4) Rerank avec cross-encoder
    model = _get_cross_encoder()
    pairs = [(question, c["chunk"].get("text", "")) for c in candidates]
    scores = model.predict(pairs)

    for c, s in zip(candidates, scores):
        c["score_rerank"] = float(s)

    # 5) Trie décroissant sur le score rerank
    candidates.sort(key=lambda x: x["score_rerank"], reverse=True)

    # 6) Retourner les k meilleurs au moteur
    return candidates[:k]


# =========================
# 4) Petit test en CLI
# =========================

if __name__ == "__main__":
    print("=== Test simple retriever (FAISS + rerank) ===")
    q = input("Question > ").strip()
    res = search_chunks(q, k=3, use_rerank=True, faiss_top_k=20)
    for i, r in enumerate(res, start=1):
        ch = r["chunk"]
        print(f"\n--- Rank final {i} ---")
        print(f"ID chunk     : {ch.get('id')}")
        print(f"Article      : {ch.get('article')}")
        print(f"Titre        : {ch.get('title')}")
        print(f"Score FAISS  : {r['score_faiss']}")
        print(f"Score rerank : {r['score_rerank']}")
        text = (ch.get("text") or "").replace("\n", " ")
        print(f"Snippet      : {text[:200]}...")
