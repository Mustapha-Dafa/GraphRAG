# src/build_graph_index.py
# Build a FAISS index over GraphRAG community profiles (title/summary/keywords)
# Output:
#   - data/graph/communities.faiss
#   - data/graph/communities_meta.json

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv

try:
    # openai>=1.0
    from openai import OpenAI
except Exception as e:
    raise RuntimeError(
        "Le package openai n'est pas installé ou incompatible. "
        "Fais: pip install --upgrade openai"
    ) from e


# ----------------------------
# Paths / Config
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

COMM_PROFILES_PATH = ROOT / "data" / "graph" / "communities" / "communities_profiles.json"
OUT_INDEX_PATH = ROOT / "data" / "graph" / "communities.faiss"
OUT_META_PATH = ROOT / "data" / "graph" / "communities_meta.json"


# ----------------------------
# Helpers
# ----------------------------
def _load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        # not fatal; env can be set in system
        pass


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquant (mets-le dans .env).")
    return OpenAI(api_key=api_key)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_community_list(data: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
      - dict: { "8": {..}, "13": {..} }
      - list: [ {..}, {..} ]
    Returns a list of community dicts.
    """
    if isinstance(data, dict):
        # values are the community objects
        return [v for v in data.values() if isinstance(v, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    raise TypeError(f"Format JSON inattendu: {type(data)}")


def _safe_str(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def _community_id(c: Dict[str, Any]) -> str:
    # your profiles contain "community_id": "8"
    return _safe_str(c.get("community_id") or c.get("id") or c.get("cid"))


def _community_text(c: Dict[str, Any]) -> str:
    """
    Text used for embedding (keep it compact but informative).
    """
    cid = _community_id(c)
    title = _safe_str(c.get("title"))
    summary = _safe_str(c.get("summary"))
    keywords = c.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []

    # Keep max keywords to avoid very long inputs
    kw = [str(k).strip() for k in keywords if str(k).strip()]
    kw = kw[:25]

    parts = []
    if cid:
        parts.append(f"COMMUNITY {cid}")
    if title:
        parts.append(f"TITLE: {title}")
    if summary:
        parts.append(f"SUMMARY: {summary}")
    if kw:
        parts.append("KEYWORDS: " + ", ".join(kw))

    text = "\n".join(parts).strip()
    return text if text else f"COMMUNITY {cid}"


def _batched(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _embed_texts(
    client: OpenAI,
    texts: List[str],
    model: str,
    batch_size: int = 64,
    max_retries: int = 5,
    sleep_base: float = 1.5,
) -> np.ndarray:
    """
    Returns embeddings as float32 array shape (n, d).
    """
    all_vecs: List[List[float]] = []

    for bi, batch in enumerate(_batched(texts, batch_size), start=1):
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vecs = [d.embedding for d in resp.data]
                all_vecs.extend(vecs)
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                wait = sleep_base * (2 ** (attempt - 1))
                print(f"[embed] batch {bi} error: {e} -> retry in {wait:.1f}s")
                time.sleep(wait)

    X = np.array(all_vecs, dtype="float32")
    if len(X.shape) != 2:
        raise RuntimeError("Embeddings retournés invalides (shape incorrect).")
    return X


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (X / norms).astype("float32")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    _load_env()

    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()
    use_cosine = os.getenv("GRAPH_INDEX_COSINE", "1").strip() not in {"0", "false", "False"}

    print(f"ENV: {ENV_PATH if ENV_PATH.exists() else '(no .env found)'}")
    print(f"Input profiles: {COMM_PROFILES_PATH}")
    print(f"Embed model: {embed_model}")
    print(f"Metric: {'cosine (IP on normalized vectors)' if use_cosine else 'L2'}")

    raw = _read_json(COMM_PROFILES_PATH)
    comms = _as_community_list(raw)

    # Build dataset (filter empty)
    meta: List[Dict[str, Any]] = []
    texts: List[str] = []
    dropped = 0

    for c in comms:
        cid = _community_id(c)
        if not cid:
            dropped += 1
            continue

        title = _safe_str(c.get("title"))
        summary = _safe_str(c.get("summary"))
        keywords = c.get("keywords") if isinstance(c.get("keywords"), list) else []

        text = _community_text(c)
        if not text.strip():
            dropped += 1
            continue

        meta.append(
            {
                "community_id": cid,
                "title": title,
                "summary": summary,
                "keywords": keywords[:50] if isinstance(keywords, list) else [],
                "nb_members": int(c.get("nb_members") or 0),
            }
        )
        texts.append(text)

    print(f"Communities raw: {len(comms)}")
    print(f"Communities indexed: {len(texts)} | dropped: {dropped}")

    if not texts:
        raise RuntimeError("Aucune communauté indexable. Vérifie communities_profiles.json.")

    # Embeddings
    client = _get_openai_client()
    X = _embed_texts(client, texts, model=embed_model, batch_size=64)

    n, d = X.shape
    print(f"Embeddings: n={n}, d={d}")

    # FAISS index
    OUT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    if use_cosine:
        X = _l2_normalize(X)
        index = faiss.IndexFlatIP(d)  # cosine similarity via inner product on normalized vectors
    else:
        index = faiss.IndexFlatL2(d)

    index.add(X)

    # Save
    faiss.write_index(index, str(OUT_INDEX_PATH))
    with open(OUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "index_path": str(OUT_INDEX_PATH),
                "meta_count": len(meta),
                "dim": d,
                "metric": "cosine" if use_cosine else "l2",
                "embed_model": embed_model,
                "items": meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("✅ Done.")
    print(f"FAISS index: {OUT_INDEX_PATH}")
    print(f"Meta:       {OUT_META_PATH}")


if __name__ == "__main__":
    main()
