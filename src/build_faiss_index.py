import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# ========= 1) CONFIG OPENAI =========

load_dotenv()  # lit le fichier .env à la racine

API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY manquant dans le fichier .env")

client = OpenAI(api_key=API_KEY)


# ========= 2) PIPELINE PRINCIPAL =========

def main():
    # chemins
    project_root = Path(__file__).resolve().parents[1]
    chunks_path = project_root / "data" / "json" / "cgi-2025_chunks.json"
    index_dir = project_root / "data" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # ---- Charger les chunks ----
    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"📦 {len(chunks)} chunks chargés.")

    texts = [c["text"] for c in chunks]

    # ---- Embeddings OpenAI (en batch) ----
    batch_size = 16  # <<--- plus petit pour rester sous 8192 tokens
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"→ Embedding batch {i}–{i + len(batch) - 1} ...")

        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )

        batch_embeddings = [d.embedding for d in resp.data]
        all_embeddings.extend(batch_embeddings)

    embeddings = np.array(all_embeddings, dtype="float32")
    print(f"✅ Embeddings shape : {embeddings.shape}")

    # ---- Construire l'index FAISS ----
    n_vectors, dim = embeddings.shape
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"✅ Index FAISS contient {index.ntotal} vecteurs.")

    # ---- Sauvegarder l'index ----
    index_path = index_dir / "cgi-2025_faiss.index"
    faiss.write_index(index, str(index_path))
    print(f"💾 Index sauvegardé : {index_path}")

    # ---- Sauvegarder les métadonnées ----
    metadata = [
        {
            "id": c["id"],
            "source": c.get("source"),
            "title": c.get("title"),
            "article": c.get("article"),
        }
        for c in chunks
    ]

    metadata_path = index_dir / "cgi-2025_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"💾 Métadonnées sauvegardées : {metadata_path}")
    print("🎉 Construction de l'index FAISS terminée.")


if __name__ == "__main__":
    main()
