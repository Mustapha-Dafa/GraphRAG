# src/neo4j_load_community_profiles.py
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

# ton fichier généré par graphrag_summarize_communities_openai.py
PROFILES_PATH = ROOT / "data" / "graph" / "communities" / "communities_profiles.json"

BATCH_SIZE = 300


def _load_profiles(path: Path) -> List[Dict[str, Any]]:
    data = json.load(open(path, "r", encoding="utf-8"))

    # Cas 1: dict {"0": {...}, "1": {...}}
    if isinstance(data, dict):
        out = []
        for k, v in data.items():
            if isinstance(v, dict):
                v = dict(v)
                v.setdefault("id", str(k))
                out.append(v)
        return out

    # Cas 2: list [{...}, {...}]
    if isinstance(data, list):
        out = []
        for obj in data:
            if isinstance(obj, dict):
                obj = dict(obj)
                # normalisation id en string si possible
                if "id" in obj and obj["id"] is not None:
                    obj["id"] = str(obj["id"])
                elif "community_id" in obj and obj["community_id"] is not None:
                    obj["id"] = str(obj["community_id"])
                out.append(obj)
        return out

    raise ValueError("Format communities_profiles.json non supporté (ni dict ni list).")


def _norm_keywords(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    if isinstance(x, str):
        # accepte "a, b, c"
        parts = [p.strip() for p in x.replace("\n", ",").split(",")]
        return [p for p in parts if p]
    return [str(x).strip()] if str(x).strip() else []


def _chunk(lst: List[Any], size: int) -> List[List[Any]]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def main():
    load_dotenv(ENV_PATH)

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD")

    if not pwd:
        raise RuntimeError("NEO4J_PASSWORD manquant dans .env")

    if not PROFILES_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable: {PROFILES_PATH}")

    profiles = _load_profiles(PROFILES_PATH)

    # normalisation + filtrage minimal
    rows = []
    for p in profiles:
        cid = p.get("id")
        if cid is None:
            continue
        title = (p.get("title") or p.get("name") or "").strip()
        summary = (p.get("summary") or "").strip()
        keywords = _norm_keywords(p.get("keywords"))

        # garde aussi stats si dispo
        size = p.get("size") or p.get("nb_members") or p.get("members_count")

        rows.append({
            "id": str(cid),
            "title": title,
            "summary": summary,
            "keywords": keywords,
            "size": int(size) if isinstance(size, (int, float)) and size is not None else None,
        })

    print(f"Loaded profiles rows: {len(rows)} from {PROFILES_PATH}")

    q = """
    UNWIND $rows AS row
    MERGE (c:Community {id: row.id})
    SET c.title = row.title,
        c.summary = row.summary,
        c.keywords = row.keywords
    FOREACH (_ IN CASE WHEN row.size IS NULL THEN [] ELSE [1] END |
        SET c.size = row.size
    )
    """

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as session:
        # (optionnel) index pour accélérer MERGE
        session.run("CREATE INDEX community_id IF NOT EXISTS FOR (c:Community) ON (c.id)")

        inserted = 0
        for batch in _chunk(rows, BATCH_SIZE):
            session.run(q, rows=batch)
            inserted += len(batch)

        # vérif
        count = session.run("""
            MATCH (c:Community)
            WHERE coalesce(c.title,'') <> '' OR coalesce(c.summary,'') <> '' OR size(coalesce(c.keywords, [])) > 0
            RETURN count(c) AS n
        """).single()["n"]

    driver.close()
    print(f"✅ Upsert terminé. Rows upserted={inserted}. Communities with profile fields={count}")


if __name__ == "__main__":
    main()
