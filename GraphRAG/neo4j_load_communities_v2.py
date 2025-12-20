# src/neo4j_load_communities_v2.py
import os, json
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

COMM_PATH = ROOT / "data" / "graph" / "communities" / "communities.json"

def iter_communities(comm_data):
    """
    Supporte:
    - dict: {"0": {...}, "1": {...}}
    - list: [{...}, {...}]
    """
    if isinstance(comm_data, dict):
        for cid, c in comm_data.items():
            # cas où la valeur est déjà un dict communauté
            if isinstance(c, dict):
                c = dict(c)
                c.setdefault("id", str(cid))
                yield c
            else:
                # cas rare: valeur = liste/string -> on encapsule
                yield {"id": str(cid), "members": c}
    elif isinstance(comm_data, list):
        for c in comm_data:
            if isinstance(c, dict):
                yield c
    else:
        raise TypeError(f"Unsupported communities format: {type(comm_data)}")

def main():
    comm_data = json.load(open(COMM_PATH, "r", encoding="utf-8"))

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        # contrainte unique
        session.run("CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE")

        inserted = 0
        linked = 0

        for c in iter_communities(comm_data):
            cid = str(c.get("community_id") or c.get("id") or c.get("cid") or "")
            if not cid:
                continue

            title = (c.get("title") or c.get("name") or "").strip()
            summary = (c.get("summary") or "").strip()

            keywords = c.get("keywords") or c.get("key_terms") or []
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",") if k.strip()]
            if not isinstance(keywords, list):
                keywords = []

            members = c.get("members") or c.get("nodes") or c.get("entities") or []
            if isinstance(members, str):
                members = [members]
            if not isinstance(members, list):
                members = []

            session.run(
                """
                MERGE (c:Community {id:$id})
                SET c.title=$title, c.summary=$summary, c.keywords=$keywords
                """,
                id=cid, title=title, summary=summary, keywords=keywords
            )
            inserted += 1

            # Lien communauté -> entités (si on a des ids ent_xxx)
            for mid in members:
                mid = str(mid)
                session.run(
                    """
                    MATCH (e:Entity {id:$eid})
                    MATCH (c:Community {id:$cid})
                    MERGE (e)-[:IN_COMMUNITY]->(c)
                    """,
                    eid=mid, cid=cid
                )
                linked += 1

    driver.close()
    print(f"✅ Communities upserted: {inserted}")
    print(f"✅ Community links attempted: {linked}")

if __name__ == "__main__":
    main()
