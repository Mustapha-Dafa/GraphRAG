# src/neo4j_load_graph_v2.py
import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

NODES_PATH = ROOT / "data" / "graph" / "graph_nodes_v2.json"
EDGES_PATH = ROOT / "data" / "graph" / "graph_edges_v2.json"

BATCH = 500


def _rid(e: dict) -> str:
    """
    rid stable pour MERGE relationship
    """
    base = f"{e['head_id']}|{e['tail_id']}|{e.get('relation','')}|{e.get('chunk_id','')}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def main():
    load_dotenv(ROOT / ".env")

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")

    print("ENV LOADED FROM:", ROOT / ".env")
    print("NEO4J_URI =", uri)
    print("NEO4J_USER =", user)
    print("NEO4J_PASSWORD is set ?", bool(pwd))

    nodes = json.load(open(NODES_PATH, "r", encoding="utf-8"))
    edges = json.load(open(EDGES_PATH, "r", encoding="utf-8"))

    # Prepare edges with rid
    for e in edges:
        e["rid"] = _rid(e)
        if not e.get("relation"):
            e["relation"] = "REL"

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as session:
        # clean DB (option)
        session.run("MATCH (n) DETACH DELETE n")

        # constraints / indexes
        session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        session.run("CREATE INDEX entity_label IF NOT EXISTS FOR (e:Entity) ON (e.label)")
        session.run("CREATE INDEX rel_rid IF NOT EXISTS FOR ()-[r:REL]-() ON (r.rid)")

        # insert nodes
        q_nodes = """
        UNWIND $rows AS row
        MERGE (e:Entity {id: row.id})
        SET e.label = row.label,
            e.type = row.type,
            e.aliases = row.aliases
        """
        for i in range(0, len(nodes), BATCH):
            batch = nodes[i:i+BATCH]
            session.run(q_nodes, rows=batch)
        print("✅ Inserted nodes:", len(nodes))

        # insert edges
        q_edges = """
        UNWIND $rows AS row
        MATCH (a:Entity {id: row.head_id})
        MATCH (b:Entity {id: row.tail_id})
        MERGE (a)-[r:REL {rid: row.rid}]->(b)
        SET r.relation   = row.relation,
            r.confidence = row.confidence,
            r.chunk_id   = row.chunk_id,
            r.evidence   = row.evidence
        """
        inserted = 0
        for i in range(0, len(edges), BATCH):
            batch = edges[i:i+BATCH]
            session.run(q_edges, rows=batch)
            inserted += len(batch)
        print("✅ Inserted edges:", inserted)

        c1 = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        c2 = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        c3 = session.run("MATCH ()-[r:REL]->() RETURN count(r) AS c, count(r.relation) AS cr").single()

        print("Neo4j count(n) =", c1)
        print("Neo4j count(r) =", c2)
        print("Neo4j REL count =", c3["c"], " | REL with relation =", c3["cr"])

    driver.close()
    print("✅ Import terminé.")


if __name__ == "__main__":
    main()
