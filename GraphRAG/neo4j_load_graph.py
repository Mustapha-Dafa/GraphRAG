# src/neo4j_load_graph.py
import json
from pathlib import Path
from dotenv import load_dotenv
import os
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

NODES_PATH = "neo4j/import/graph_nodes_norm.json"
EDGES_PATH = "neo4j/import/graph_edges_norm.json"

BATCH = 500

ID_KEYS = ["id", "node_id", "entity_id", "uid", "key"]
SRC_KEYS = ["source", "src", "from", "source_id"]
TGT_KEYS = ["target", "dst", "to", "target_id"]
REL_KEYS = ["relation", "rel", "type", "predicate"]

def pick(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def normalize_nodes(nodes):
    out = []
    skipped = 0
    for n in nodes:
        nid = pick(n, ID_KEYS)
        label = n.get("label") or n.get("name") or n.get("text") or n.get("entity") or ""
        ntype = n.get("type") or n.get("entity_type") or "Entity"

        # si pas d'id, on SKIP (sinon MERGE impossible)
        if nid is None:
            skipped += 1
            continue

        out.append({"id": str(nid), "label": str(label), "type": str(ntype)})
    return out, skipped

def normalize_edges(edges):
    out = []
    skipped = 0
    for e in edges:
        s = pick(e, SRC_KEYS)
        t = pick(e, TGT_KEYS)
        rel = pick(e, REL_KEYS) or "RELATED_TO"
        if s is None or t is None:
            skipped += 1
            continue
        out.append({"source": str(s), "target": str(t), "relation": str(rel)})
    return out, skipped

def main():
    load_dotenv(ENV_PATH)

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd  = os.getenv("NEO4J_PASSWORD")

    print("ENV LOADED FROM:", ENV_PATH)
    print("NEO4J_URI =", uri)
    print("NEO4J_USER =", user)
    print("NEO4J_PASSWORD is set ?", bool(pwd))

    nodes_raw = json.load(open(NODES_PATH, "r", encoding="utf-8"))
    edges_raw = json.load(open(EDGES_PATH, "r", encoding="utf-8"))

    print(f"Loaded nodes: {len(nodes_raw)} from {NODES_PATH}")
    print(f"Loaded edges: {len(edges_raw)} from {EDGES_PATH}")

    nodes, skipped_nodes = normalize_nodes(nodes_raw)
    edges, skipped_edges = normalize_edges(edges_raw)
    print("Normalized nodes:", len(nodes), "| skipped(no id):", skipped_nodes)
    print("Normalized edges:", len(edges), "| skipped(bad endpoints):", skipped_edges)

    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    with driver.session(database="neo4j") as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")

        q_nodes = """
        UNWIND $rows AS row
        MERGE (e:Entity {id: row.id})
        SET e.label = row.label,
            e.type  = row.type
        """
        inserted_nodes = 0
        for batch in chunked(nodes, BATCH):
            session.run(q_nodes, rows=batch)
            inserted_nodes += len(batch)
        print("Inserted nodes:", inserted_nodes)

        q_edges = """
        UNWIND $rows AS row
        MATCH (a:Entity {id: row.source})
        MATCH (b:Entity {id: row.target})
        MERGE (a)-[:REL {type: row.relation}]->(b)
        """
        inserted_edges = 0
        for batch in chunked(edges, BATCH):
            session.run(q_edges, rows=batch)
            inserted_edges += len(batch)
        print("Inserted edges:", inserted_edges)

        n = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print("Neo4j count(n) =", n)
        print("Neo4j count(r) =", r)

    driver.close()
    print("✅ Import terminé.")

if __name__ == "__main__":
    main()
