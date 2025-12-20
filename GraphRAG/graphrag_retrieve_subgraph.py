# src/graphrag_retrieve_subgraph.py
import os
import re
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

TOP_COMMUNITIES = 3
TOP_ENTITIES = 8
TOP_TRIPLES = 40


def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_candidates(query: str) -> List[str]:
    """
    Simple heuristique (rapide) : extrait des candidats entités
    - tokens longs
    - groupes de mots avec majuscules / acronymes
    - termes fiscaux fréquents (IR, IS, TVA, CGI...)
    """
    q = _clean_text(query)

    # candidats par acronymes / mots clés
    base = re.findall(r"\b[A-Z]{2,}\b", q)  # IR, IS, TVA...
    base += re.findall(r"\b(?:imp[oô]t|taxe|TVA|IR|IS|CGI|dahir|loi|article)\b", q, flags=re.I)

    # candidats par mots >= 4 lettres
    base += [w for w in re.findall(r"[A-Za-zÀ-ÿ\-']+", q) if len(w) >= 4]

    # normalisation & uniques
    seen = set()
    out = []
    for x in base:
        x = _clean_text(x).strip(" ,;:.()[]{}")
        if not x:
            continue
        key = x.lower()
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out[:30]


def retrieve_graph_context(driver, user_query: str) -> Dict[str, Any]:
    candidates = _extract_candidates(user_query)

    # 1) Trouver entités par similarité sur label/aliases
    q_entities = """
    WITH $cands AS cands
    UNWIND cands AS c
    MATCH (e:Entity)
    WHERE toLower(e.label) CONTAINS toLower(c)
       OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) CONTAINS toLower(c))
    RETURN e.id AS id, e.label AS label, e.type AS type
    LIMIT $limit
    """
    with driver.session() as session:
        ent_rows = session.run(q_entities, cands=candidates, limit=TOP_ENTITIES).data()

    entity_ids = [r["id"] for r in ent_rows if r.get("id")]
    if not entity_ids:
        return {
            "entities": [],
            "communities": [],
            "triples": [],
            "graph_context_text": ""
        }

    # 2) Router vers communautés: celles qui contiennent le plus d’entités matchées
    q_comms = """
    MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community)
    WHERE e.id IN $ids
    WITH c, count(DISTINCT e) AS hit
    RETURN c.id AS id, c.title AS title, c.summary AS summary, c.keywords AS keywords, c.size AS size, hit
    ORDER BY hit DESC, coalesce(c.size,0) DESC
    LIMIT $k
    """
    with driver.session() as session:
        comm_rows = session.run(q_comms, ids=entity_ids, k=TOP_COMMUNITIES).data()

    comm_ids = [r["id"] for r in comm_rows if r.get("id")]
    if not comm_ids:
        comm_ids = []

    # 3) Extraire sous-graphe: triples impliquant les entités, filtré par communautés retenues
    q_triples = """
    MATCH (a:Entity)-[r:REL]->(b:Entity)
    WHERE a.id IN $ids OR b.id IN $ids
    OPTIONAL MATCH (a)-[:IN_COMMUNITY]->(ca:Community)
    OPTIONAL MATCH (b)-[:IN_COMMUNITY]->(cb:Community)
    WITH a,r,b,
         collect(DISTINCT ca.id) + collect(DISTINCT cb.id) AS comms
    WITH a,r,b, [x IN comms WHERE x IS NOT NULL] AS comms2
    WHERE size($comm_ids)=0 OR any(x IN comms2 WHERE x IN $comm_ids)
    RETURN a.id AS a_id, a.label AS a_label, a.type AS a_type,
           r.relation AS relation, r.evidence AS evidence, r.chunk_id AS chunk_id, r.confidence AS confidence,
           b.id AS b_id, b.label AS b_label, b.type AS b_type
    LIMIT $limit
    """
    with driver.session() as session:
        triple_rows = session.run(q_triples, ids=entity_ids, comm_ids=comm_ids, limit=TOP_TRIPLES).data()

    # Construire un texte compact (pour le LLM)
    lines = []
    lines.append("## GraphRAG: Entités détectées")
    for e in ent_rows:
        lines.append(f"- {e.get('label')} (type={e.get('type')}, id={e.get('id')})")

    lines.append("\n## GraphRAG: Communautés sélectionnées")
    for c in comm_rows:
        title = _clean_text(c.get("title"))
        summary = _clean_text(c.get("summary"))
        kw = c.get("keywords") or []
        lines.append(f"- C{c.get('id')} | {title}")
        if summary:
            lines.append(f"  summary: {summary}")
        if kw:
            lines.append(f"  keywords: {', '.join(kw[:12])}")

    lines.append("\n## GraphRAG: Triples (preuves)")
    for t in triple_rows:
        rel = t.get("relation") or "REL"
        ev = _clean_text(t.get("evidence"))
        ck = t.get("chunk_id")
        conf = t.get("confidence")
        lines.append(f"- ({t['a_label']}) -[{rel}]-> ({t['b_label']}) | chunk={ck} | conf={conf}")
        if ev:
            lines.append(f"  evidence: {ev}")

    graph_context_text = "\n".join(lines)

    return {
        "entities": ent_rows,
        "communities": comm_rows,
        "triples": triple_rows,
        "graph_context_text": graph_context_text
    }


def main():
    load_dotenv(ENV_PATH)
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD")
    if not pwd:
        raise RuntimeError("NEO4J_PASSWORD manquant dans .env")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    q = input("Question: ").strip()
    out = retrieve_graph_context(driver, q)
    print("\n" + out["graph_context_text"])

    driver.close()


if __name__ == "__main__":
    main()
