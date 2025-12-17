# src/graphrag_make_ids_v2.py
import json
import re
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NODES_IN = ROOT / "data" / "graph" / "graph_nodes.json"
EDGES_IN = ROOT / "data" / "graph" / "graph_edges.json"

NODES_OUT = ROOT / "data" / "graph" / "graph_nodes_v2.json"
EDGES_OUT = ROOT / "data" / "graph" / "graph_edges_v2.json"


def _norm_label(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _make_id(label: str, ntype: str) -> str:
    """
    ID stable: type + hash(normalized label)
    """
    base = f"{_norm_label(label)}|{(ntype or '').strip().upper()}"
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:12]
    return f"ent_{h}"


def _relation_str(rel):
    """
    rel peut être str ou list (tu as déjà eu ce bug).
    """
    if rel is None:
        return ""
    if isinstance(rel, list):
        rel = rel[0] if rel else ""
    if not isinstance(rel, str):
        rel = str(rel)
    rel = rel.strip()
    rel = rel.upper()
    rel = re.sub(r"\s+", "_", rel)
    if not rel:
        rel = "REL"
    return rel


def main():
    nodes_raw = json.load(open(NODES_IN, "r", encoding="utf-8"))
    edges_raw = json.load(open(EDGES_IN, "r", encoding="utf-8"))

    # 1) nodes -> ajout id
    label_to_id = {}
    nodes_v2 = []

    for n in nodes_raw:
        label = (n.get("label") or "").strip()
        ntype = (n.get("type") or "ENTITY").strip().upper()

        if not label:
            continue

        nid = _make_id(label, ntype)

        # attention: plusieurs labels proches peuvent exister (casse/accents),
        # on mappe sur label original + label normalisé
        label_to_id[label] = nid
        label_to_id[_norm_label(label)] = nid

        nodes_v2.append({
            "id": nid,
            "label": label,
            "type": ntype,
            "aliases": n.get("aliases") or []
        })

    # 2) edges -> head/tail id + relation propre
    edges_v2 = []
    skipped_missing = 0

    for e in edges_raw:
        head = (e.get("head") or "").strip()
        tail = (e.get("tail") or "").strip()

        head_id = label_to_id.get(head) or label_to_id.get(_norm_label(head))
        tail_id = label_to_id.get(tail) or label_to_id.get(_norm_label(tail))

        if not head_id or not tail_id:
            skipped_missing += 1
            continue

        edges_v2.append({
            "head_id": head_id,
            "tail_id": tail_id,
            "relation": _relation_str(e.get("relation")),
            "confidence": float(e.get("confidence") or 0.0),
            "chunk_id": e.get("chunk_id"),
            "evidence": e.get("evidence") or ""
        })

    json.dump(nodes_v2, open(NODES_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(edges_v2, open(EDGES_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("✅ V2 export ok")
    print(f"- Nodes in:  {len(nodes_raw)}   -> Nodes v2: {len(nodes_v2)}  ({NODES_OUT})")
    print(f"- Edges in:  {len(edges_raw)}   -> Edges v2: {len(edges_v2)}  ({EDGES_OUT})")
    print(f"- Skipped edges (missing endpoints): {skipped_missing}")


if __name__ == "__main__":
    main()
