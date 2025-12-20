# src/graphrag_refine_communities.py
import json
from pathlib import Path
from collections import Counter, defaultdict

import networkx as nx


# -----------------------
# Paths projet (adaptés à ta structure)
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = PROJECT_ROOT / "data" / "graph"
NODES_PATH = GRAPH_DIR / "graph_nodes.json"
EDGES_PATH = GRAPH_DIR / "graph_edges.json"

OUT_COMMUNITIES = GRAPH_DIR / "communities_v2.json"          # mapping node_id -> community_id
OUT_COMMUNITY_SUMMARY = GRAPH_DIR / "communities_v2_info.json"  # infos + thème + top nodes


# -----------------------
# Config qualité (tu peux ajuster après)
# -----------------------
USE_LEIDEN_IF_AVAILABLE = True

# resolution plus basse => moins de communautés (souvent mieux pour CGI)
RESOLUTION = 0.6

# taille minimale d'une communauté ; en dessous => fusion automatique
MIN_COMMUNITY_SIZE = 8

# nb de mots/entités pour thème
TOP_KEYWORDS = 10


# -----------------------
# Helpers
# -----------------------
def load_graph(nodes_path: Path, edges_path: Path) -> nx.Graph:
    nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
    edges = json.loads(edges_path.read_text(encoding="utf-8"))

    G = nx.Graph()

    for n in nodes:
        nid = str(n.get("id"))
        label = (n.get("label") or n.get("name") or "").strip()
        ntype = (n.get("type") or "ENTITY").strip()

        attrs = dict(n)
        attrs.pop("label", None)
        attrs.pop("type", None)
        attrs.pop("id", None)

        G.add_node(nid, label=label, type=ntype, **attrs)

    for e in edges:
        s = str(e.get("source") or e.get("head") or e.get("from") or "")
        t = str(e.get("target") or e.get("tail") or e.get("to") or "")
        rel = e.get("relation") or e.get("type") or "RELATED_TO"
        w = e.get("weight", 1.0)

        if s and t and s in G and t in G and s != t:
            G.add_edge(s, t, relation=str(rel), weight=float(w))

    return G


def detect_communities(G: nx.Graph, resolution: float = 0.6) -> dict:
    """
    Retourne: dict node_id -> community_id
    Leiden si dispo, sinon Louvain.
    """
    # 1) Leiden (meilleur)
    if USE_LEIDEN_IF_AVAILABLE:
        try:
            import igraph as ig
            import leidenalg

            # conversion NX -> iGraph
            nodes = list(G.nodes())
            index = {nid: i for i, nid in enumerate(nodes)}
            edges = [(index[u], index[v]) for u, v in G.edges()]

            g = ig.Graph(n=len(nodes), edges=edges, directed=False)

            # Leiden (modularity / CPM)
            # CPM marche bien avec resolution
            part = leidenalg.find_partition(
                g,
                leidenalg.CPMVertexPartition,
                resolution_parameter=float(resolution),
            )

            node_to_comm = {}
            for cid, members in enumerate(part):
                for m in members:
                    node_to_comm[nodes[m]] = int(cid)
            return node_to_comm

        except Exception:
            pass

    # 2) Louvain (fallback)
    try:
        import community as community_louvain  # python-louvain
        part = community_louvain.best_partition(G, resolution=float(resolution), weight="weight")
        return {str(k): int(v) for k, v in part.items()}
    except Exception as e:
        raise RuntimeError(
            "Impossible d'exécuter Leiden ou Louvain. "
            "Installe igraph/leidenalg ou python-louvain."
        ) from e


def merge_small_communities(G: nx.Graph, node_to_comm: dict, min_size: int = 8) -> dict:
    """
    Fusionne les communautés trop petites vers la communauté voisine la plus connectée.
    """
    comm_to_nodes = defaultdict(list)
    for n, c in node_to_comm.items():
        comm_to_nodes[c].append(n)

    small_comms = [c for c, ns in comm_to_nodes.items() if len(ns) < min_size]
    if not small_comms:
        return node_to_comm

    # Pour chaque petite communauté, on la fusionne
    for c in small_comms:
        nodes = comm_to_nodes[c]

        # Cherche la meilleure communauté cible (par connexions sortantes)
        score = Counter()
        for n in nodes:
            for nb in G.neighbors(n):
                c2 = node_to_comm.get(nb)
                if c2 is not None and c2 != c:
                    score[c2] += 1

        if not score:
            # pas de voisins externes => on laisse (rare)
            continue

        target = score.most_common(1)[0][0]
        for n in nodes:
            node_to_comm[n] = target

    return node_to_comm


def build_community_info(G: nx.Graph, node_to_comm: dict) -> dict:
    comm_to_nodes = defaultdict(list)
    for n, c in node_to_comm.items():
        comm_to_nodes[c].append(n)

    info = {}
    for cid, nodes in comm_to_nodes.items():
        labels = [G.nodes[n].get("label", "") for n in nodes if G.nodes[n].get("label")]
        types = [G.nodes[n].get("type", "") for n in nodes if G.nodes[n].get("type")]

        # mots dominants simples (top labels)
        top_labels = [x for x, _ in Counter(labels).most_common(TOP_KEYWORDS)]
        top_types = [x for x, _ in Counter(types).most_common(5)]

        # score "centralité" simple => top nœuds
        deg = sorted(((n, G.degree(n)) for n in nodes), key=lambda x: x[1], reverse=True)[:10]
        top_nodes = [{"id": n, "label": G.nodes[n].get("label", ""), "degree": d} for n, d in deg]

        theme = " | ".join(top_labels[:5]) if top_labels else "COMMUNAUTE"

        info[str(cid)] = {
            "community_id": int(cid),
            "size": int(len(nodes)),
            "theme": theme,
            "top_labels": top_labels,
            "top_types": top_types,
            "top_nodes": top_nodes,
        }
    return info


def main():
    if not NODES_PATH.exists() or not EDGES_PATH.exists():
        raise FileNotFoundError("graph_nodes.json / graph_edges.json introuvables dans data/graph/")

    print("📥 Chargement du graphe...")
    G = load_graph(NODES_PATH, EDGES_PATH)
    print(f"✅ Graphe: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")

    print("🧠 Détection des communautés (Leiden si dispo sinon Louvain)...")
    node_to_comm = detect_communities(G, resolution=RESOLUTION)

    # stats avant fusion
    before_sizes = Counter(node_to_comm.values())
    print(f"➡️ Communautés initiales: {len(before_sizes)}")

    print(f"🧩 Fusion des petites communautés (min_size={MIN_COMMUNITY_SIZE})...")
    node_to_comm = merge_small_communities(G, node_to_comm, min_size=MIN_COMMUNITY_SIZE)

    after_sizes = Counter(node_to_comm.values())
    print(f"✅ Communautés après fusion: {len(after_sizes)}")

    print("🏷️ Génération des thèmes + résumé...")
    info = build_community_info(G, node_to_comm)

    OUT_COMMUNITIES.parent.mkdir(parents=True, exist_ok=True)
    OUT_COMMUNITIES.write_text(json.dumps(node_to_comm, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_COMMUNITY_SUMMARY.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Sauvegardé :")
    print(f"   - {OUT_COMMUNITIES}")
    print(f"   - {OUT_COMMUNITY_SUMMARY}")


if __name__ == "__main__":
    main()
