# src/graphrag_summarize_communities_openai.py
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- OpenAI SDK (compatible new/old) ---
# Ref: openai-python library + API reference
# https://github.com/openai/openai-python :contentReference[oaicite:0]{index=0}
# https://platform.openai.com/docs/api-reference/chat/create?lang=python :contentReference[oaicite:1]{index=1}
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def _get_openai_client():
    from openai import OpenAI  # type: ignore
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ROOT = Path(__file__).resolve().parents[1]
COMM_PATH = ROOT / "data" / "graph" / "communities" / "communities.json"

OUT_PROFILES = ROOT / "data" / "graph" / "communities" / "communities_profiles.json"
OUT_SELECTION = ROOT / "data" / "graph" / "communities" / "communities_selection.json"

# Paramètres "benchmark"
COVERAGE_TARGET = 0.85
N_CAP = 500
MIN_SIZE = 5

# Modèles (tu peux changer)
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
# keywords = 8-12 maximum (par communauté)
KW_MIN = 6
KW_MAX = 12

def load_communities(path: Path) -> Dict[str, Any]:
    data = json.load(open(path, "r", encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("communities.json doit être un dict {community_id: ...}")
    return data

def extract_member_ids(comm_val: Any) -> List[str]:
    """
    Supporte plusieurs formats possibles:
    - dict: { "members": [...]} ou { "nodes": [...]} etc.
    - list: [...]
    - str: (cas rare) => on ne peut pas
    """
    if isinstance(comm_val, list):
        return [str(x) for x in comm_val]
    if isinstance(comm_val, dict):
        for k in ["members", "nodes", "node_ids", "entities", "items"]:
            if k in comm_val and isinstance(comm_val[k], list):
                return [str(x) for x in comm_val[k]]
        # parfois dict { "ent_x": true, ... }
        # => on prend les clés
        return [str(k) for k in comm_val.keys()]
    return []

def choose_topN(comm_map: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # build sizes
    rows = []
    all_members = 0
    for cid, val in comm_map.items():
        members = extract_member_ids(val)
        size = len(set(members))
        if size >= MIN_SIZE:
            rows.append({"community_id": str(cid), "nb_members": size})
            all_members += size

    rows.sort(key=lambda x: x["nb_members"], reverse=True)

    # choose N by cumulative coverage, capped
    cum = 0
    selected = []
    for r in rows:
        if len(selected) >= N_CAP:
            break
        selected.append(r)
        cum += r["nb_members"]
        if all_members > 0 and (cum / all_members) >= COVERAGE_TARGET:
            break

    stats = {
        "total_communities_raw": len(comm_map),
        "total_communities_kept_min_size": len(rows),
        "min_size": MIN_SIZE,
        "coverage_target": COVERAGE_TARGET,
        "cap": N_CAP,
        "selected_N": len(selected),
        "approx_coverage_over_kept": (cum / all_members) if all_members else 0.0,
        "kept_total_members_sum": all_members,
        "selected_members_sum": cum,
    }
    return selected, stats

def load_nodes_labels() -> Dict[str, str]:
    """
    Optionnel mais utile:
    - si on a graph_nodes_v2.json: {id,label,type,...}
    - sinon on fait sans (on résume quand même).
    """
    candidates = [
        ROOT / "data" / "graph" / "graph_nodes_v2.json",
        ROOT / "data" / "graph" / "graph_nodes.json",
    ]
    for p in candidates:
        if p.exists():
            data = json.load(open(p, "r", encoding="utf-8"))
            if isinstance(data, list):
                out = {}
                for n in data:
                    if isinstance(n, dict):
                        nid = n.get("id")
                        lab = n.get("label")
                        if nid and lab:
                            out[str(nid)] = str(lab)
                return out
    return {}

def openai_generate_profile(client, community_id: str, labels: List[str]) -> Dict[str, Any]:
    """
    Génère title/summary/keywords à partir d'un échantillon de labels d'entités.
    """
    # limiter pour tokens
    labels = [l.strip() for l in labels if l and l.strip()]
    labels = list(dict.fromkeys(labels))  # unique, conserve ordre
    sample = labels[:120]  # suffisant

    system = (
        "Tu es un expert fiscal (CGI Maroc). "
        "Tu vas donner un titre court, un résumé clair, et des mots-clés. "
        "Sois précis et utile pour la recherche d'information."
    )
    user = f"""
Communauté ID: {community_id}

Voici une liste (échantillon) d'entités/termes appartenant à cette communauté:
{", ".join(sample)}

Tâche:
1) Donne un TITLE très court (5 à 12 mots max).
2) Donne un SUMMARY (3 à 6 phrases) qui décrit exactement le thème fiscal/juridique.
3) Donne KEYWORDS ({KW_MIN} à {KW_MAX} items) : mots-clés courts, sans phrases.

Réponds en JSON STRICT:
{{
  "title": "...",
  "summary": "...",
  "keywords": ["...", "..."]
}}
""".strip()

    # On utilise Chat Completions (stable) ; si tu veux Responses, je peux te le basculer après.
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    txt = resp.choices[0].message.content.strip()

    # robust JSON parsing
    try:
        return json.loads(txt)
    except Exception:
        # fallback: on encapsule brut
        return {"title": "", "summary": txt, "keywords": []}

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY manquant (mets-le dans .env et charge-le).")

    comm_map = load_communities(COMM_PATH)
    selected, stats = choose_topN(comm_map)

    node_labels_map = load_nodes_labels()

    # cache
    profiles = {}
    if OUT_PROFILES.exists():
        profiles = json.load(open(OUT_PROFILES, "r", encoding="utf-8"))
        if not isinstance(profiles, dict):
            profiles = {}

    client = _get_openai_client()

    # run
    for i, item in enumerate(selected, start=1):
        cid = item["community_id"]
        if cid in profiles and all(k in profiles[cid] for k in ["title", "summary", "keywords"]):
            continue  # déjà fait

        members = extract_member_ids(comm_map[cid])
        labels = []
        for mid in members:
            # si on a le mapping id->label (v2)
            if mid in node_labels_map:
                labels.append(node_labels_map[mid])
            else:
                # sinon on pousse l'id comme "signal" minimal
                labels.append(str(mid))

        # retry simple
        for attempt in range(1, 4):
            try:
                prof = openai_generate_profile(client, cid, labels)
                prof["community_id"] = cid
                prof["nb_members"] = item["nb_members"]
                profiles[cid] = prof

                # save incremental (important)
                OUT_PROFILES.parent.mkdir(parents=True, exist_ok=True)
                json.dump(profiles, open(OUT_PROFILES, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                break
            except Exception as e:
                if attempt == 3:
                    profiles[cid] = {
                        "community_id": cid,
                        "nb_members": item["nb_members"],
                        "title": "",
                        "summary": f"ERROR: {e}",
                        "keywords": []
                    }
                    json.dump(profiles, open(OUT_PROFILES, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                time.sleep(2 * attempt)

        if i % 20 == 0:
            print(f"[{i}/{len(selected)}] communautés traitées...")

    # selection file
    OUT_SELECTION.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"stats": stats, "selected": selected}, open(OUT_SELECTION, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("✅ Terminé.")
    print("Stats:", stats)
    print("Profiles:", OUT_PROFILES)
    print("Selection:", OUT_SELECTION)

if __name__ == "__main__":
    main()
