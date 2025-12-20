# src/engine_graph.py
import json
import re
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI

from config_graph import ENV_PATH, OPENAI_CHAT_MODEL, SOURCE_NAME_GRAPH, TOP_K_COMMUNITIES, K_CANDIDATES
from retriever_graph import search_communities

load_dotenv(ENV_PATH)
client = OpenAI()

def _safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.M)
        i, j = cleaned.find("{"), cleaned.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(cleaned[i:j+1])
    except Exception:
        pass
    return {
        "type_reponse": "graphrag",
        "reponse_textuelle": text,
        "source_document": SOURCE_NAME_GRAPH,
        "communities_citees": [],
    }

def _build_context_graph(question: str):
    results = search_communities(question, k_candidates=K_CANDIDATES)
    if not results:
        return "", []

    top = results[:TOP_K_COMMUNITIES]

    # context_data style "table" : chaque community est un enregistrement avec source_id = community_id
    rows = []
    comm_ids = []
    for r in top:
        c = r["community"]
        cid = c.get("community_id") or c.get("id")
        title = c.get("title", "")
        summary = c.get("summary", "")
        keywords = c.get("keywords", [])
        if isinstance(keywords, list):
            keywords = ", ".join(keywords)

        comm_ids.append(cid)
        rows.append(
            f"- source_id: {cid}\n"
            f"  title: {title}\n"
            f"  keywords: {keywords}\n"
            f"  summary: {summary}\n"
        )

    context_data = "\n".join(rows)
    return context_data, comm_ids

def ask_graph(question: str) -> Dict[str, Any]:
    context_data, comm_ids = _build_context_graph(question)

    if not context_data:
        return {
            "type_reponse": "graphrag",
            "reponse_textuelle": "Les communautés extraites ne contiennent pas suffisamment d'information pour répondre.",
            "source_document": SOURCE_NAME_GRAPH,
            "communities_citees": [],
        }

    response_type = "Réponse structurée en markdown avec sections. Style clair et professionnel."

    system_msg = (
        "---Rôle---\n"
        "Vous êtes un assistant utile qui répond aux questions concernant le droit fiscal marocain (CGI).\n"
        "Vous devez vous baser uniquement sur les données fournies.\n\n"
        "---Format---\n"
        "Répondez en JSON valide uniquement (un seul objet)."
    )

    user_prompt = f"""
---Objectif---
Générez une réponse de la longueur et du format ciblés qui répond à la question de l'utilisateur,
en résumant toutes les informations pertinentes dans les tableaux de données d'entrée.

Si vous ne connaissez pas la réponse ou si les données ne suffisent pas, dites-le. N'inventez rien.

Les points soutenus par les données doivent lister leurs références :
"... [Data: Sources (identifiants)]."
Ne listez pas plus de 5 identifiants. Si plus, ajoutez "+more".

---Longueur et format de réponse ciblés---
{response_type}

---Tableaux de données---
{context_data}

---Question utilisateur---
{question}

---Sortie JSON (STRICT)---
Retournez exactement un objet JSON :

{{
  "type_reponse": "graphrag",
  "reponse_textuelle": "Réponse en markdown + citations [Data: Sources (...)]",
  "source_document": "{SOURCE_NAME_GRAPH}",
  "communities_citees": [1, 2, 3]
}}
"""

    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = resp.choices[0].message.content or ""
    payload = _safe_parse_json(raw)

    payload.setdefault("type_reponse", "graphrag")
    payload.setdefault("source_document", SOURCE_NAME_GRAPH)
    if not payload.get("communities_citees"):
        payload["communities_citees"] = comm_ids

    return payload
