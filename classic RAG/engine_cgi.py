# src/engine_cgi.py
import json
import re
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from config_cgi import (
    ENV_PATH,
    OPENAI_CHAT_MODEL,
    SOURCE_NAME,
    TOP_K,
)
from retriever_faiss import search_chunks

# Init OpenAI
load_dotenv(ENV_PATH)
client = OpenAI()


# ---------- Helpers JSON ----------

def _safe_parse_json(text: str) -> Dict[str, Any]:
    """
    Essaie de parser la réponse du LLM en JSON.
    Tolère la présence de ```json ... ``` etc.
    """
    text = text.strip()

    # 1) tentative directe
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) on enlève les fences ```json ... ```
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                         flags=re.IGNORECASE | re.MULTILINE)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end+1])
    except Exception:
        pass

    # 3) fallback : on renvoie un JSON minimal
    return {
        "type_reponse": "reglementaire",
        "reponse_textuelle": text,
        "articles_cites": [],
        "source_document": SOURCE_NAME,
        "chunks_ids": [],
    }


# ---------- Construction du contexte ----------

def _build_context(question: str):
    """
    Récupère les chunks pertinents et construit le bloc de contexte.
    """
    results = search_chunks(question, k=TOP_K)

    if not results:
        return "", [], []

    blocks = []
    articles = []
    chunk_ids = []

    for r in results:
        c = r["chunk"]
        article = c.get("article") or c.get("title") or ""
        title = c.get("title") or ""
        text = c.get("text") or ""

        articles.append({"article": article, "titre": title})
        chunk_ids.append(c.get("id"))

        # On “tag” chaque chunk avec son id => l'LLM peut citer [Data: Sources (id)]
        block = (
            f"source_id: {c.get('id')}\n"
            f"article: {article}\n"
            f"titre: {title}\n"
            f"texte:\n{text}"
        )
        blocks.append(block)

    context_str = "\n\n---\n\n".join(blocks)
    return context_str, articles, chunk_ids


# ---------- Moteur principal ----------

def ask_cgi(question: str) -> Dict[str, Any]:
    """
    Pose une question au moteur RAG et renvoie un JSON structuré.
    """
    context_str, articles, chunk_ids = _build_context(question)

    # Si aucun contexte pertinent
    if not context_str:
        return {
            "type_reponse": "reglementaire",
            "reponse_textuelle": (
                "Les extraits disponibles ne contiennent pas suffisamment "
                "d'information pour répondre à cette question."
            ),
            "articles_cites": [],
            "source_document": SOURCE_NAME,
            "chunks_ids": [],
        }

    # ✅ Prompt système (ton rôle)
    system_msg = (
        "---Rôle---\n"
        "Vous êtes un assistant utile qui répond aux questions concernant le droit fiscal marocain "
        "(Code Général des Impôts - CGI) à partir du contexte fourni.\n\n"
        "---Règle anti-hallucination---\n"
        "Répondez UNIQUEMENT avec des informations présentes dans le contexte. "
        "Si le contexte est insuffisant, dites-le clairement. N'inventez rien.\n\n"
        "---Format---\n"
        "Répondez en JSON valide uniquement (un seul objet JSON, pas de texte autour)."
    )

    # ✅ Prompt utilisateur (ton template adapté)
    # Ici, on n'a pas un vrai "tableau", mais on fournit un contexte structuré avec source_id.
    response_type = (
        "Réponse structurée en markdown avec sections (Définition, Principe, Points clés, Références). "
        "Style clair et professionnel."
    )

    user_prompt = f"""
---Objectif---
Générez une réponse de la longueur et du format ciblés qui répond à la question de l'utilisateur,
en résumant toutes les informations pertinentes dans le contexte fourni,
y compris les articles fiscaux applicables, taux, seuils et obligations si présents.

Si vous ne connaissez pas la réponse ou si le contexte ne contient pas suffisamment d'informations,
dites-le simplement. N'inventez rien.

---Règles de citation (OBLIGATOIRE)---
Chaque phrase factuelle doit citer ses preuves ainsi :
"… [Data: Sources (source_id1, source_id2)]"

- Ne listez pas plus de 5 identifiants dans une citation.
- Utilisez les identifiants "source_id" présents dans le contexte.
- Si plus de 5, mettez : (id1, id2, id3, id4, id5, +more)

---Longueur et format de réponse ciblés---
{response_type}

---Question utilisateur---
{question}

---Contexte (extraits CGI)---
{context_str}

---Consignes de sortie JSON---
Retournez un seul objet JSON EXACTEMENT sous cette forme :

{{
  "type_reponse": "reglementaire",
  "reponse_textuelle": "Votre réponse en markdown (avec les citations [Data: Sources (...)]).",
  "articles_cites": [
    {{"article": "...", "titre": "..."}}
  ],
  "source_document": "{SOURCE_NAME}",
  "chunks_ids": [1, 2, 3]
}}

- "reponse_textuelle" : en français, riche, clair, structuré (markdown), et avec citations.
- "articles_cites" : uniquement les articles réellement utilisés.
- "chunks_ids" : les ids des chunks réellement utilisés.
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

    # Compléter si le modèle oublie des champs
    payload.setdefault("type_reponse", "reglementaire")
    payload.setdefault("source_document", SOURCE_NAME)

    if not payload.get("articles_cites"):
        payload["articles_cites"] = articles

    if not payload.get("chunks_ids"):
        payload["chunks_ids"] = chunk_ids

    return payload
