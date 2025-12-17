# src/graphrag_extract_entities.py
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI

from config_cgi import ENV_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "json" / "cgi-2025_chunks.json"

OUT_DIR = PROJECT_ROOT / "data" / "graph" / "entities"
OUT_PATH = OUT_DIR / "entities.jsonl"

# Choix modèle : rapide + pas cher pour extraction
EXTRACT_MODEL = "gpt-4o-mini"

# Types d'entités (simple et utile)
ENTITY_TYPES = [
    "IMPOT", "PERSONNE", "REVENU", "TAUX", "DEDUCTION", "ABATTEMENT", "EXONERATION",
    "PROCEDURE", "DELAI", "SANCTION", "OBLIGATION", "DOCUMENT", "ADMINISTRATION",
    "ACTIVITE", "BASE_IMPOSABLE", "ASSIETTE", "RECouvrement", "DECLARATION"
]

SYSTEM = (
    "Tu es un expert en fiscalité marocaine. "
    "Tu extrais des ENTITÉS JURIDIQUES depuis un extrait du CGI. "
    "Tu réponds uniquement en JSON valide."
)

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_chunks() -> List[Dict[str, Any]]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_entities_one(client: OpenAI, chunk: Dict[str, Any]) -> Dict[str, Any]:
    chunk_id = chunk.get("id")
    title = chunk.get("title") or ""
    article = chunk.get("article")
    text = chunk.get("text") or ""

    prompt = f"""
Extrait CGI (chunk):
- id: {chunk_id}
- title: {title}
- article: {article}

Texte:
{text}

Tâche:
1) Extrais les entités importantes (juridiques/fiscales) présentes dans le texte.
2) Normalise: retire doublons, forme courte (ex: "Impôt sur le Revenu" -> "IR" si explicitement présent, sinon garde libellé).
3) Chaque entité doit avoir:
   - "label": string (ex: "Impôt sur le Revenu")
   - "type": un parmi {ENTITY_TYPES}
   - "aliases": liste (peut être vide)
   - "confidence": float 0..1

Réponds EXACTEMENT avec ce JSON:
{{
  "chunk_id": {chunk_id},
  "source": "cgi-2025",
  "title": {json.dumps(title, ensure_ascii=False)},
  "article": {json.dumps(article, ensure_ascii=False)},
  "entities": [
    {{"label":"...", "type":"...", "aliases":["..."], "confidence":0.0}}
  ]
}}
"""

    resp = client.chat.completions.create(
        model=EXTRACT_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)

def main():
    load_dotenv(ENV_PATH)
    client = OpenAI()

    ensure_dirs()
    chunks = load_chunks()
    print(f"📦 {len(chunks)} chunks chargés.")

    # Reprise: si fichier existe, on saute ceux déjà traités
    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(obj.get("chunk_id"))
                except Exception:
                    pass
        print(f"↩️ Reprise activée: {len(done)} chunks déjà traités.")

    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for i, ch in enumerate(chunks, start=1):
            cid = ch.get("id")
            if cid in done:
                continue

            try:
                obj = extract_entities_one(client, ch)
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                if i % 20 == 0:
                    print(f"✅ {i}/{len(chunks)}")
                time.sleep(0.05)  # petite pause anti-rate-limit
            except Exception as e:
                print(f"❌ chunk {cid} erreur: {e}")
                time.sleep(1.0)

    print(f"✅ Terminé: {OUT_PATH}")

if __name__ == "__main__":
    main()
