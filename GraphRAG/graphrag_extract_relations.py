# src/graphrag_extract_relations.py
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI

from config_cgi import ENV_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "json" / "cgi-2025_chunks.json"
ENTITIES_PATH = PROJECT_ROOT / "data" / "graph" / "entities" / "entities.jsonl"

OUT_DIR = PROJECT_ROOT / "data" / "graph" / "relations"
OUT_PATH = OUT_DIR / "relations.jsonl"

REL_MODEL = "gpt-4o-mini"  # rapide

SYSTEM = (
    "Tu es un expert en fiscalité marocaine. "
    "Tu extrais des RELATIONS juridiques/fiscales depuis un extrait du CGI. "
    "Tu réponds uniquement en JSON valide."
)

# Relations utiles et simples (pas besoin d’en inventer 50)
ALLOWED_RELATIONS = [
    "DEFINI_PAR", "APPLIQUE_A", "CONCERNE", "INCLUT",
    "EXONERE", "DEDUIT", "IMPOSE", "TAUX_DE", "CALCULE_SUR",
    "OBLIGE_A", "DECLARER", "PAYER", "DELAI_DE", "SANCTIONNE",
    "REFERENCE"
]

def load_chunks() -> Dict[int, Dict[str, Any]]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        arr = json.load(f)
    return {int(c["id"]): c for c in arr}

def iter_entities():
    with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_relations_one(client: OpenAI, chunk: Dict[str, Any], entities_obj: Dict[str, Any]) -> Dict[str, Any]:
    chunk_id = entities_obj["chunk_id"]
    title = entities_obj.get("title") or ""
    article = entities_obj.get("article")
    text = chunk.get("text") or ""

    entities = entities_obj.get("entities", [])
    # On ne garde que les entités “fiables”
    entities = [e for e in entities if (e.get("label") and (e.get("confidence", 0) >= 0.55))]

    prompt = f"""
Chunk:
- id: {chunk_id}
- title: {title}
- article: {article}

Texte:
{text}

Entités (déjà extraites):
{json.dumps(entities, ensure_ascii=False)}

Tâche:
1) Déduis UNIQUEMENT les relations explicites ou très clairement implicites dans le texte.
2) Utilise seulement ces types de relations: {ALLOWED_RELATIONS}
3) Chaque relation doit contenir:
   - "head": label entité source
   - "relation": type (liste ci-dessus)
   - "tail": label entité cible
   - "evidence": courte citation/fragment (<= 25 mots) prouvant la relation
   - "confidence": 0..1

Réponds EXACTEMENT avec ce JSON:
{{
  "chunk_id": {chunk_id},
  "source": "cgi-2025",
  "title": {json.dumps(title, ensure_ascii=False)},
  "article": {json.dumps(article, ensure_ascii=False)},
  "relations": [
    {{"head":"...", "relation":"...", "tail":"...", "evidence":"...", "confidence":0.0}}
  ]
}}
"""

    resp = client.chat.completions.create(
        model=REL_MODEL,
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
    chunks_by_id = load_chunks()

    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(obj.get("chunk_id"))
                except Exception:
                    pass
        print(f"↩️ Reprise: {len(done)} chunks déjà traités.")

    count = 0
    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for ent in iter_entities():
            cid = int(ent["chunk_id"])
            if cid in done:
                continue

            chunk = chunks_by_id.get(cid)
            if not chunk:
                continue

            try:
                obj = extract_relations_one(client, chunk, ent)
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
                if count % 30 == 0:
                    print(f"✅ {count} relations-chunks traités")
                time.sleep(0.05)
            except Exception as e:
                print(f"❌ chunk {cid} erreur: {e}")
                time.sleep(1.0)

    print(f"✅ Terminé: {OUT_PATH}")

if __name__ == "__main__":
    main()
