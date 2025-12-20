import json
import re
from pathlib import Path


def build_chunks_from_markdown(md_path: Path, source_id: str = "cgi-2025"):
    """
    Lit le fichier Markdown du CGI et construit des chunks.
    -> Chaque chunk correspond à une section commençant par '## '
       (jusqu'à la prochaine '## ' ou la fin du fichier).

    Champs de chaque chunk :
      - id      : numéro du chunk
      - source  : identifiant du document (ex: 'cgi-2025')
      - title   : texte du titre (sans '##')
      - text    : markdown complet de la section (titre + contenu)
      - article : 'ARTICLE X' si détecté dans le titre, sinon None
    """
    if not md_path.exists():
        raise FileNotFoundError(f"Fichier Markdown introuvable : {md_path}")

    content = md_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    chunks = []
    current_title = None
    current_lines = []

    # Regex pour détecter "Article 5", "ARTICLE 247A", etc.
    article_re = re.compile(r"(?i)\barticle\s+(\d+[A-Za-z]*)")

    def flush_current_section():
        nonlocal current_title, current_lines, chunks

        if current_title is None or not current_lines:
            return

        block_text = "\n".join(current_lines).rstrip()
        if not block_text.strip():
            return

        # Détection de l'article à partir du titre
        m = article_re.search(current_title)
        article = f"ARTICLE {m.group(1)}" if m else None

        chunk_id = len(chunks) + 1
        chunks.append(
            {
                "id": chunk_id,
                "source": source_id,
                "title": current_title,
                "text": block_text,
                "article": article,
            }
        )

        # Réinitialiser pour la prochaine section
        current_title = None
        current_lines = []

    for line in lines:
        stripped = line.strip()

        # Nouveau bloc de niveau "## " (mais pas "### ")
        if stripped.startswith("## ") and not stripped.startswith("### "):
            # on clôt la section en cours si elle existe
            flush_current_section()

            # nouveau titre : on enlève les "##"
            current_title = stripped.lstrip("#").strip()
            current_lines = [line]  # on garde le markdown exact (avec "## ...")
            continue

        # Si on n'a pas encore rencontré de "##", on ignore les lignes
        if current_title is None:
            continue

        # Sinon, on ajoute la ligne au bloc courant
        current_lines.append(line)

    # Dernière section à la fin du fichier
    flush_current_section()

    return chunks


def main():
    project_root = Path(__file__).resolve().parents[1]

    md_path = project_root / "data" / "markdown" / "cgi-2025.md"
    json_dir = project_root / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks_from_markdown(md_path, source_id="cgi-2025")

    output_path = json_dir / "cgi-2025_chunks.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(chunks)} chunks sauvegardés dans : {output_path}")

    if chunks:
        print("\n🧩 Exemple de premier chunk :")
        ex = chunks[0]
        for k, v in ex.items():
            print(f"- {k}: {repr(v) if isinstance(v, str) else v}")


if __name__ == "__main__":
    main()
