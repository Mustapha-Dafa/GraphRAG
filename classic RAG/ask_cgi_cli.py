# src/ask_cgi_cli.py

import json
from engine_cgi import ask_cgi


def main():
    print("==============================================")
    print(" Assistant CGI 2025 – Moteur RAG réglementaire")
    print(" (tape 'exit' ou 'quit' pour quitter)")
    print("==============================================\n")

    while True:
        question = input("Question CGI > ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("👋 Fin de session.")
            break

        # Appel du moteur RAG
        result = ask_cgi(question)

        # Nettoyage léger de la réponse textuelle (retours à la ligne, espaces)
        rt = result.get("reponse_textuelle", "")
        rt_clean = str(rt).strip()

        result["reponse_textuelle"] = rt_clean

        # --- Affichage lisible pour l'humain ---
        print("\n--- Réponse textuelle ---")
        print(rt_clean)

        print("\n--- Articles cités ---")
        articles = result.get("articles_cites", [])
        if not articles:
            print("- Aucun article cité.")
        else:
            for art in articles:
                article_code = art.get("article") or "?"
                titre = art.get("titre") or ""
                print(f"- {article_code} | {titre}")

        print("\n--- Chunks utilisés (debug) ---")
        chunk_ids = result.get("chunks_ids", [])
        print(chunk_ids if chunk_ids else "[]")

        # --- JSON final pour le front (clean) ---
        print("\n--- JSON complet (pour le front) ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
