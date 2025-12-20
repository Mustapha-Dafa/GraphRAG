# src/ask_graph_cli.py
import json
from engine_graph import ask_graph

def main():
    print("==============================================")
    print(" Assistant CGI 2025 – GraphRAG (GraphOnly)")
    print(" (tape 'exit' ou 'quit' pour quitter)")
    print("==============================================\n")

    while True:
        q = input("Question CGI > ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("👋 Fin de session.")
            break

        result = ask_graph(q)

        print("\n--- Réponse textuelle ---")
        print(result.get("reponse_textuelle", "").strip())

        print("\n--- Communities citées (debug) ---")
        print(result.get("communities_citees", []))

        print("\n--- JSON complet (pour le front) ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()
