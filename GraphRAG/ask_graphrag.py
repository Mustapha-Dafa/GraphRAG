import json
import sys
import argparse
from engine_graph import ask_graph

def main():
    parser = argparse.ArgumentParser(
        description='Assistant CGI 2025 – GraphRAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python ask_graph_cli.py "Quel est le taux de l'IS?"
  python ask_graph_cli.py "Quel est le taux de l'IS?" --format text
  python ask_graph_cli.py "Quelles sont les sanctions?" --format json
        """
    )
    
    parser.add_argument(
        'question', 
        type=str, 
        help='Question à poser au GraphRAG'
    )
    
    parser.add_argument(
        '--format', 
        choices=['json', 'text'], 
        default='json',
        help='Format de sortie: json (complet) ou text (réponse seule). Par défaut: json'
    )
    
    args = parser.parse_args()
    
    # Validation
    question = args.question.strip()
    if not question:
        print("❌ Erreur: La question ne peut pas être vide")
        sys.exit(1)
    
    # Appel du moteur GraphRAG
    result = ask_graph(question)
    
    # Nettoyage léger de la réponse textuelle
    rt = result.get("reponse_textuelle", "")
    rt_clean = str(rt).strip()
    result["reponse_textuelle"] = rt_clean
    
    # Affichage selon le format demandé
    if args.format == 'json':
        # Format JSON complet
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # Format texte simple (réponse uniquement)
        print(rt_clean)

if __name__ == "__main__":
    main()