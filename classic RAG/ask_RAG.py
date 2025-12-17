# src/ask_cgi_cli.py
import json
import sys
import argparse
from engine_cgi import ask_cgi

def main():
    parser = argparse.ArgumentParser(description='Assistant CGI 2025')
    parser.add_argument('question', type=str, help='Question à poser au CGI')
    parser.add_argument('--format', choices=['json', 'text'], default='json',
                       help='Format de sortie (json ou text)')
    
    args = parser.parse_args()
    
    # Appel du moteur RAG
    result = ask_cgi(args.question)
    
    # Nettoyage
    rt = result.get("reponse_textuelle", "")
    rt_clean = str(rt).strip()
    result["reponse_textuelle"] = rt_clean
    
    if args.format == 'json':
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(rt_clean)

if __name__ == "__main__":
    main()