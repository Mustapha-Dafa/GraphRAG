import subprocess
import os
import sys
import json
import pandas as pd

ROOT_PATH = r"D:\final_graphrag\GraphRAG2" 

def repense_graphrag2(query,level) :
    cmd = [
            "graphrag", "query",
            "--root", ROOT_PATH,
            "--method", "global",
            "--query", query ,
            "--community-level" , str(level)
        ]
    result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                encoding='cp1252',
                errors='replace'
            )
    output = result.stdout
    return output



def repense_rag(query) :
    script_path = os.path.join("classic RAG", "ask_RAG.py")
    
    # Commande avec la question en argument
    cmd = [sys.executable, script_path, query, "--format", "text"]

    result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                encoding='cp1252',
                errors='replace'
            )
    output = result.stdout
    return output


def repense_graphrag(query) :
    script_path = os.path.join("GraphRAG", "ask_graphrag.py")
    
    # Commande avec la question en argument
    cmd = [sys.executable, script_path , query, "--format", "text"]
    result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                encoding='cp1252',
                errors='replace'
            )
    output = result.stdout
    return output


df=pd.read_csv("all_questions.csv",sep=";")

L=[]

for i in range(len(df)):
    question=df['question'][i]
    level=df['level'][i]
    print(f"Question{i+1}: {question}")
    L.append({
    "question": question,
    "level": level,
    "réponse_rag": repense_rag(question),
    "réponse_graphrag": repense_graphrag(question),
    "réponse_graphrag2_C0": repense_graphrag2(question,0),
    "réponse_graphrag2_C1": repense_graphrag2(question,1),
    "réponse_graphrag2_C2": repense_graphrag2(question,2)
    })
with open("resultats2.json", "w", encoding="utf-8") as f:
    json.dump(L, f, ensure_ascii=False, indent=2)






