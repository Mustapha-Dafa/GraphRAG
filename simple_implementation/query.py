import os
from openai import OpenAI
import networkx as nx
from cdlib import algorithms
import os 
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from simple_implementation.index import graph_rag_index

client = OpenAI(api_key=os.getenv("GRAPHRAG_API_KEY"))

print(os.getenv("GRAPHRAG_API_KEY"))

def read_documents_from_files():
    documents = []
    directory = "input"
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents


# Read documents and store them in the DOCUMENTS list
DOCUMENTS = read_documents_from_files()

community_summaries,elements ,summaries ,graph ,communities= graph_rag_index(DOCUMENTS ,2000 ,400)



# 6. Community Summaries → Community Answers → Global Answer
def generate_answers_from_communities(community_summaries, query):
    intermediate_answers = []
    for index, summary in enumerate(community_summaries):
        print(f"Summary index {index} of {len(community_summaries)}:")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer the following query based on the provided summary."},
                {"role": "user", "content": f"Query: {query} Summary: {summary}"}
            ]
        )
        print("Intermediate answer:", response.choices[0].message.content)
        intermediate_answers.append(
            response.choices[0].message.content)

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
                "content": "Combine these answers into a final, concise response."},
            {"role": "user", "content": f"Intermediate answers: {intermediate_answers}"}
        ]
    )
    final_answer = final_response.choices[0].message.content
    return final_answer

query = input("Enter your query: ")
answer = generate_answers_from_communities(community_summaries, query)
print(answer)