import os
from openai import OpenAI
import networkx as nx
from cdlib import algorithms
import os 
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Function to read the content of each document from the example_text directory
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

client = OpenAI(api_key=os.getenv("GRAPHRAG_API_KEY"))

# 1. Source Documents → Text Chunks
def split_documents_into_chunks(documents, chunk_size, overlap_size):
    chunks = []
    for document in documents:
        for i in range(0, len(document), chunk_size - overlap_size):
            chunk = document[i:i + chunk_size]
            chunks.append(chunk)
    return chunks

def extraction_prompt(prompt_path,input_text,tuple_delimiter="<|>",record_delimiter="##",completion_delimiter="<|COMPLETE|>"):
    with open(prompt_path ,"r") as f :
        prompt=f.read()
    prompt = str(prompt)
    return prompt.format(
        input_text=input_text,
        tuple_delimiter=tuple_delimiter,
        record_delimiter=record_delimiter,
        completion_delimiter=completion_delimiter,
    )
prompt_path = "prompts\extract_graph.txt"

# 2. Text Chunks → Element Instances
def extract_elements_from_chunks(chunks):
    elements = []
    for index, chunk in enumerate(chunks):
        print(f"Chunk index {index} of {len(chunks)}:")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract entities and relationships from the following text."},
                {"role": "user", "content": extraction_prompt(prompt_path,chunk)}
            ]
        )
        print(response.choices[0].message.content)
        entities_and_relations = response.choices[0].message.content
        elements.append(entities_and_relations)
    return elements


# 3. Element Instances → Element Summaries
def summarize_elements(elements):
    summaries = []
    for index, element in enumerate(elements):
        print(f"Element index {index} of {len(elements)}:")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following entities and relationships in a structured format. Use \"->\" to represent relationships, after the \"Relationships:\" word."},
                {"role": "user", "content": element}
            ]
        )
        print("Element summary:", response.choices[0].message.content)
        summary = response.choices[0].message.content
        summaries.append(summary)
    return summaries

# 4. Element Summaries → Graph Communities
def build_graph_from_summaries(summaries):
    G = nx.Graph()
    for index, summary in enumerate(summaries):
        print(f"Summary index {index} of {len(summaries)}:")
        lines = summary.split("\n")
        entities_section = False
        relationships_section = False
        entities = []
        for line in lines:
            if line.startswith("### Entities:") or line.startswith("**Entities:**"):
                entities_section = True
                relationships_section = False
                continue
            elif line.startswith("### Relationships:") or line.startswith("**Relationships:**"):
                entities_section = False
                relationships_section = True
                continue
            if entities_section and line.strip():
                if line[0].isdigit() and line[1] == ".":
                    line = line.split(".", 1)[1].strip()
                entity = line.strip()
                entity = entity.replace("**", "")
                entities.append(entity)
                G.add_node(entity)
            elif relationships_section and line.strip():
                parts = line.split("->")
                if len(parts) >= 2:
                    source = parts[0].strip()
                    target = parts[-1].strip()
                    relation = " -> ".join(parts[1:-1]).strip()
                    G.add_edge(source, target, label=relation)
    return G


# 5. Graph Communities → Community Summaries
def detect_communities(graph):
    communities = []
    index = 0
    for component in nx.connected_components(graph):
        print(
            f"Component index {index} of {len(list(nx.connected_components(graph)))}:")
        subgraph = graph.subgraph(component)
        if len(subgraph.nodes) > 1:  # Leiden algorithm requires at least 2 nodes
            try:
                sub_communities = algorithms.leiden(subgraph)
                for community in sub_communities.communities:
                    communities.append(list(community))
            except Exception as e:
                print(f"Error processing community {index}: {e}")
        else:
            communities.append(list(subgraph.nodes))
        index += 1
    print("Communities from detect_communities:", communities)
    return communities

def summarize_communities(communities, graph):
    community_summaries = []
    for index, community in enumerate(communities):
        print(f"Summarize Community index {index} of {len(communities)}:")
        subgraph = graph.subgraph(community)
        nodes = list(subgraph.nodes)
        edges = list(subgraph.edges(data=True))
        description = "Entities: " + ", ".join(nodes) + "\nRelationships: "
        relationships = []
        for edge in edges:
            relationships.append(
                f"{edge[0]} -> {edge[2]['label']} -> {edge[1]}")
        description += ", ".join(relationships)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following community of entities and relationships."},
                {"role": "user", "content": description}
            ]
        )
        summary = response.choices[0].message.content.strip()
        community_summaries.append(summary)
    return community_summaries

# Putting It All Together
def graph_rag_index(documents, chunk_size=2500 ,overlap_size=500):
    # Step 1: Split documents into chunks
    chunks = split_documents_into_chunks(documents, chunk_size, overlap_size)

    # Step 2: Extract elements from chunks
    elements = extract_elements_from_chunks(chunks)
    entities = []
    relationships = []
    for i in range(len(elements)):
    # Parser les données
        lines = elements[i].split('##\n')
        for line in lines:
            if not line.strip():
                continue
            
            # Diviser par le séparateur <|>
            parts = line.split('<|>')
            
            # Vérifier si c'est une entité
            if '"entity"' in parts[0]:
                entities.append({
                    'name': parts[1],
                    'type': parts[2],
                    'description': parts[3].rstrip(')')
                })
            elif '"relationship"' in parts[0]:
                weight = parts[4].rstrip(')') if len(parts) > 4 else None
                relationships.append({
                    'source': parts[1],
                    'target': parts[2],
                    'description': parts[3],
                    'weight': weight
                })
    df_entities = pd.DataFrame(entities)
    df_relationships = pd.DataFrame(relationships)
    df_entities.to_csv('output/entities.csv', index=False)
    df_relationships.to_csv('output/relationships.csv', index=False)

    # Step 3: Summarize elements
    summaries = summarize_elements(elements)
    

    # Step 4: Build graph and detect communities
    graph = build_graph_from_summaries(summaries)
    print("graph:", graph)
    communities = detect_communities(graph)
    
    print("communities:", communities[0])
    # Step 5: Summarize communities
    community_summaries = summarize_communities(communities, graph)


    # Step 6: Generate answers from community summaries
    # final_answer = generate_answers_from_communities(
    #     community_summaries, query)

    return community_summaries,elements ,summaries ,graph ,communities

