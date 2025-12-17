# GraphRAG Implementation

This folder contains a custom GraphRAG (Graph-based Retrieval-Augmented Generation) system for querying the CGI 2025 regulatory document. It extracts entities and relations using OpenAI, builds a knowledge graph with NetworkX, detects communities, generates summaries, and loads data into Neo4j for advanced retrieval and querying.

## Overview

The system processes text to extract entities and relations, constructs a graph, identifies communities, summarizes them, and enables graph-based search using Neo4j. It supports CLI and script-based querying with structured answers citing sources.

## Files

### Configuration and Setup
- [`GraphRAG/config_graph.py`](GraphRAG/config_graph.py): Configuration file defining paths, models, API keys, and parameters (e.g., OpenAI models, Neo4j credentials, graph settings).

### Data Extraction
- [`GraphRAG/graphrag_extract_entities.py`](GraphRAG/graphrag_extract_entities.py): Extracts entities from text chunks using OpenAI prompts.
- [`GraphRAG/graphrag_extract_relations.py`](GraphRAG/graphrag_extract_relations.py): Extracts relations between entities from text chunks using OpenAI prompts.
- [`GraphRAG/graphrag_make_ids_v2.py`](GraphRAG/graphrag_make_ids_v2.py): Generates unique IDs for entities and relations in version 2 format.

### Graph Construction and Processing
- [`GraphRAG/graphrag_build_graph_and_communities.py`](GraphRAG/graphrag_build_graph_and_communities.py): Builds the graph using NetworkX, detects communities (e.g., via Louvain method), and saves graph data.
- [`GraphRAG/graphrag_summarize_communities_openai.py`](GraphRAG/graphrag_summarize_communities_openai.py): Generates summaries for detected communities using OpenAI.
- [`GraphRAG/build_graph_index.py`](GraphRAG/build_graph_index.py): Builds and indexes the graph, possibly including FAISS for vector search on graph elements.

### Retrieval and Engine
- [`GraphRAG/retriever_graph.py`](GraphRAG/retriever_graph.py): Implements graph-based retrieval, querying subgraphs or communities relevant to the query.
- [`GraphRAG/graphrag_retrieve_subgraph.py`](GraphRAG/graphrag_retrieve_subgraph.py): Retrieves relevant subgraphs based on query embeddings or keywords.
- [`GraphRAG/engine_graph.py`](GraphRAG/engine_graph.py): Core engine that integrates retrieval with OpenAI chat model to generate answers from graph context.

### Neo4j Integration
- [`GraphRAG/neo4j_load_graph.py`](GraphRAG/neo4j_load_graph.py): Loads graph nodes and edges into Neo4j database.
- [`GraphRAG/neo4j_load_graph_v2.py`](GraphRAG/neo4j_load_graph_v2.py): Updated version for loading graph data into Neo4j.
- [`GraphRAG/neo4j_load_communities_v2.py`](GraphRAG/neo4j_load_communities_v2.py): Loads community data into Neo4j.
- [`GraphRAG/neo4j_load_community_profiles.py`](GraphRAG/neo4j_load_community_profiles.py): Loads community profiles and summaries into Neo4j.

### Querying Interfaces
- [`GraphRAG/ask_graph_cli.py`](GraphRAG/ask_graph_cli.py): Interactive CLI for posing questions and displaying graph-based responses with citations.
- [`GraphRAG/ask_graphrag.py`](GraphRAG/ask_graphrag.py): Command-line script for querying the GraphRAG system, outputting results in JSON or text format.

## Usage

1. Configure environment variables in [`.env`](.env) (e.g., OpenAI API key, Neo4j URI, credentials).
2. Run extraction scripts: [`GraphRAG/graphrag_extract_entities.py`](GraphRAG/graphrag_extract_entities.py) and [`GraphRAG/graphrag_extract_relations.py`](GraphRAG/graphrag_extract_relations.py) to extract data.
3. Build the graph: [`GraphRAG/graphrag_build_graph_and_communities.py`](GraphRAG/graphrag_build_graph_and_communities.py).
4. Summarize communities: [`GraphRAG/graphrag_summarize_communities_openai.py`](GraphRAG/graphrag_summarize_communities_openai.py).
5. Load into Neo4j: Use the `neo4j_load_*.py` scripts.
6. Query via [`GraphRAG/ask_graph_cli.py`](GraphRAG/ask_graph_cli.py) or [`GraphRAG/ask_graphrag.py`](GraphRAG/ask_graphrag.py).

## Dependencies

- OpenAI API
- Neo4j
- NetworkX
- FAISS
- Sentence Transformers
- Python libraries: [`dotenv`](.venv/Lib/site-packages/dotenv/__init__.py), [`numpy`](.venv/Lib/site-packages/numpy/__init__.py), [`pathlib`](/c:/Users/masta/AppData/Local/Programs/Python/Python311/Lib/pathlib.py), etc.

See [`requirements.txt`](requirements.txt) in the root for full dependencies.

## Notes

- Ensure Neo4j is running and configured correctly.
- The system uses OpenAI for LLM tasks; monitor API usage.
- Graph data is stored in JSON files in [`data/graph/`](data/graph/).
- For advanced queries, leverage Neo4j's Cypher language via the retrieval scripts.