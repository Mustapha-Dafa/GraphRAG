# GraphRAG2 Implementation

This folder contains an implementation using the Microsoft GraphRAG library for a complete GraphRAG pipeline on the CGI 2025 regulatory document. It handles text extraction, entity/relation extraction, graph construction, community detection, summarization, and both local and global search capabilities, storing data in LanceDB.

## Overview

GraphRAG2 leverages the official Microsoft GraphRAG library to process input text, build a knowledge graph, generate community reports, and enable advanced querying with local (vector-based) and global (map-reduce) search strategies. It uses OpenAI for LLM tasks and embeddings.

## Directory Structure

### Configuration
- [`.env`](GraphRAG2/.env): Environment variables, including OpenAI API key and other secrets.
- [`settings.yaml`](GraphRAG2/settings.yaml): YAML configuration file for GraphRAG settings, such as models, storage paths, and pipeline parameters.

### Input and Output
- [`input/cgi-2025.txt`](GraphRAG2/input/cgi-2025.txt): Input text file containing the CGI 2025 document.
- [`output/`](GraphRAG2/output/): Directory containing output files:
  - `context.json`: Context data for queries.
  - `stats.json`: Statistics from the pipeline run.
  - `lancedb/`: LanceDB database files for storing graph and vector data.

### Cache
- [`cache/`](GraphRAG2/cache/): Caches intermediate results to speed up re-runs:
  - `community_reporting/`: Cached community report generations (e.g., chat logs for each community).
  - `extract_graph/`: Cached graph extraction data.
  - `summarize_descriptions/`: Cached entity/relation summarizations.
  - `text_embedding/`: Cached text embeddings.

### Logs
- [`logs/`](GraphRAG2/logs/): Directory for log files generated during pipeline execution.

### Prompts
- [`prompts/`](GraphRAG2/prompts/): Custom prompt templates for various GraphRAG stages:
  - `basic_search_system_prompt.txt`: Prompt for basic search.
  - `community_report_graph.txt`: Prompt for generating community reports from graph data.
  - `community_report_text.txt`: Prompt for generating community reports from text.
  - `drift_reduce_prompt.txt`: Prompt for drift reduction in search.
  - `drift_search_system_prompt.txt`: System prompt for drift-aware search.
  - `extract_claims.txt`: Prompt for extracting claims.
  - `extract_graph.txt`: Prompt for extracting graph elements (entities, relations).
  - `global_search_knowledge_system_prompt.txt`: System prompt for global search knowledge integration.
  - `global_search_map_system_prompt.txt`: Map prompt for global search.
  - `global_search_reduce_system_prompt.txt`: Reduce prompt for global search.
  - `local_search_system_prompt.txt`: System prompt for local search.
  - `question_gen_system_prompt.txt`: Prompt for question generation.
  - `summarize_descriptions.txt`: Prompt for summarizing descriptions.

### Existing README
- [`readme.md`](GraphRAG2/readme.md): Existing README file (may contain additional notes or instructions).

## Usage

1. Install dependencies from [`requirements.txt`](requirements.txt) in the root.
2. Configure [`.env`](GraphRAG2/.env) with your OpenAI API key.
3. Adjust [`settings.yaml`](GraphRAG2/settings.yaml) for your needs (e.g., model names, storage paths).
4. Place input text in [`input/cgi-2025.txt`](GraphRAG2/input/cgi-2025.txt).
5. Run the GraphRAG pipeline using the library's CLI or scripts (e.g., `graphrag index` command if installed globally).
6. Query the system using GraphRAG's query commands (local or global search), outputting results to [`output/`](GraphRAG2/output/).

For detailed CLI usage, refer to the Microsoft GraphRAG documentation.

## Dependencies

- Microsoft GraphRAG library
- OpenAI API
- LanceDB
- Python libraries: [`dotenv`](.venv/Lib/site-packages/dotenv/__init__.py), [`pyyaml`](.venv/Lib/site-packages/yaml/__init__.py), etc.

See [`requirements.txt`](requirements.txt) in the root for full dependencies.

## Notes

- The cache directories help avoid re-computation; delete them to force a full re-run.
- Logs in [`logs/`](GraphRAG2/logs/) can be monitored for errors or progress.
- Prompts can be customized for domain-specific improvements.
- Ensure sufficient OpenAI API quota for embeddings and LLM calls.
- Output in LanceDB allows for efficient querying and can be integrated with other tools.