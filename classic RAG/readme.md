# Classic RAG Implementation

This folder contains a classic Retrieval-Augmented Generation (RAG) system for querying the CGI 2025 regulatory document. It uses vector search with FAISS, OpenAI embeddings, cross-encoder reranking, and an OpenAI chat model to provide structured answers.

## Overview

The system processes a Markdown document into chunks, builds a FAISS vector index, and retrieves relevant chunks to answer user questions via a CLI interface.

## Files

- [`classic RAG/config_cgi.py`](classic RAG/config_cgi.py "classic RAG/config_cgi.py"): Configuration file defining paths, models, and parameters (e.g., OpenAI models, FAISS settings).
- [`classic RAG/build_chunks_from_markdown.py`](classic RAG/build_chunks_from_markdown.py "classic RAG/build_chunks_from_markdown.py"): Script to parse Markdown and create JSON chunks based on sections starting with "##".
- [`classic RAG/build_faiss_index.py`](classic RAG/build_faiss_index.py "classic RAG/build_faiss_index.py"): Builds and saves a FAISS index from chunk embeddings using OpenAI's embedding model.
- [`classic RAG/retriever_faiss.py`](classic RAG/retriever_faiss.py "classic RAG/retriever_faiss.py"): Implements chunk retrieval using FAISS search followed by cross-encoder reranking.
- [`classic RAG/engine_cgi.py`](classic RAG/engine_cgi.py "classic RAG/engine_cgi.py"): Core engine that constructs context from retrieved chunks and queries the OpenAI chat model for answers.
- [`classic RAG/ask_cgi_cli.py`](classic RAG/ask_cgi_cli.py "classic RAG/ask_cgi_cli.py"): Interactive CLI for posing questions and displaying responses with articles cited.
- [`classic RAG/ask_RAG.py`](classic RAG/ask_RAG.py "classic RAG/ask_RAG.py"): Command-line script for querying with output in JSON or text format.
- [`classic RAG/extract_cgi.py`](classic RAG/extract_cgi.py "classic RAG/extract_cgi.py"): (Details not fully provided; likely for additional extraction logic.)

## Usage

1. Configure environment variables in [`.env`](.env ) (e.g., OpenAI API key).
2. Run [`classic RAG/build_chunks_from_markdown.py`](classic RAG/build_chunks_from_markdown.py "classic RAG/build_chunks_from_markdown.py") to generate chunks.
3. Run [`classic RAG/build_faiss_index.py`](classic RAG/build_faiss_index.py "classic RAG/build_faiss_index.py") to create the FAISS index.
4. Use [`classic RAG/ask_cgi_cli.py`](classic RAG/ask_cgi_cli.py "classic RAG/ask_cgi_cli.py") or [`classic RAG/ask_RAG.py`](classic RAG/ask_RAG.py "classic RAG/ask_RAG.py") to query the system.

## Dependencies

- OpenAI API
- FAISS
- Sentence Transformers (for cross-encoder)
- Python libraries: [`dotenv`](.venv/Lib/site-packages/dotenv/__init__.py ), [`numpy`](.venv/Lib/site-packages/numpy/__init__.py ), [`pathlib`](/c:/Users/masta/AppData/Local/Programs/Python/Python311/Lib/pathlib.py ), etc.

See [`requirements.txt`](requirements.txt "requirements.txt") in the root for full dependencies.