# A股上市公司关联性RAG项目

A knowledge graph-based RAG system for analyzing relationships between A-share listed companies.

## Project Status

Under active development - implementing Story 1.1: Project Setup and Data Ingestion

## Technology Stack

- Python 3.10+
- HippoRAG (Graph RAG framework)
- LanceDB (Vector storage)
- Transformers & Sentence-Transformers
- FastAPI (API server)

## Setup

1. Install dependencies:
   ```bash
   uv sync --all-extras
   ```

2. Copy environment template:
   ```bash
   cp .env.example .env
   ```

3. Configure your API keys in `.env`

## Development

This project follows Test-Driven Development (TDD) practices.

Run tests:
```bash
pytest
```

Format code:
```bash
ruff format .
```