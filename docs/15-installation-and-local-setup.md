# Installation And Local Setup

Last reviewed: 2026-02-10

## Summary
This page is a pragmatic "what to install" checklist for common AI/LLM work: calling hosted model APIs, running local models, and running local vector stores.

## Baseline (Most Projects)
### Git
- Install Git: https://git-scm.com/downloads

### Python (Recommended For Most LLM Tooling)
- If you want a fast, modern Python workflow, use `uv`.
- Install `uv`: https://github.com/astral-sh/uv
- Create a project and add dependencies:
```bash
uv init
uv venv
uv add openai
```

### Node.js (If You Build Web Apps Or TS SDK Integrations)
- Install Node.js (LTS): https://nodejs.org/en/download
- Prefer a lockfile and a modern package manager (`npm`, `pnpm`, or `yarn`).

### Docker (For Local Databases And Services)
- Install Docker: https://docs.docker.com/get-docker/
- Expect local stacks to use `docker compose` for repeatability.

## Hosted Model APIs (Quick Start)
### OpenAI
- Docs: https://platform.openai.com/docs/
- Python SDK: `pip install -U openai`
- JS/TS SDK: `npm install openai`

### Anthropic
- Docs: https://docs.anthropic.com/
- Python SDK: `pip install -U anthropic`

### Common Setup Pitfalls
- Never commit API keys. Use env vars (`OPENAI_API_KEY`, etc.) and a secrets manager in production.
- Log and store prompts/outputs carefully (redact secrets/PII).

## Local Models (Quick Start)

### Ollama (Easy)
Good for: local experimentation, quick demos, simple RAG prototypes.

- Linux install (As of 2026-02-10): `curl -fsSL https://ollama.com/install.sh | sh`
- Docs: https://docs.ollama.com/linux

### vLLM (Hard)
Good for: high-throughput serving on Linux with NVIDIA GPUs.

- Install: `pip install vllm`
- Docs: https://docs.vllm.ai/en/latest/getting_started/installation.html
- Docker deployment docs: https://docs.vllm.ai/en/latest/deployment/docker/

### SGLang (Hard)
Good for: performance-focused serving; check docs for supported backends.

- Install (As of 2026-02-10): `uv pip install sglang`
- Docs: https://docs.sglang.io/get_started/install.html

## Vector Stores (Quick Start)

### Postgres + pgvector (Medium)
Good for: teams already running Postgres; SQL + vectors in one place.

- Docs: https://github.com/pgvector/pgvector
- Docker image (tags change; check docs): https://hub.docker.com/r/pgvector/pgvector

### Qdrant (Easy)
Good for: simple local dev and production-ready vector search.

- Docker: `docker run -p 6333:6333 qdrant/qdrant`
- Docs: https://qdrant.tech/documentation/quick-start/

### Chroma (Easy)
Good for: local dev and small deployments.

- Docker: `docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma`
- Docs: https://docs.trychroma.com/production/containers/docker

### Weaviate (Medium)
Good for: feature-rich vector DB; typically configured via Docker Compose.

- Docs: https://docs.weaviate.io/weaviate/installation/docker-compose

### Milvus (Hard)
Good for: large-scale vector search; heavier operational footprint.

- Docs: https://milvus.io/docs/install_standalone-docker.md/

## Observability (Quick Start)
### Langfuse (Medium)
Good for: prompt/trace tracking and evaluation workflows (self-hosted option).

- Self-hosting docs: https://github.com/langfuse/langfuse

### OpenTelemetry (Medium)
Good for: standard tracing across your services; GenAI semantics are evolving.

- GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
