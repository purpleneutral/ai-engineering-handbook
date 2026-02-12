# Installation And Local Setup

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](16-stacks-and-difficulty.md) | [Next](01-llm-fundamentals.md)

## Summary
This page is a pragmatic "what to install" checklist for common AI/LLM work: calling hosted model APIs, running local models, and running local vector stores.

## See Also
- [Stacks And Difficulty](16-stacks-and-difficulty.md)
- [Ops: Shipping And Running LLM Systems](08-ops.md)
- [Tool Index (Curated)](../README.md#tool-index-curated)

## Baseline (Most Projects)

Regardless of whether you are calling a hosted API, running a local model, or building a full RAG pipeline, there is a common foundation that nearly every AI project shares. Getting these baseline tools installed and configured correctly saves time on every project that follows. The choices here are deliberately conservative -- Git, Python, Node.js, and Docker are not exciting, but they are the substrate that everything else builds on.

### Git
- Install Git: https://git-scm.com/downloads

### Python (Recommended For Most LLM Tooling)

Python remains the default language for LLM tooling. The major model providers ship Python SDKs first, most open-source AI libraries are Python-native, and the ecosystem for data processing, evaluation, and experimentation is unmatched. If you want a fast, modern Python workflow, `uv` is worth adopting -- it handles project creation, virtual environments, and dependency resolution in a single tool and is significantly faster than traditional `pip` workflows.

- Install `uv`: https://github.com/astral-sh/uv
- Create a project and add dependencies:
```bash
uv init
uv venv
uv add openai
```

### Node.js (If You Build Web Apps Or TS SDK Integrations)

If your AI features live inside a web application or you prefer TypeScript, you will need Node.js. Several providers offer first-class TypeScript SDKs, and frameworks like LangChain and Vercel AI SDK have strong JS/TS support.

- Install Node.js (LTS): https://nodejs.org/en/download
- Prefer a lockfile and a modern package manager (`npm`, `pnpm`, or `yarn`).

### Docker (For Local Databases And Services)

Docker is effectively required for running local vector stores, observability tools, and other supporting services. Most of the tools listed later on this page provide official Docker images, and `docker compose` is the standard way to spin up multi-service local stacks reproducibly.

- Install Docker: https://docs.docker.com/get-docker/
- Expect local stacks to use `docker compose` for repeatability.

## Hosted Model APIs (Quick Start)

For most teams, hosted APIs are the fastest path to production. You avoid GPU infrastructure entirely, get automatic scaling, and can switch providers with relatively modest code changes. The tradeoff is cost at scale and the need to send your data to a third party. The setup is minimal -- install an SDK, set an API key, and you are making calls.

### OpenAI
- Docs: https://platform.openai.com/docs/
- Python SDK: `pip install -U openai`
- JS/TS SDK: `npm install openai`

### Anthropic
- Docs: https://docs.anthropic.com/
- Python SDK: `pip install -U anthropic`

### Common Setup Pitfalls

These mistakes are simple to avoid but surprisingly common, especially on teams where AI work is new and developers are moving fast to get a prototype running.

- **Never commit API keys.** Use environment variables (`OPENAI_API_KEY`, etc.) during development and a proper secrets manager (Vault, AWS Secrets Manager, or your platform's equivalent) in production. A leaked API key can generate a large bill remarkably quickly, and revoking one often means redeploying every service that uses it.
- **Log and store prompts/outputs carefully.** Prompts and completions frequently contain user data, and some of that data may be sensitive. Establish a redaction policy early -- decide what gets logged, what gets stored, and what gets scrubbed -- rather than retrofitting one after an incident.

## Local Models (Quick Start)

Running models locally is valuable for experimentation, offline development, data-sensitive workloads, and cost control at scale. The experience ranges from "install one tool and pull a model" (Ollama on a laptop) to "configure GPU drivers, container runtimes, and serving frameworks" (vLLM or SGLang in production). Be honest about which end of that spectrum your project actually needs before investing in infrastructure.

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

If you are building any kind of retrieval system -- RAG, semantic search, recommendation -- you need somewhere to store and query embeddings. The vector store landscape has many options, and the right choice depends on your existing infrastructure, scale requirements, and operational appetite. For most teams starting out, the simplest answer is: if you already run Postgres, add pgvector; if you do not, pick one of the purpose-built stores with a good Docker story and low operational overhead.

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
- Docs: https://docs.trychroma.com/docs/overview/getting-started

### Weaviate (Medium)
Good for: feature-rich vector DB; typically configured via Docker Compose.

- Docs: https://weaviate.io/developers/weaviate/installation/docker-compose

### Milvus (Hard)
Good for: large-scale vector search; heavier operational footprint.

- Docs: https://milvus.io/docs/install_standalone-docker.md

## Observability (Quick Start)

Observability for LLM systems is still maturing, but getting *something* in place early is far better than adding it after your first production incident. At minimum, you want to be able to trace a user request through your system: what prompt was sent, what was returned, how long it took, and what retrieval or tool calls happened along the way. The two categories below cover LLM-specific tracing (Langfuse) and general-purpose distributed tracing that is increasingly gaining AI-specific conventions (OpenTelemetry).

### Langfuse (Medium)
Good for: prompt/trace tracking and evaluation workflows (self-hosted option).

- Self-hosting docs: https://github.com/langfuse/langfuse

### OpenTelemetry (Medium)
Good for: standard tracing across your services; GenAI semantics are evolving.

- GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/

*Last audited: 2026-02-10 · [Audit methodology](23-audit-methodology.md)*

---
[Contents](README.md) | [Prev](16-stacks-and-difficulty.md) | [Next](01-llm-fundamentals.md)
