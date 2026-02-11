# AI Engineering: A Practical Reference

Last reviewed: 2026-02-10

## About This Book

This is a practitioner's guide to building reliable systems with large language models. It is written for software engineers, technical leads, and architects who need to ship AI-powered features that work in production — not just in demos.

The field moves fast. Rather than chasing daily announcements, this reference focuses on **durable concepts**: the architectural patterns, failure modes, and operational practices that remain useful even as specific models and providers change. Where details are volatile (pricing, context limits, API surfaces), they are date-stamped so you know when to re-verify.

### Who This Is For

- **Engineers** adding LLM features to existing products (chat, search, extraction, automation).
- **Tech leads** making architectural decisions about model selection, tool design, and safety boundaries.
- **Teams** that need a shared vocabulary and a set of checklists for reviewing AI work.

### What You Will Find

- **Concepts explained**, not just listed. Each chapter teaches the "why" before the "how."
- **Decision criteria** instead of "best model" rankings. Your constraints determine the right choice.
- **Checklists** for production readiness, safety, and operational hygiene.
- **A curated tool index** with install commands and difficulty ratings.
- **Primary-source references** (papers, vendor docs, specs) so you can go deeper.

### What This Is Not

- A model leaderboard or product comparison site.
- A tutorial for a specific framework or SDK.
- A substitute for reading the vendor documentation for the tools you actually use.

## How to Read This

| Goal | Start Here |
|------|------------|
| Learn end-to-end | [Book table of contents](docs/README.md) — read front-to-back |
| Look something up | Jump to a chapter via the [table of contents](docs/README.md) |
| Set up your environment | [Installation and Local Setup](docs/15-installation-and-local-setup.md) |
| Understand what is hard | [Stacks and Difficulty](docs/16-stacks-and-difficulty.md) |

## How This Reference Stays Current

- **Date-stamp volatile claims** with `As of YYYY-MM-DD, ...` so readers know when to re-check.
- **Prefer primary sources** (vendor docs, papers, specifications) over summaries and blog posts.
- **Treat pricing, limits, and policies as volatile** unless re-verified within the last 90 days.
- See [Scope and Update Policy](docs/00-scope-and-update-policy.md) for the full editorial approach.

## Tool Index (Curated)

The following is a curated index of tools, frameworks, and services commonly used in LLM engineering. Each entry includes a brief description, install command (where applicable), difficulty rating, and a link to official documentation.

**Difficulty legend:**
- **Easy** — One command to install; minimal configuration needed.
- **Medium** — Requires configuration, environment setup, or familiarity with the ecosystem.
- **Hard** — Requires GPU infrastructure, complex ops, or significant systems work.

### Providers and Primary Documentation

| Provider | Documentation |
|----------|--------------|
| OpenAI | [Platform docs](https://platform.openai.com/docs/) — Responses API, tools, structured outputs, caching, data controls |
| Anthropic | [Docs](https://docs.anthropic.com/) — Tool use, MCP, prompt engineering |
| Google Gemini | [API docs](https://ai.google.dev/docs) |
| Amazon Bedrock | [Docs](https://docs.aws.amazon.com/bedrock/) |
| Azure OpenAI | [Docs](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/responses) — Responses API |
| Mistral | [Docs](https://docs.mistral.ai/) |
| Cohere | [Docs](https://docs.cohere.com/) |

### SDKs and Gateways

| Tool | Install | Difficulty | Docs |
|------|---------|------------|------|
| OpenAI SDK (Python) | `pip install -U openai` | Easy | [Libraries](https://platform.openai.com/docs/libraries) |
| OpenAI SDK (JS/TS) | `npm install openai` | Easy | [Libraries](https://platform.openai.com/docs/libraries) |
| LiteLLM | `pip install litellm` | Medium | [GitHub](https://github.com/BerriAI/litellm) |

### Agents and Orchestration

| Tool | Install | Difficulty | Docs |
|------|---------|------------|------|
| OpenAI Agents SDK (Python) | `pip install openai-agents` | Easy | [GitHub](https://github.com/openai/openai-agents-python) |
| OpenAI Agents SDK (JS/TS) | `npm install @openai/agents zod` | Easy | [GitHub](https://github.com/openai/openai-agents-js) |
| LangGraph (Python) | `pip install -U langgraph` | Medium | [GitHub](https://github.com/langchain-ai/langgraph) |
| LangGraph.js | `npm install @langchain/langgraph @langchain/core` | Medium | [GitHub](https://github.com/langchain-ai/langgraphjs) |
| PydanticAI | `pip install pydantic-ai` | Medium | [Docs](https://ai.pydantic.dev/) |
| Microsoft Agent Framework | `pip install agent-framework --pre` | Medium | [GitHub](https://github.com/microsoft/agent-framework) |
| Semantic Kernel | `pip install semantic-kernel` | Medium | [GitHub](https://github.com/microsoft/semantic-kernel) |

### RAG Frameworks

| Tool | Install | Difficulty | Docs |
|------|---------|------------|------|
| LlamaIndex | `pip install llama-index` | Medium | [Docs](https://docs.llamaindex.ai/) |
| LangChain (Python) | `pip install -U langchain` | Medium | [Docs](https://docs.langchain.com/oss/python/langchain/install) |
| LangChain (JS/TS) | `npm install langchain @langchain/core` | Medium | [Docs](https://docs.langchain.com/oss/javascript/langchain/install) |
| Haystack | `pip install haystack-ai` | Medium | [GitHub](https://github.com/deepset-ai/haystack) |

### Vector Stores

| Tool | Install / Run | Difficulty | Docs |
|------|--------------|------------|------|
| Postgres + pgvector | `docker pull pgvector/pgvector:pg18` | Medium | [GitHub](https://github.com/pgvector/pgvector) |
| Qdrant | `docker run -p 6333:6333 qdrant/qdrant` | Easy | [Quick start](https://qdrant.tech/documentation/quick-start/) |
| Weaviate | Docker Compose (see docs) | Medium | [Docs](https://docs.weaviate.io/weaviate/installation/docker-compose) |
| Milvus | Docker Compose (see docs) | Hard | [Docs](https://milvus.io/docs/install_standalone-docker.md/) |
| Chroma | `docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma` | Easy | [Docs](https://docs.trychroma.com/production/containers/docker) |
| Pinecone (managed) | `pip install pinecone` | Easy | [Quick start](https://docs.pinecone.io/guides/getting-started/quickstart) |

### Local Models and Serving

| Tool | Install | Difficulty | Docs |
|------|---------|------------|------|
| Ollama | `curl -fsSL https://ollama.com/install.sh \| sh` | Easy | [Docs](https://docs.ollama.com/linux) |
| vLLM (pip) | `pip install vllm` | Hard (Linux + GPU) | [Install guide](https://docs.vllm.ai/en/latest/getting_started/installation.html) |
| vLLM (Docker) | See docs | Hard (GPU) | [Docker guide](https://docs.vllm.ai/en/latest/deployment/docker/) |
| SGLang | `uv pip install sglang` | Hard (GPU) | [Install guide](https://docs.sglang.io/get_started/install.html) |
| HF TGI (maintenance mode) | See docs | Hard (GPU) | [GitHub](https://github.com/huggingface/text-generation-inference) |

### Structured Output Helpers

| Tool | Install | Difficulty | Docs |
|------|---------|------------|------|
| Instructor | `pip install instructor` | Easy | [GitHub](https://github.com/instructor-ai/instructor) |
| Outlines | `pip install outlines` | Medium | [Docs](https://dottxt-ai.github.io/outlines/latest/) |

### Evals and Testing

| Tool | Install | Difficulty | Docs |
|------|---------|------------|------|
| OpenAI Evals | See docs | Medium | [GitHub](https://github.com/openai/evals) |
| promptfoo | See docs | Medium | [GitHub](https://github.com/promptfoo/promptfoo) |
| Ragas | `pip install ragas` | Medium | [Docs](https://docs.ragas.io/en/stable/getstarted/install/) |

### Observability and Tracing

| Tool | Install | Difficulty | Docs |
|------|---------|------------|------|
| Langfuse | Docker Compose (see docs) | Medium | [GitHub](https://github.com/langfuse/langfuse) |
| Arize Phoenix | See docs | Medium | [Docs](https://phoenix.arize.com/) |
| OpenTelemetry GenAI | Semantic conventions (in development) | Medium | [Spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/) |
| OpenInference | LLM tracing spec + instrumentations | Medium | [GitHub](https://github.com/Arize-ai/openinference) |

### Security, Governance, and Threat Modeling

| Resource | Description | Link |
|----------|-------------|------|
| OWASP Top 10 for LLMs | Threat taxonomy for LLM applications | [Project](https://owasp.org/www-project-top-10-for-large-language-model-applications/) |
| OWASP Top 10 for LLMs 2025 | Latest version (PDF) | [PDF](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf) |
| NIST AI RMF 1.0 | AI risk management framework | [Framework](https://www.nist.gov/itl/ai-risk-management-framework) |
| NIST GenAI Profile | Generative AI-specific risk guidance | [Profile](https://www.nist.gov/itl/ai-risk-management-framework/generative-ai-profile) |
| MITRE ATLAS | Adversarial threat landscape for AI systems | [Atlas](https://atlas.mitre.org/) |

### Protocols (Tooling Interop)

| Resource | Description | Link |
|----------|-------------|------|
| Model Context Protocol | Standard for tool/context integration | [Docs](https://modelcontextprotocol.io/) |
| MCP Specification | Stable protocol spec | [Spec](https://modelcontextprotocol.io/specification/2024-11-05/index) |
| OpenAI Docs MCP Server | Developer docs accessible in-editor via MCP | [Docs](https://platform.openai.com/docs/docs-mcp) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
