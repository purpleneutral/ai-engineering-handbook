# AI Notes

Last reviewed: 2026-02-10

A curated, GitHub-first reference for modern AI/LLM engineering: practical concepts, decision checklists, and a tooling index with install notes.

## How This Repo Stays "Current"
- Date-stamp volatile claims with `As of YYYY-MM-DD, ...`
- Prefer primary sources (vendor docs, papers, specs) over summaries
- Treat pricing, limits, and policies as volatile unless re-checked

## Start Here
- Local setup and installs: `docs/15-installation-and-local-setup.md`
- Scope and update policy: `docs/00-scope-and-update-policy.md`
- Foundations: `docs/01-llm-fundamentals.md`

## Guides (In This Repo)
- Prompting: `docs/02-prompting.md`
- RAG: `docs/03-rag.md`
- Agents: `docs/04-agents.md`
- Evals: `docs/05-evals.md`
- Safety/privacy/security: `docs/06-safety-privacy-security.md`
- Architecture recipes: `docs/07-architecture-recipes.md`
- Ops: `docs/08-ops.md`
- Structured outputs + tool calling: `docs/11-structured-outputs-and-tool-calling.md`
- Embeddings + vector search: `docs/12-embeddings-and-vector-search.md`
- Staying current: `docs/13-staying-current.md`
- Governance + risk: `docs/14-governance-and-risk.md`
- Glossary: `docs/09-glossary.md`
- Reading list: `docs/10-reading-list.md`

## Tool Index (Curated)

Legend: Ease = Easy (one command), Medium (needs config), Hard (GPU/ops heavy).

### Providers And Primary Docs
- OpenAI Platform docs (Responses API, tools, structured outputs, caching, data controls). https://platform.openai.com/docs/
- Anthropic docs (tool use, MCP, prompt engineering). https://docs.anthropic.com/
- Google Gemini API docs. https://ai.google.dev/docs
- Amazon Bedrock docs. https://docs.aws.amazon.com/bedrock/
- Azure OpenAI (Responses API). https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/responses
- Mistral docs. https://docs.mistral.ai/
- Cohere docs. https://docs.cohere.com/

### SDKs And Gateways
- OpenAI SDK (Python): `pip install -U openai` (Easy). https://platform.openai.com/docs/libraries
- OpenAI SDK (JS/TS): `npm install openai` (Easy). https://platform.openai.com/docs/libraries
- LiteLLM (multi-provider OpenAI-compatible SDK + proxy): `pip install litellm` (Medium). https://github.com/BerriAI/litellm

### Agents And Orchestration
- OpenAI Agents SDK (Python): `pip install openai-agents` (Easy). https://github.com/openai/openai-agents-python
- OpenAI Agents SDK (JS/TS): `npm install @openai/agents zod` (Easy). https://github.com/openai/openai-agents-js
- LangGraph (Python): `pip install -U langgraph` (Medium). https://github.com/langchain-ai/langgraph
- LangGraph.js: `npm install @langchain/langgraph @langchain/core` (Medium). https://github.com/langchain-ai/langgraphjs
- PydanticAI: `pip install pydantic-ai` (Medium). https://ai.pydantic.dev/
- Microsoft Agent Framework: `pip install agent-framework --pre` (Medium). https://github.com/microsoft/agent-framework
- Semantic Kernel: `pip install semantic-kernel` (Medium). https://github.com/microsoft/semantic-kernel

### RAG Frameworks
- LlamaIndex: `pip install llama-index` (Medium). https://docs.llamaindex.ai/
- LangChain (Python): `pip install -U langchain` (Medium). https://docs.langchain.com/oss/python/langchain/install
- LangChain (JS/TS): `npm install langchain @langchain/core` (Medium). https://docs.langchain.com/oss/javascript/langchain/install
- Haystack: `pip install haystack-ai` (Medium). https://github.com/deepset-ai/haystack

### Vector Stores
- Postgres + pgvector (Docker image): `docker pull pgvector/pgvector:pg18` (Medium). https://github.com/pgvector/pgvector
- Qdrant (Docker): `docker run -p 6333:6333 qdrant/qdrant` (Easy). https://qdrant.tech/documentation/quick-start/
- Weaviate (Docker Compose): see docs (Medium). https://docs.weaviate.io/weaviate/installation/docker-compose
- Milvus (Docker Compose): see docs (Hard). https://milvus.io/docs/install_standalone-docker.md/
- Chroma (Docker): `docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma` (Easy). https://docs.trychroma.com/production/containers/docker
- Pinecone (managed): `pip install pinecone` (Easy). https://docs.pinecone.io/guides/getting-started/quickstart

### Local Models And Serving
- Ollama (local runner): `curl -fsSL https://ollama.com/install.sh | sh` (Easy). https://docs.ollama.com/linux
- vLLM (pip): `pip install vllm` (Hard; Linux + GPU). https://docs.vllm.ai/en/latest/getting_started/installation.html
- vLLM (Docker): see docs (Hard; GPU). https://docs.vllm.ai/en/latest/deployment/docker/
- SGLang: `uv pip install sglang` (Hard; GPU). https://docs.sglang.io/get_started/install.html
- Hugging Face TGI (maintenance mode): see docs (Hard; GPU). https://github.com/huggingface/text-generation-inference

### Structured Output Helpers
- Instructor (Pydantic-first structured outputs): `pip install instructor` (Easy). https://github.com/instructor-ai/instructor
- Outlines (structured generation / constrained outputs): `pip install outlines` (Medium). https://dottxt-ai.github.io/outlines/latest/

### Evals And Testing
- OpenAI Evals (framework). https://github.com/openai/evals
- promptfoo (prompt testing). https://github.com/promptfoo/promptfoo
- Ragas (RAG evaluation): `pip install ragas` (Medium). https://docs.ragas.io/en/stable/getstarted/install/

### Observability And Tracing
- Langfuse (self-host with Docker Compose): see docs (Medium). https://github.com/langfuse/langfuse
- Arize Phoenix (open source): https://phoenix.arize.com/
- OpenTelemetry GenAI semantic conventions (status: development). https://opentelemetry.io/docs/specs/semconv/gen-ai/
- OpenInference (LLM tracing spec + instrumentations). https://github.com/Arize-ai/openinference

### Security, Governance, And Threat Modeling
- OWASP Top 10 for LLM Applications (project). https://owasp.org/www-project-top-10-for-large-language-model-applications/
- OWASP Top 10 for LLM Applications 2025 (PDF). https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf
- NIST AI Risk Management Framework (AI RMF 1.0). https://www.nist.gov/itl/ai-risk-management-framework
- NIST Generative AI Profile. https://www.nist.gov/itl/ai-risk-management-framework/generative-ai-profile
- MITRE ATLAS (AI threat techniques). https://atlas.mitre.org/

### Protocols (Tooling Interop)
- Model Context Protocol (official docs). https://modelcontextprotocol.io/
- MCP specification (stable). https://modelcontextprotocol.io/specification/2024-11-05/index
- OpenAI Docs MCP server (developer docs in-editor). https://platform.openai.com/docs/docs-mcp

## Contributing
See `CONTRIBUTING.md`.
