# Stacks And Difficulty

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](00-scope-and-update-policy.md) | [Next](15-installation-and-local-setup.md)

## Summary
Most AI projects are "easy to demo" and "hard to ship". This page maps common feature types to typical stacks and where the difficulty actually shows up.

## See Also
- [Architecture Recipes](07-architecture-recipes.md)
- [Ops: Shipping And Running LLM Systems](08-ops.md)
- [Governance And Risk](14-governance-and-risk.md)

## Difficulty Legend

One of the most common mistakes in AI project planning is confusing "how hard is the demo" with "how hard is production." A chat assistant can look impressive in a five-minute screencast, but shipping it with consistent behavior, proper data retention, and regression coverage is a fundamentally different undertaking. The difficulty ratings below reflect production readiness, not prototype speed.

- **Easy:** can ship an MVP with minimal infrastructure. The happy path works quickly, and the failure modes are manageable with basic validation and logging.
- **Medium:** quality depends on data pipelines, indexing, and repeatable evals. You will spend meaningful time on data quality, retrieval tuning, or schema validation -- things that barely matter in a demo but dominate production effort.
- **Hard:** requires strong safety boundaries and/or heavy ops (GPUs, multi-service systems, permissions). These projects often involve side effects, complex infrastructure, or security concerns that demand careful engineering beyond the model itself.

These ratings are not about the inherent complexity of the LLM calls. They reflect the total effort to build, test, deploy, and operate the feature responsibly.

## Hosted Chat Assistant (Easy To Medium)

The hosted chat assistant is the most common starting point for teams adding AI features. The core loop is straightforward -- accept user input, call a hosted model API, return a response -- and the tooling ecosystem is mature enough that a basic version can be running in an afternoon. The difficulty escalates when you move beyond happy-path demos into the realities of production: long conversations that drift, edge cases that produce surprising behavior, and privacy requirements that constrain what you can log and retain.

### Typical Stack
- Hosted model API
- Tool calling for real-world actions (search, DB lookup, calculations)
- Basic memory policy (what persists vs what does not)
- Logging + redaction

### Easy Parts
- Initial UX and "helpfulness" on happy paths
- Adding simple tools with allowlisted parameters

### Hard Parts
- Consistent behavior across edge cases and long chats
- Privacy and data retention policy
- Regression testing across prompt/model changes

### Minimum Bar Checklist
- Schema validation for tool calls
- Safe defaults (refuse or escalate when uncertain)
- Basic eval set for your top user intents

## Document Q&A (RAG) (Medium)

RAG systems are the workhorse of enterprise AI -- nearly every organization has a corpus of documents that people want to query conversationally. The prototype is deceptively easy: embed some documents, run a similarity search, and feed the results to a model. The difficulty lives in the gap between "works on ten clean PDFs" and "works reliably on your actual document corpus with its messy formatting, access controls, and ambiguous content." Retrieval quality, not generation quality, is almost always the bottleneck in production RAG systems.

### Typical Stack
- Ingestion (PDF/HTML/text) and normalization
- Chunking + embeddings + vector store (+ metadata filters)
- Optional reranking
- Grounded answer synthesis with citations
- Retrieval + faithfulness evals

### Easy Parts
- "Search my docs" prototype on a small corpus

### Hard Parts
- High-precision retrieval on messy real corpora
- Access control (per-user/per-team permissions) enforced at retrieval time
- Measuring and improving retrieval quality (not just answer quality)

### Minimum Bar Checklist
- Log retrieved chunk ids (not just the final answer)
- Handle empty/low-confidence retrieval (say "I don't know")
- Version the index build inputs and embedding model

## Extraction / Structuring (Easy To Medium)

Extraction tasks -- pulling structured data out of unstructured text -- are among the highest-ROI applications of LLMs because they replace tedious manual work with something that can be validated programmatically. When the input is clean and the schema is well-defined, these systems are straightforward to build and test. The difficulty spikes when you encounter real-world documents: inconsistent formatting, ambiguous fields, tables that do not parse cleanly, and edge cases where the "correct" extraction is genuinely unclear. Strict schema validation is non-negotiable here, because silent extraction errors can propagate through downstream systems undetected.

### Typical Stack
- Structured outputs (schema-constrained JSON)
- Validation + bounded retries
- Human review loop for failures (optional but common)

### Easy Parts
- Clean text extraction and straightforward schemas

### Hard Parts
- PDFs, tables, and non-standard formatting
- Ambiguous fields (what counts as "the title"?)
- Silent errors if you skip validation

### Minimum Bar Checklist
- Strict schema validation before using outputs
- Golden-set regression tests for tricky documents

## Workflow Automation (Agents) (Hard)

Agent systems -- where the model plans and executes multi-step workflows with real-world side effects -- represent the highest-risk, highest-reward category. The appeal is obvious: automate complex tasks that currently require human judgment and action. The danger is equally obvious: an agent that can send emails, modify tickets, or write files can cause real damage if it misbehaves. Every tool you expose is an attack surface, and every side effect is a potential incident. The engineering effort here is dominated by safety boundaries, permission models, and the operational infrastructure to detect and recover from failures.

### Typical Stack
- Agent loop with tool calling
- Per-tool permissions and allowlists
- Budgets (token/time/tool-call limits)
- Sandboxing for code and file operations
- Auditing for tool calls and side effects

### Easy Parts
- Basic multi-step workflows with read-only tools

### Hard Parts
- Side effects (emailing, writing files, changing tickets) safely and reliably
- Prompt injection via web/content sources
- Avoiding infinite loops / thrashing

### Minimum Bar Checklist
- Least-privilege tools
- Explicit stop conditions and "no-progress" detection
- Human-in-the-loop gates for high-impact actions

## Local Models And Serving (Hard)

Running your own models gives you control over data residency, latency, and per-token cost at scale -- but it trades API simplicity for operational complexity. On a laptop with Ollama, local models are genuinely easy to experiment with. In production, you are signing up for GPU driver management, container orchestration, throughput tuning, and a model upgrade process that needs its own evaluation pipeline. The gap between "it runs" and "it runs well, reliably, at the throughput we need" is where most of the effort lives.

### Typical Stack
- Local runner (Ollama) for experimentation
- Production serving (vLLM/SGLang/other) for performance and control
- GPU drivers + container runtime setup (when applicable)
- Monitoring (latency, throughput, GPU memory)

### Easy Parts
- Running local experiments on a laptop (Ollama-class tools)

### Hard Parts
- GPU ops (drivers, CUDA/toolchain compatibility)
- Throughput tuning (batching, KV cache, quantization tradeoffs)
- Keeping quality stable across model upgrades

## References
- OWASP Top 10 for LLM Applications. https://owasp.org/www-project-top-10-for-large-language-model-applications/
- NIST AI RMF 1.0. https://www.nist.gov/itl/ai-risk-management-framework

---
[Contents](README.md) | [Prev](00-scope-and-update-policy.md) | [Next](15-installation-and-local-setup.md)
