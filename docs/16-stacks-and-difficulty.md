# Stacks And Difficulty

Last reviewed: 2026-02-10

## Summary
Most AI projects are "easy to demo" and "hard to ship". This page maps common feature types to typical stacks and where the difficulty actually shows up.

## Difficulty Legend
- Easy: can ship an MVP with minimal infrastructure.
- Medium: quality depends on data pipelines, indexing, and repeatable evals.
- Hard: requires strong safety boundaries and/or heavy ops (GPUs, multi-service systems, permissions).

## Hosted Chat Assistant (Easy To Medium)
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

