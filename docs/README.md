# AI Engineering: Table of Contents

Last reviewed: 2026-02-10

[Home](../README.md) | [Start reading](00-scope-and-update-policy.md)

## How to Use This Book

These chapters are designed to be read front-to-back as a cohesive guide, but each one also stands alone as a reference. If you are learning the field, follow the Core Reading Path in order and use the Prev/Next links at the bottom of each page. If you are looking something up, jump directly to the relevant chapter and follow the See Also links to explore related topics.

## Core Reading Path

### Part 0: Orientation

Before diving into technical content, these chapters establish what this reference covers, what is genuinely difficult in practice, and how to set up your development environment.

1. [Scope and Update Policy](00-scope-and-update-policy.md) — What this book covers, what it deliberately omits, and the editorial conventions for keeping content current.
2. [Stacks and Difficulty](16-stacks-and-difficulty.md) — A honest map of common AI feature types, their typical technology stacks, and where the real difficulty hides (spoiler: it is rarely the model).
3. [Installation and Local Setup](15-installation-and-local-setup.md) — A pragmatic checklist for setting up SDKs, local model runners, vector stores, and observability tools.

### Part I: Foundations

The building blocks that every LLM-powered system relies on. These chapters cover how models work, how to communicate with them effectively, how to get structured data out, and how to search over your own content.

4. [LLM Fundamentals](01-llm-fundamentals.md) — How large language models work under the hood: tokenization, the transformer architecture, training stages, and the failure modes that follow from how they are built.
5. [Prompting](02-prompting.md) — Prompt engineering as interface design: anatomy of a prompt, practical patterns, few-shot examples, chain-of-thought reasoning, and the pitfalls that break prompts in production.
6. [Structured Outputs and Tool Calling](11-structured-outputs-and-tool-calling.md) — Turning free-form text into validated data and controlled actions: schema-constrained outputs, function calling, validation loops, and tool safety.
7. [Embeddings and Vector Search](12-embeddings-and-vector-search.md) — How embedding models represent meaning as vectors, how similarity search works, and practical design choices for chunking, indexing, and retrieval.
8. [Fine-Tuning And Model Customization](17-fine-tuning.md) — When and how to adapt a pre-trained model to your task or domain: the decision framework, data requirements, the training workflow, and the pitfalls that make fine-tuning fail.

### Part II: System Patterns

With the foundations in place, these chapters cover the dominant architectural patterns for building LLM-powered systems: retrieval-augmented generation, autonomous agents, and the protocol layer that connects them to external tools.

9. [Retrieval-Augmented Generation (RAG)](03-rag.md) — Grounding model outputs in external knowledge: the full pipeline from ingestion through chunking, retrieval, re-ranking, and grounded synthesis.
10. [Agents](04-agents.md) — LLM-driven loops that plan, call tools, observe results, and iterate: architecture, design trade-offs, failure modes, and safety boundaries.
11. [Model Context Protocol (MCP)](18-model-context-protocol.md) — The open standard for connecting LLM applications to external tools and data sources: architecture, primitives, security model, and practical adoption guidance.

### Part III: Reliability and Production

Building a working prototype is the easy part. These chapters cover what it takes to make LLM systems reliable, safe, and operable at production scale.

12. [Evals and Testing](05-evals.md) — Treating prompts and model behavior as testable code: golden sets, model-graded evaluation, regression testing, and continuous monitoring.
13. [Safety, Privacy, and Security](06-safety-privacy-security.md) — The expanded attack surface of LLM systems: prompt injection, data exfiltration, tool abuse, supply chain security, agent threats, and defense-in-depth strategies.
14. [Architecture Recipes](07-architecture-recipes.md) — Concrete blueprints for the most common AI feature shapes: chat assistants, document Q&A, extraction pipelines, and workflow automation agents.
15. [Ops: Shipping and Running LLM Systems](08-ops.md) — The operational concerns that determine whether an LLM feature survives contact with production: versioning, observability, cost control, and incident response.
16. [Governance and Risk](14-governance-and-risk.md) — Organizational practices for shipping AI responsibly: system documentation, risk assessment, change control, and compliance.
17. [Staying Current (Without Chasing Hype)](13-staying-current.md) — A practical framework for tracking what matters in a fast-moving field without drowning in noise.

### Part IV: Specialized Topics

Deeper dives into specific capabilities and domains that are increasingly important for production AI systems.

18. [Multimodal AI](20-multimodal.md) — Working with images, audio, video, and documents: provider capabilities, token economics, multimodal RAG, document processing, and visual prompt injection defenses.
19. [Cost Engineering And Optimization](21-cost-engineering.md) — Token pricing models, prompt caching strategies, batching, model tiering and routing, self-host vs. API economics, and cost monitoring.
20. [Guardrails And Content Moderation](22-guardrails-and-moderation.md) — Moderation APIs, guardrail frameworks (NeMo, Guardrails AI, LLM Guard), PII detection pipelines, input and output validation, and policy enforcement architecture.
21. [Prompt Management](24-prompt-management.md) — Treating prompts as managed artifacts: version control, registries, templating systems, A/B testing, environment-specific configuration, and team collaboration patterns.
22. [Data Pipelines For AI](25-data-pipelines.md) — The upstream work of turning messy documents into clean data: parsing, OCR, metadata extraction, deduplication, quality measurement, and pipeline orchestration.
23. [Caching And Latency Optimization](26-caching-and-latency.md) — Prompt caching, semantic caching, response caching, streaming, parallel tool calls, model tiering for latency, and measuring TTFT/P95/P99.
24. [UX Design For AI Features](27-ux-design-for-ai.md) — The engineering side of AI interfaces: streaming display, uncertainty indicators, error states, chat vs. structured UI, feedback collection, and human-in-the-loop patterns.

### Appendices

- [Glossary](09-glossary.md) — Definitions for key terms used throughout this book.
- [Reading List (Curated)](10-reading-list.md) — Annotated list of foundational papers, primary documentation, and high-signal information sources.
- [A Brief History of AI and LLMs](19-history-of-ai.md) — From the 1943 McCulloch-Pitts neuron to today's reasoning models: the key milestones, breakthroughs, and winters that shaped the current landscape.
- [Audit Methodology](23-audit-methodology.md) — How factual claims, links, and code examples are verified: scope, severity classification, and limitations.

## Adding New Pages

1. Start from the [page template](_template.md).
2. Add the page to this table of contents and wire up Prev/Next links on adjacent pages.
3. Date-stamp volatile claims with `As of YYYY-MM-DD, ...` (see [Scope and Update Policy](00-scope-and-update-policy.md)).

---
[Home](../README.md) | [Start reading](00-scope-and-update-policy.md)
