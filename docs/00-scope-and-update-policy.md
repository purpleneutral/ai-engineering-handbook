# Scope And Update Policy

Last reviewed: 2026-02-10

## Summary
These notes aim to be a durable reference for building reliable AI features, not a running news feed.

## What This Repo Covers
- LLM fundamentals and typical failure modes
- Prompting patterns and structured outputs
- Retrieval-Augmented Generation (RAG) design and evaluation
- Agent loops, tool design, and safety boundaries
- Evaluation strategies and regression testing
- Production concerns: latency, cost, observability, and incident response
- A curated tooling index with install pointers (see `README.md` and `docs/15-installation-and-local-setup.md`)

## What This Repo Does Not Try To Do
- Maintain a constantly-updated "best model/provider" leaderboard
- Track day-to-day product announcements
- Replace primary documentation

## Update Policy (Practical)
- If it changes quickly, date it. Example: `As of 2026-02-10, Provider X supports ...`
- Keep provider-specific details localized. Add a short subsection instead of sprinkling provider quirks everywhere.
- Write to "selection criteria" and "tradeoffs". Your future self can re-check the numbers, but the reasoning stays useful.

## Page Template
Use `docs/_template.md` when adding new topics.
