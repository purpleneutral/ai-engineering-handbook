# Scope And Update Policy

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](README.md) | [Next](16-stacks-and-difficulty.md)

## Summary
These notes aim to be a durable reference for building reliable AI features, not a running news feed.

## See Also
- [AI Notes (Book)](README.md)
- [Staying Current (Without Chasing Hype)](13-staying-current.md)

## What This Repo Covers

The AI/LLM space generates an extraordinary amount of noise -- new models every week, benchmarks that shift overnight, and tooling that appears (and sometimes disappears) just as quickly. These notes deliberately step back from that churn. The goal is to capture the reasoning, patterns, and architectural decisions that remain useful even after the specific numbers change. If you understand *why* a design decision was made, you can adapt when the specific tools or services change underneath you.

With that philosophy in mind, the topics covered here include:

- LLM fundamentals and typical failure modes
- Prompting patterns and structured outputs
- Retrieval-Augmented Generation (RAG) design and evaluation
- Agent loops, tool design, and safety boundaries
- Evaluation strategies and regression testing
- Production concerns: latency, cost, observability, and incident response
- A curated tooling index with install pointers (see [README.md](../README.md) and [Installation And Local Setup](15-installation-and-local-setup.md))

Each topic is written to be self-contained enough to read on its own, but cross-references tie them together into a coherent picture of what it takes to ship AI features responsibly.

## What This Repo Does Not Try To Do

It is just as important to be clear about what is *out* of scope. Keeping sharp boundaries prevents these notes from drifting into yet another aggregator of model announcements.

- **Maintain a constantly-updated "best model/provider" leaderboard.** Model rankings shift too frequently for a static document to stay accurate. Instead, these notes focus on the criteria you should use to evaluate models yourself -- latency characteristics, output structure support, safety controls, and pricing models -- so that you can make your own informed comparison when the time comes.
- **Track day-to-day product announcements.** If a provider ships a new feature that changes a fundamental design pattern, that will show up here. A new pricing tier or a renamed API endpoint will not. See [Staying Current](13-staying-current.md) for strategies on monitoring the news without drowning in it.
- **Replace primary documentation.** These notes link to official docs liberally. The intent is to provide the *context* around a tool or API -- when to use it, what to watch out for, how it fits into larger patterns -- not to duplicate the reference material that the maintainers already keep up to date.

## Update Policy (Practical)

A reference that silently goes stale is worse than no reference at all, because it breeds false confidence. The update policy below is designed to keep the content honest about its own shelf life while minimizing the maintenance burden.

- **If it changes quickly, date it.** Any claim that is likely to become outdated within a few months should carry an explicit date. Example: `As of 2026-02-10, Provider X supports ...`. This makes it easy for a future reader (including your future self) to spot stale information and verify it.
- **Keep provider-specific details localized.** When you need to mention a provider's quirk or limitation, add a short subsection rather than sprinkling provider-specific notes throughout a general discussion. This keeps the core material clean and makes it straightforward to update or remove provider details when they change.
- **Write to "selection criteria" and "tradeoffs", not to "best X".** Concrete numbers go stale; the reasoning behind a decision does not. If you write "Provider A had the lowest p99 latency in our tests as of 2026-01," a reader six months later knows to re-run the comparison. If you write "Provider A is the best," they have nothing actionable. Prefer the former. Your future self can re-check the numbers, but the framework for making the decision stays useful indefinitely.

## Page Template

Every page in this collection follows a consistent structure so that readers can quickly orient themselves regardless of which topic they land on. When adding a new topic, start from `docs/_template.md`. The template includes the standard sections -- Summary, See Also, the main content area, a Checklist, and References -- along with the navigation links that tie the pages together. Consistency matters more than it might seem: when every page has the same bones, readers spend less time figuring out where to look and more time absorbing the material.

*Last audited: 2026-02-10 · [Audit methodology](23-audit-methodology.md)*

---
[Contents](README.md) | [Prev](README.md) | [Next](16-stacks-and-difficulty.md)
