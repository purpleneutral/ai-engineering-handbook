# Ops: Shipping And Running LLM Systems

Last reviewed: 2026-02-10

## Summary
Most failures in production are not "the model is dumb". They are observability, versioning, data quality, retrieval drift, and safety boundary issues.

## Versioning
- Prompt version
- Model version
- Tool/API versions
- Retrieval index version
- Policy/config version

## Observability
- Logs: request metadata and artifacts (prompt id, model id, retrieval ids, tool calls, validation results).
- Redaction: do not log secrets/PII; redact at the edge.
- Tracing: measure latency per step (retrieval, tool calls, generation).

## Cost And Latency Controls
- Caching: prompt/output caching for stable tasks.
- Budgeting: per-request token budgets and tool budgets.
- Degradation: fallback models or simpler behavior when under load.

## Incident Response
- Define "bad output" categories (safety, correctness, privacy).
- Capture reproducible artifacts (sanitized).
- Add regression cases after an incident.

## Checklist
- Can you reproduce a problematic output?
- Can you roll back prompt/model/index versions?
- Do you have dashboards for latency, cost, and error rates?

## References
- Add ops references here.
