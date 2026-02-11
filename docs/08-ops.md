# Ops: Shipping And Running LLM Systems

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](07-architecture-recipes.md) | [Next](14-governance-and-risk.md)

## Summary
Most failures in production are not "the model is dumb." They are observability, versioning, data quality, retrieval drift, and safety boundary issues. Operating LLM systems in production is fundamentally different from operating traditional software because the core component (the model) is non-deterministic, opaque, and frequently updated by a third party. The operational practices that make LLM systems reliable are not exotic; they are the same disciplines of versioning, monitoring, and incident management that good engineering teams have always practiced, applied to a new and more unpredictable domain.

## See Also
- [Evals And Testing](05-evals.md)
- [Safety, Privacy, And Security](06-safety-privacy-security.md)
- [Governance And Risk](14-governance-and-risk.md)

## Versioning

In traditional software, a deployment is defined by a code version. In LLM systems, a deployment is defined by the intersection of multiple independently changing components, and a regression can be caused by a change to any one of them.

**Prompt version.** Prompts are code. They should be stored in version control, reviewed through pull requests, and tagged with version identifiers. Every prompt change can alter system behavior in unpredictable ways, so treat prompt changes with the same rigor as code changes: review, test, deploy to staging, evaluate, and then promote to production.

**Model version.** Model providers update their models continuously, sometimes without explicit notice. A model that behaved one way last week might behave differently this week because the provider shipped an update. Where possible, pin to specific model versions (snapshot IDs) rather than aliases like "gpt-4" that can resolve to different underlying models over time. When you do upgrade, run your full eval suite before promoting the new version.

**Tool and API versions.** If your system calls external tools or APIs, changes to those interfaces can break your system even when your prompts and model are unchanged. Version your tool definitions alongside your prompts, and include tool behavior in your eval coverage.

**Retrieval index version.** For RAG systems, the contents and configuration of the vector index are a deployment parameter. A new batch of documents, a change to the chunking strategy, or an update to the embedding model can all alter retrieval behavior and, by extension, answer quality. Version your index builds and maintain the ability to roll back to a previous index.

**Policy and configuration version.** Safety filters, content policies, rate limits, and other configuration affect system behavior. Version these alongside everything else so that you can reconstruct the complete configuration of the system at any point in time.

The goal is reproducibility: given a problematic output, you should be able to determine exactly which versions of each component produced it. This requires a deployment manifest that records all version identifiers, and logging that associates each request with the manifest that was active at the time.

## Deployment Strategies

LLM system deployments carry more risk than typical software deployments because behavior changes are harder to predict and harder to detect automatically. Adopt deployment strategies that limit blast radius and enable fast rollback.

**Canary deployments** route a small percentage of traffic to the new version while the majority continues to use the current version. Monitor error rates, latency, and quality metrics on the canary traffic. If metrics are healthy after a defined observation period, gradually increase the canary percentage until the new version handles all traffic. If metrics degrade, roll back immediately.

**Blue-green deployments** maintain two complete environments. The "blue" environment serves production traffic while the "green" environment receives the new deployment. After testing the green environment (with synthetic traffic and a subset of production traffic), you switch all traffic from blue to green. If problems emerge, you switch back. This provides the fastest possible rollback.

**Shadow deployments** (also called dark launches) run the new version on production traffic but do not serve its outputs to users. Instead, the outputs are logged and evaluated offline. This is the safest approach for high-risk changes because there is zero user impact, but it also provides the least realistic signal because it cannot capture user interaction effects.

For prompt changes specifically, consider a graduated rollout: deploy the new prompt to internal users first, then to a small percentage of external traffic, then to all traffic. At each stage, run evals and monitor metrics before proceeding.

## Observability

Observability for LLM systems requires more than traditional application monitoring because the most important failures (wrong answers, hallucinations, policy violations) do not manifest as errors in the conventional sense. The system returns a 200 status code with a perfectly formatted response that happens to be wrong.

### Logging

Every request should produce a log record containing the prompt identifier, model identifier, retrieval query and result identifiers (for RAG systems), tool calls and their results, the model's response, validation results, and any quality signals (user feedback, automated quality scores). These records are the foundation for debugging, evaluation, and incident response.

**Redaction is non-negotiable.** Before any data enters the logging pipeline, strip secrets, PII, and other sensitive content. Implement redaction at the edge (as close to the data source as possible) rather than relying on downstream processing. Test your redaction logic regularly because new data formats and user behaviors can introduce PII in unexpected places.

Structure your logs for queryability. You will need to search by request ID (for debugging specific incidents), by time range (for investigating production issues), by prompt version (for evaluating changes), and by error type (for measuring regression). Choose a log format and storage system that supports these access patterns efficiently.

### Tracing

End-to-end request tracing is essential for understanding system behavior and diagnosing performance issues. For a typical RAG request, tracing should capture the time spent in each pipeline stage: query preprocessing, embedding generation, vector search, re-ranking, prompt assembly, model inference, output validation, and response delivery.

Trace data reveals bottlenecks and anomalies that aggregate metrics can hide. An average latency of 500 milliseconds might conceal a bimodal distribution where most requests complete in 200 milliseconds but 10 percent take 2 seconds due to cache misses in the retrieval layer. Only tracing makes this visible.

Use distributed tracing standards (OpenTelemetry is widely supported) to propagate trace context across service boundaries. This is especially important when your LLM system calls external APIs or delegates work to other services.

### Dashboards

Build dashboards that answer the questions your team actually asks during incidents and routine monitoring.

**Operational health:** request volume, error rates (by type), latency distributions (p50, p95, p99), token usage per request, and cost per request. These tell you whether the system is up and performing within expected parameters.

**Quality signals:** schema validation failure rate, user feedback (thumbs up/down ratio, complaint rate), automated quality scores (if you run online evaluations), and retrieval hit rate (what percentage of queries return at least one relevant result). These tell you whether the system is producing good outputs.

**Resource utilization:** API rate limit headroom, vector index size and query latency, cache hit rates, and budget consumption (daily and cumulative). These tell you whether the system is operating sustainably.

**Safety signals:** content filter trigger rates, refusal rates, unusual query patterns, and access control enforcement metrics. These tell you whether the system is operating safely.

### Alerting

Alert on conditions that require human attention, not on every fluctuation in metrics. Effective alerting for LLM systems requires careful threshold setting because normal variance is higher than in deterministic systems.

Alert when error rates exceed historical baselines by a statistically significant margin. Alert when latency p99 crosses a threshold that affects user experience. Alert when cost per request spikes (which might indicate a prompt that triggers unexpectedly verbose outputs). Alert when user complaint rates increase. Alert when safety filter triggers spike.

Avoid alert fatigue by tuning thresholds based on actual incident history. An alert that fires weekly without leading to action is worse than no alert because it trains the team to ignore alerts.

## Cost And Latency Controls

LLM API calls are expensive relative to traditional API calls, and costs can escalate rapidly without controls. A production system should have cost management built into its architecture from the start, not bolted on after the first surprising invoice.

### Caching

Caching is the single most effective cost optimization for most LLM systems. Many systems receive repeated or near-identical queries, and serving a cached response is essentially free.

**Exact match caching** stores the response for each unique prompt and returns it on subsequent identical requests. This is most effective for systems with stable prompts and low input variability, such as extraction pipelines processing standardized document formats.

**Semantic caching** stores responses keyed by the semantic meaning of the query rather than its exact text. When a new query is sufficiently similar to a cached query (as measured by embedding similarity), the cached response is returned. This captures a broader range of cache hits but introduces the risk of returning a response that does not exactly match the new query.

**Provider-level caching** (such as prompt caching offered by some providers) caches the processing of long prompt prefixes across requests. This is especially valuable for systems with large system prompts or many few-shot examples, because the static portion of the prompt is processed only once and reused across requests.

### Budgeting

Set per-request token budgets that limit how much the model can generate. This prevents runaway responses and provides predictable cost ceilings. Set per-user and per-hour budgets to prevent individual users or automated processes from consuming disproportionate resources.

Monitor budget consumption in real time and alert when utilization approaches limits. Sudden increases in token usage often indicate a problem: a prompt that is triggering verbose outputs, a loop in an agent, or a spike in traffic.

### Graceful Degradation

Design your system to degrade gracefully under load or when approaching budget limits rather than failing completely.

**Model fallback:** When the primary model is unavailable or too slow, fall back to a smaller, faster, cheaper model. The quality may be lower, but the system continues to function. This requires that your prompts work (possibly with reduced quality) across multiple model tiers.

**Feature reduction:** Under load, disable expensive features like re-ranking or multi-step reasoning and serve simpler but still useful responses. A RAG system that skips re-ranking still provides grounded answers, just with slightly lower precision.

**Queue and backpressure:** For batch or asynchronous workloads, implement queuing with backpressure so that spikes in demand are absorbed rather than causing cascading failures. Return clear status information so that callers know their request is queued rather than lost.

## Prompt and Model Rollback

The ability to roll back to a previous known-good configuration is one of the most important operational capabilities for an LLM system. When a new prompt or model version causes a regression, the fastest mitigation is reverting to the previous version while you investigate.

Rollback requires that you have maintained previous versions of all components (prompts, model identifiers, configurations, retrieval index snapshots) and that you can switch between them quickly. This is a deployment infrastructure concern: your system should support configuration-driven version selection that can be changed without a code deployment.

Test your rollback procedure before you need it. A rollback that has never been exercised is a rollback that does not work. Include rollback in your regular operational drills, and verify that reverting to a previous version actually restores the expected behavior by running your eval suite against the rolled-back configuration.

Document the rollback procedure clearly enough that any on-call engineer can execute it at 3 AM without making decisions. Specify the exact steps, the commands to run, the metrics to check afterward, and the criteria for declaring the rollback successful.

## Incident Response

Incident response for LLM systems follows the same general structure as any software incident, with some domain-specific considerations.

### Defining "Bad Output"

Traditional software incidents are often clear: the service is down, the API returns errors, the database is corrupted. LLM system incidents can be subtler: the service is up and returning valid responses, but the responses are wrong, biased, or unsafe. Establish clear categories for "bad output" and define the severity of each.

- **Safety incidents:** The system generates harmful content, facilitates dangerous actions, or violates critical policies. Highest severity; may require immediate shutdown of the affected feature.
- **Correctness incidents:** The system consistently produces wrong answers for a category of queries. High severity; may require prompt or model rollback.
- **Privacy incidents:** The system leaks PII or confidential information in its responses or logs. High severity; may require data cleanup in addition to a system fix.
- **Quality incidents:** The system's outputs degrade in helpfulness, tone, or formatting. Lower severity but still requires investigation, especially if it affects user trust.

### Artifact Capture

When investigating an incident, capture reproducible artifacts: the exact input that triggered the problem, the exact prompt (including system message and any retrieved context), the model identifier, and the model's response. Sanitize these artifacts to remove PII or confidential information, but preserve enough detail to reproduce the issue.

These artifacts serve double duty: they drive the immediate investigation, and they become regression test cases that prevent the same failure from recurring.

### Regression Integration

After every incident, add one or more test cases to your eval suite that specifically target the failure mode. This is the mechanism that converts operational pain into lasting quality improvement. Over time, your eval suite becomes a comprehensive record of everything that has gone wrong, ensuring that no fixed issue regresses silently.

## Checklist
- Can you reproduce a problematic output given request logs?
- Can you roll back prompt, model, and index versions independently?
- Do you have dashboards for latency, cost, error rates, and quality signals?
- Are alerts tuned to actionable thresholds?
- Is there a documented incident response procedure with clear severity definitions?
- Do you have cost controls (budgets, caching, fallback models)?
- Are logs structured, queryable, and properly redacted?
- Do incidents produce regression tests that are added to the eval suite?
- Has the rollback procedure been tested?

## References
- OpenAI docs: Production best practices. https://platform.openai.com/docs/guides/production-best-practices
- OpenAI docs: Prompt caching. https://platform.openai.com/docs/guides/prompt-caching

---
[Contents](README.md) | [Prev](07-architecture-recipes.md) | [Next](14-governance-and-risk.md)
