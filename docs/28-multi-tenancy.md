# Multi-Tenancy And Enterprise Patterns

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](27-ux-design-for-ai.md) | [Next](29-migration-and-vendor-strategy.md)

## Summary
Multi-tenancy is the practice of serving multiple customers (or departments, or business units) from a single shared AI system while keeping each customer's data, configuration, and usage strictly separated. Most AI systems start as single-tenant prototypes and hit hard architectural walls when a second customer shows up. Multi-tenancy in LLM applications is fundamentally more complex than in traditional SaaS because the shared resources are more expensive (model inference), the isolation boundaries are less obvious (vector stores, conversation histories, fine-tuned weights), and the consequences of cross-tenant leakage are more severe (one tenant's proprietary data appearing in another tenant's model responses). This chapter is deeply technical and covers the engineering patterns that make multi-tenant AI systems work in production: per-tenant configuration, data isolation, rate limiting, billing, compliance, and the operational machinery that holds it all together.

## See Also
- [Safety, Privacy, And Security](06-safety-privacy-security.md)
- [Ops: Shipping And Running LLM Systems](08-ops.md)
- [Governance And Risk](14-governance-and-risk.md)
- [Retrieval-Augmented Generation (RAG)](03-rag.md)
- [Architecture Recipes](07-architecture-recipes.md)

## When To Use

Multi-tenancy becomes relevant the moment your AI system serves more than one organizational boundary. That boundary might be separate customers of your SaaS product, separate departments within an enterprise, separate brands under a holding company, or separate regulatory jurisdictions that require data separation. The defining question is: does any data, configuration, or behavior need to be isolated between groups of users?

If the answer is yes, you need a multi-tenancy strategy, and you need it before you have two tenants, not after. Retrofitting tenant isolation into a system designed for a single customer is one of the most expensive refactoring exercises in software engineering, and it is worse in AI systems because the isolation boundaries extend to vector stores, embedding spaces, model configurations, conversation histories, and fine-tuned weights.

Single-tenant deployments (one instance per customer) are the simplest approach and appropriate when you have a small number of high-value customers with strong isolation requirements. But they do not scale: each new customer requires a full deployment, and operational overhead grows linearly. Multi-tenancy trades deployment simplicity for operational efficiency, and the patterns in this chapter make that trade manageable.

## How It Works

### Per-Tenant Model Configuration

In a multi-tenant LLM system, different tenants typically need different model behaviors. One tenant may want a conservative, formal assistant; another wants a creative, casual one. One tenant may be on a budget and willing to use a smaller model; another pays for the best available. The system must support these differences without requiring separate deployments.

A **tenant configuration object** is the central abstraction. It captures everything that varies per tenant: the model identifier, temperature, max tokens, system prompt, safety policy, and any custom instructions. This configuration is loaded at the beginning of every request and threaded through the entire pipeline.

```python
from dataclasses import dataclass, field

@dataclass
class TenantConfig:
    tenant_id: str
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = "You are a helpful assistant."
    safety_policy: str = "standard"
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100_000
    allowed_tools: list[str] = field(default_factory=list)
    data_region: str = "us-east-1"
    custom_instructions: dict = field(default_factory=dict)
```

Store tenant configurations in a fast, reliable data store (a relational database, a configuration service, or a key-value store with caching). Load the configuration once per request and validate it against a schema to catch misconfigurations before they reach the model. Cache aggressively but invalidate on configuration changes; a stale configuration that uses the wrong model or the wrong system prompt can produce tenant-visible errors that are difficult to diagnose.

**System prompt management** deserves particular attention. System prompts are the primary mechanism for customizing model behavior per tenant, and they accumulate complexity quickly. A typical production system has a base prompt (common to all tenants), a tenant-specific overlay (tone, domain, constraints), and a request-specific layer (retrieved context, tool descriptions). Compose these layers in a well-defined order with clear precedence rules so that tenant customizations override defaults without contradicting safety constraints.

For tenants that require **tenant-specific fine-tuned models**, the configuration object includes a model identifier that points to the fine-tuned variant. As of 2026-02-10, [OpenAI](https://platform.openai.com/docs/guides/fine-tuning), [Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/fine-tuning), and [Google](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/tune-models) all support fine-tuning with tenant-scoped training data. The key constraint is that training data from one tenant must never contaminate another tenant's fine-tuned model. Implement strict data pipeline separation for fine-tuning: separate training data storage, separate training jobs, and separate model artifacts, all tagged with the tenant identifier. Version fine-tuned models per tenant and maintain the ability to roll back to the base model if a fine-tuning run produces regressions.

### Tenant Isolation In Vector Stores

Data isolation in vector stores is the most critical and most commonly mishandled aspect of multi-tenant RAG systems. The failure mode is **cross-tenant contamination**: a query from Tenant A retrieves documents that belong to Tenant B. This is not merely a privacy violation; in many industries it is a regulatory breach, a contractual violation, or both.

There are three primary isolation strategies, each with different trade-offs.

**Index-per-tenant** gives each tenant a completely separate vector index (a separate collection in [Qdrant](https://qdrant.tech/documentation/guides/multiple-partitions/), a separate namespace in [Pinecone](https://docs.pinecone.io/guides/indexes/use-namespaces), a separate collection in [Weaviate](https://weaviate.io/developers/weaviate), or a separate schema in [pgvector](https://github.com/pgvector/pgvector)). This provides the strongest isolation guarantee: there is no mechanism by which a query can return results from another tenant's index because the indexes are physically separate. The downsides are operational overhead (you manage N indexes instead of one, each needing monitoring, backup, and capacity management) and resource inefficiency (each index has fixed overhead regardless of size, so small tenants waste resources).

**Shared index with metadata filtering** stores all tenants' vectors in a single index and attaches a `tenant_id` metadata field to every vector. At query time, a metadata filter restricts results to the requesting tenant's vectors. This is operationally simpler and more resource-efficient, especially when you have many small tenants. The risks are that a missing or incorrect filter returns cross-tenant results, and that very large tenants can degrade search performance for smaller tenants sharing the same index. This approach requires rigorous enforcement of the tenant filter at every query path, including any internal or administrative queries. A defense-in-depth approach adds a post-retrieval validation step that verifies every returned chunk belongs to the requesting tenant before it reaches the synthesis prompt.

```python
def retrieve_for_tenant(query_vector: list[float], tenant_id: str, top_k: int = 5):
    """Retrieve chunks with mandatory tenant filtering."""
    results = vector_store.query(
        vector=query_vector,
        top_k=top_k,
        filter={"tenant_id": {"$eq": tenant_id}},  # mandatory filter
    )
    # Defense-in-depth: verify tenant ownership on every result
    verified = [r for r in results if r.metadata.get("tenant_id") == tenant_id]
    if len(verified) < len(results):
        log.warning(
            "Cross-tenant leak prevented",
            expected_tenant=tenant_id,
            leaked_count=len(results) - len(verified),
        )
    return verified
```

**Hybrid approach** uses index-per-tenant for large or high-security tenants and a shared index with metadata filtering for smaller tenants. This balances isolation strength with operational efficiency. The routing logic adds complexity, but it lets you right-size the isolation level to each tenant's requirements and willingness to pay.

The choice depends on your compliance requirements, tenant count, and data volume. Regulated industries (healthcare, finance, government) typically require index-per-tenant or equivalent physical separation. SaaS products with thousands of small tenants typically use shared indexes with metadata filtering. Document the isolation model you have chosen and make it part of your tenant onboarding checklist.

### Rate Limiting And Fair Scheduling

LLM inference is expensive, and a single tenant making large or frequent requests can degrade service for everyone else. This is the **noisy neighbor problem**, and it requires rate limiting at multiple levels.

**Request-level rate limiting** caps the number of API requests a tenant can make per unit of time (requests per minute, requests per hour). This is the simplest control and prevents a single tenant from monopolizing your request queue. Implement it with a token bucket or sliding window algorithm, keyed by tenant identifier. Return a clear error (HTTP 429 with a `Retry-After` header) when limits are exceeded so that clients can back off gracefully.

**Token-level rate limiting** is equally important and often more consequential than request-level limiting because LLM costs scale with token consumption, not request count. A single request with a large context window and a long generation can consume more resources than a hundred short requests. Track input tokens and output tokens separately, and enforce per-tenant budgets on both. Token tracking requires inspecting usage metadata from the model provider's response (all major providers include token counts in their API responses) and maintaining a running tally per tenant.

```python
import time
from collections import defaultdict

class TenantRateLimiter:
    def __init__(self):
        self._windows: dict[str, list[tuple[float, int]]] = defaultdict(list)

    def check_and_record(self, tenant_id: str, tokens: int, config: TenantConfig) -> bool:
        """Returns True if request is within budget, False if throttled."""
        now = time.time()
        window_start = now - 60  # 1-minute sliding window
        # Prune expired entries
        self._windows[tenant_id] = [
            (ts, t) for ts, t in self._windows[tenant_id] if ts > window_start
        ]
        current_usage = sum(t for _, t in self._windows[tenant_id])
        if current_usage + tokens > config.rate_limit_tpm:
            return False
        self._windows[tenant_id].append((now, tokens))
        return True
```

**Fair scheduling** goes beyond rate limiting to ensure equitable access to shared inference resources. When multiple tenants are submitting requests simultaneously, a naive FIFO queue lets a burst from one tenant delay all others. Weighted fair queuing assigns each tenant a share of the available capacity proportional to their tier or contract. Priority queues with per-tenant quotas ensure that no tenant can starve others, while allowing higher-tier tenants to get faster service. In practice, most teams implement this at the application layer with a priority queue backed by a message broker (such as [Redis](https://redis.io/) or [RabbitMQ](https://www.rabbitmq.com/)), with tenant priority and remaining budget as the sorting key.

**Queue isolation** is the strongest form of fair scheduling: each tenant (or tenant tier) gets a dedicated queue with dedicated worker capacity. This eliminates noisy-neighbor effects entirely but is more expensive and only practical for a small number of high-value tenants or tenant tiers.

### Billing And Usage Tracking

Accurate usage tracking is the foundation of multi-tenant billing, cost allocation, and capacity planning. Every model API call must be attributed to a tenant with enough granularity to support per-request auditing.

**Metering** captures token consumption (input and output, separately), model identifier (different models have different per-token costs), request count, latency, tool calls, retrieval queries, and any other billable events. Emit metering events as structured log entries or publish them to a metering pipeline (such as a message queue feeding a time-series database). The metering pipeline must be reliable; lost metering events mean lost revenue or inaccurate cost allocation.

A practical metering record looks like this:

```python
@dataclass
class UsageEvent:
    tenant_id: str
    request_id: str
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    tool_calls: int = 0
    retrieval_queries: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
```

**Cost allocation** translates raw usage into dollar amounts. This requires maintaining a pricing table that maps model identifiers to per-token costs and updating it when provider pricing changes. As of 2026-02-10, provider pricing changes happen frequently (OpenAI, Anthropic, and Google have all adjusted pricing multiple times in the past year), so build your pricing table as configuration, not code. Include a margin for infrastructure overhead (your compute, storage, networking costs) on top of raw provider costs.

**Usage dashboards** give both your operations team and your tenants visibility into consumption patterns. At minimum, show daily and monthly token consumption by model, request volume over time, cost trends, and budget utilization. Expose tenant-facing dashboards through your admin UI so that tenants can self-service usage questions rather than filing support tickets. Alert when a tenant approaches their budget ceiling, both to the tenant and to your operations team, so that usage spikes are addressed proactively rather than resulting in hard cutoffs or surprise invoices.

### Compliance And Data Residency

Multi-tenant AI systems operating across jurisdictions must handle **data residency requirements**: legal constraints on where tenant data can be stored and processed. The [GDPR](https://gdpr.eu/) requires that EU personal data be processed within the EU or in jurisdictions with adequate data protection (with limited exceptions). Similar requirements exist in other jurisdictions, and enterprise customers frequently impose contractual data residency clauses regardless of regulatory requirements.

Data residency in LLM systems is more complex than in traditional applications because data flows through more components: the model API (where is the inference endpoint?), the vector store (where are the embeddings stored?), the conversation history store, the logging pipeline, and any caching layers. Each component must respect the tenant's residency requirements.

**Per-tenant routing** directs each tenant's requests to infrastructure in the appropriate region. This means maintaining model API endpoints, vector store instances, and supporting infrastructure in each required region. As of 2026-02-10, [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models) offers deployments in multiple regions including EU, and [Anthropic](https://docs.anthropic.com/en/docs/about-claude/models) offers EU endpoints through AWS and GCP. Google's [Vertex AI](https://cloud.google.com/vertex-ai/docs/general/locations) supports regional deployments globally. When selecting a provider, verify that the specific models you need are available in the regions your tenants require; not all models are available in all regions.

**Keeping tenant data out of training** is a contractual and regulatory necessity for most enterprise deployments. All major model providers offer options to opt out of data being used for training. [OpenAI's API data usage policy](https://openai.com/policies/usage-policies/) states that API data is not used for training by default. [Anthropic's commercial terms](https://www.anthropic.com/policies/privacy) similarly exclude API data from training. Verify these policies for every provider you use, ensure that the opt-out is active for your account, and document it in your data processing agreements. Enterprise customers will ask for evidence, so maintain records of your provider agreements and their data handling commitments.

**Audit trails for compliance** require logging which data was processed, where, when, and by which model, with enough detail to demonstrate compliance in response to regulatory inquiries. The logging infrastructure must itself comply with residency requirements: logs containing EU tenant data must be stored in the EU. This is easy to overlook when using centralized logging services.

### Multi-Tenant RAG

Multi-tenant RAG is where data isolation and retrieval quality intersect, and getting it wrong has the most visible consequences. The architecture must ensure that Tenant A's proprietary documents never appear in Tenant B's answers, while maintaining the retrieval quality and performance that users expect.

**Shared knowledge bases vs. isolated knowledge bases** is the first design decision. Some content is shared across all tenants (product documentation, public knowledge bases, general reference material), while other content is tenant-specific (internal documents, customer data, proprietary processes). A common architecture maintains a shared knowledge base for common content and per-tenant knowledge bases for tenant-specific content. At query time, the retrieval pipeline searches both the shared base and the tenant's private base, merges results, and feeds them to the synthesis step.

The merging step requires care. Shared content and tenant content may have overlapping topics, and the retrieval pipeline must rank results based on relevance, not source. A simple approach is to retrieve from both sources independently and then re-rank the combined result set. A more sophisticated approach uses the tenant's preference settings to bias toward private content (for questions about internal processes) or shared content (for general product questions).

**Cross-tenant contamination risks** extend beyond vector store isolation. Conversation histories, cached responses, and model context can all leak information between tenants. A cached response generated for Tenant A, if served to Tenant B, may reference Tenant A's documents or data. Cache keys must include the tenant identifier, and cache invalidation must be tenant-aware. Conversation histories must be strictly scoped to the tenant, and session identifiers must not be guessable or reusable across tenants.

Fine-tuned models introduce another contamination vector. If you fine-tune a model on Tenant A's data, that model may reproduce Tenant A's proprietary information in its outputs. Never serve a tenant-fine-tuned model to a different tenant. Track which model each tenant is assigned and validate the mapping on every request.

### Secrets And API Key Management

Each tenant may bring their own model API keys (a "bring your own key" model), or you may provision keys on their behalf. Either way, key management must be rigorous.

**Tenant-provided keys** should be stored in a secrets manager (such as [AWS Secrets Manager](https://aws.amazon.com/secrets-manager/), [HashiCorp Vault](https://www.hashicorp.com/products/vault), or [Google Secret Manager](https://cloud.google.com/secret-manager)) and never in application databases, configuration files, or logs. Retrieve keys at request time from the secrets manager, and do not cache them in application memory longer than necessary. Rotate keys on a defined schedule and support immediate revocation when a tenant reports a compromise.

**Platform-managed keys** (where you provision and manage the model API keys) simplify the tenant experience but concentrate risk. If your key is compromised, all tenants are affected. Use separate keys per provider, per region, and ideally per tenant tier. Monitor key usage for anomalies (unexpected spikes, requests from unusual IP ranges) and rotate proactively.

Never log API keys, even partially. Redact them from error messages, stack traces, and diagnostic output. Validate that your logging pipeline does not capture keys in request headers or environment variables.

### Tenant Onboarding And Offboarding

**Onboarding** a new tenant involves provisioning their configuration, creating their data stores (vector index, conversation history store, usage tracking records), setting up their rate limits and billing, and validating that isolation is working correctly. Automate this process end-to-end. A manual onboarding procedure that works for your first ten tenants will not work for your hundredth.

Build an onboarding validation step that runs a set of smoke tests after provisioning: submit a query and verify it uses the correct model and system prompt, ingest a test document and verify it is retrievable only by the new tenant, verify that rate limiting is active and correctly configured, and verify that usage events are being emitted with the correct tenant identifier.

**Offboarding** is harder and more important than onboarding because it involves data deletion with regulatory implications. When a tenant leaves, you must delete their documents from vector stores, delete their conversation histories, delete their fine-tuned models and training data, delete their configuration and secrets, and retain audit logs for the legally required period (which may differ from the data retention policy). Automate offboarding with a checklist that covers every data store and every region where the tenant's data exists. Verify deletion by querying for the tenant's data after the cleanup process completes. Document the offboarding procedure and its verification so that you can demonstrate compliance.

### Scaling Patterns

Multi-tenant systems face scaling challenges that single-tenant systems do not, primarily because load is unevenly distributed across tenants and usage patterns are difficult to predict.

**The noisy neighbor problem** occurs when one tenant's workload degrades performance for others. In LLM systems, this manifests as queue delays (one tenant's burst of requests delays inference for others), vector store saturation (one tenant's large corpus slows retrieval for all tenants sharing the index), and budget exhaustion (one tenant consuming the provider's rate limit, which is shared across your platform). Mitigations include per-tenant rate limiting (discussed above), queue isolation or weighted fair queuing, and separate vector store instances for tenants with large corpora.

**Resource reservation** guarantees minimum capacity for each tenant by pre-allocating inference slots, vector store capacity, or queue bandwidth. This is the opposite of statistical multiplexing: instead of hoping that not all tenants peak simultaneously, you provision for each tenant's needs individually. Resource reservation is expensive but may be contractually required for enterprise tenants with SLA commitments.

**Horizontal scaling** of multi-tenant LLM systems requires stateless request handling. Tenant configuration, conversation history, and retrieval state must be externalized to shared stores so that any instance can handle any tenant's request. This is standard stateless-service design, but it is worth emphasizing because LLM applications often accumulate in-memory state (cached embeddings, conversation buffers, model connections) that can create affinity between requests and instances.

## Design Notes

### Tenant-Aware Observability

Standard application monitoring gives you system-level metrics: aggregate latency, total error rate, overall throughput. Multi-tenant systems require **tenant-segmented observability**: the ability to answer questions like "what is Tenant X's p95 latency?" and "which tenant is responsible for the cost spike at 2 PM?"

Instrument every request with the tenant identifier as a dimension on all metrics, traces, and logs. Use [OpenTelemetry](https://opentelemetry.io/) attributes to propagate tenant context through distributed traces. Build dashboards that support filtering and grouping by tenant, and set up per-tenant alerting for tenants with SLA commitments.

Key metrics to segment by tenant include request latency (p50, p95, p99), token consumption (input and output), error rate by type, retrieval quality (if you run online evaluation), and cost. Per-tenant SLA dashboards that show uptime, latency compliance, and error budget consumption are valuable both for internal operations and for tenant-facing reporting.

### Audit Logging

Every action in a multi-tenant system must be attributable to a tenant and a user within that tenant. Audit logs serve three purposes: security investigation (who did what and when), compliance demonstration (evidence that controls are operating), and debugging (tracing a bad output to its root cause).

Audit log entries should include the tenant identifier, user identifier, request identifier (for correlation with traces), timestamp, action type (query, document ingest, configuration change, admin action), inputs (sanitized for PII), outputs (sanitized), model and configuration used, and the result (success or failure with reason). Store audit logs in append-only, tamper-evident storage. Apply tenant-appropriate retention policies, and ensure that audit logs for EU tenants are stored in EU-compliant infrastructure.

### Role-Based Access Control Within Tenants

Multi-tenancy operates at two levels: isolation between tenants, and access control within tenants. A tenant's administrators need to manage their team's access to AI features, configure model settings, view usage dashboards, and manage their knowledge base. Regular users within the tenant need to interact with the AI system but should not be able to change configuration or access other users' conversation histories.

Implement a role hierarchy within each tenant. A practical starting point is three roles: **tenant admin** (manages configuration, users, knowledge base, and billing), **user** (interacts with the AI system, views own conversation history), and **viewer** (read-only access to shared resources). Map these roles to specific permissions and enforce them at the API layer, not just in the UI. A role check that only exists in the frontend is not a security control.

## Pitfalls

**Treating tenant isolation as an application-layer concern only.** If your vector store does not support metadata filtering or namespace separation natively, bolting it on at the application layer is fragile. A single missed filter in any code path leaks data. Choose infrastructure that supports tenant isolation as a first-class concept, and verify isolation with automated tests that attempt cross-tenant queries and assert zero results.

**Underestimating the blast radius of shared model API keys.** If you use a single API key across all tenants and the provider rate-limits or suspends that key, every tenant is affected simultaneously. Use separate keys per tenant tier or per tenant where practical, and implement circuit breakers that degrade gracefully for affected tenants without cascading to others.

**Ignoring the cost of per-tenant fine-tuning at scale.** Fine-tuning a model for each tenant sounds like a powerful differentiator, but the operational cost is significant: separate training pipelines, separate model hosting (or dynamic model loading), separate evaluation, and separate rollback capability per tenant. For most multi-tenant systems, per-tenant system prompts and RAG provide 90 percent of the customization benefit at a fraction of the operational cost. Reserve per-tenant fine-tuning for tenants with genuinely unique requirements that cannot be met through prompt engineering or retrieval.

**Caching without tenant awareness.** A semantic cache that does not include the tenant identifier in its key can serve one tenant's cached response to another tenant. This is a data leak. Always include the tenant identifier in cache keys, and invalidate caches per-tenant when configurations change.

**Offboarding as an afterthought.** If you do not design for tenant offboarding from the start, you will discover at the worst possible moment that tenant data is scattered across vector stores, conversation databases, log aggregators, caches, model training datasets, and backup systems across multiple regions. Build the deletion workflow alongside the creation workflow, and test it regularly.

**Assuming uniform usage patterns.** Multi-tenant pricing and capacity planning that assumes all tenants use the system similarly will be wrong. Some tenants will have ten users making occasional queries; others will have automated pipelines making thousands of requests per hour. Instrument usage from the start and build capacity models based on actual observed patterns, not assumptions.

## Checklist
- Is every vector store query filtered by tenant identifier, with post-retrieval verification?
- Is the tenant configuration schema-validated and cached with proper invalidation?
- Are rate limits enforced at both the request level and the token level per tenant?
- Is token consumption metered and attributed to the correct tenant on every request?
- Are API keys stored in a secrets manager and excluded from all logs?
- Is there an automated onboarding workflow with isolation smoke tests?
- Is there an automated offboarding workflow that covers every data store and region?
- Are cache keys tenant-scoped to prevent cross-tenant data leakage?
- Are audit logs tenant-attributed, append-only, and stored in compliant regions?
- Are observability metrics segmented by tenant for SLA monitoring?
- Are fine-tuned models strictly scoped to their owning tenant?
- Is data residency enforced across all components (model API, vector store, logs, caches)?
- Do you have per-tenant role-based access control enforced at the API layer?
- Has cross-tenant isolation been verified with automated tests that assert zero leakage?

## References
- Pinecone: Namespaces for multi-tenancy. https://docs.pinecone.io/guides/indexes/use-namespaces
- Qdrant: Multitenancy guide. https://qdrant.tech/documentation/guides/multiple-partitions/
- Weaviate: Multi-tenancy documentation. https://weaviate.io/developers/weaviate/concepts/data#multi-tenancy
- OpenAI: Data usage and API privacy. https://openai.com/policies/usage-policies/
- OpenAI: Fine-tuning guide. https://platform.openai.com/docs/guides/fine-tuning
- Azure OpenAI: Regional availability. https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models
- HashiCorp Vault: Secrets management. https://www.hashicorp.com/products/vault
- OpenTelemetry: Observability framework. https://opentelemetry.io/
- GDPR official text. https://gdpr.eu/

*Last audited: 2026-02-10 · [Audit methodology](23-audit-methodology.md)*

---
[Contents](README.md) | [Prev](27-ux-design-for-ai.md) | [Next](29-migration-and-vendor-strategy.md)
