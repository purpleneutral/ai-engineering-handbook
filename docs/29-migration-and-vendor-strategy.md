# Migration And Vendor Strategy

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](28-multi-tenancy.md) | [Next](30-legal-and-ip.md)

## Summary

Every team that builds on LLM APIs will eventually need to switch providers, add a second provider, or renegotiate terms with their current one. The teams that plan for this before it becomes urgent migrate in weeks; the teams that do not plan spend months untangling implicit dependencies they did not know they had. This chapter covers the abstraction patterns, evaluation methodology, migration mechanics, and contractual considerations that make provider transitions manageable rather than traumatic.

## See Also
- [Cost Engineering And Optimization](21-cost-engineering.md)
- [Evals And Testing](05-evals.md)
- [Embeddings And Vector Search](12-embeddings-and-vector-search.md)
- [Fine-Tuning And Model Customization](17-fine-tuning.md)
- [Ops: Shipping And Running LLM Systems](08-ops.md)

## When To Use

Migration planning is relevant the moment you make your first LLM API call in production. You do not need to build a full abstraction layer on day one, but you do need to understand where your lock-in vectors are and make conscious decisions about which dependencies you are willing to accept.

Active migration -- actually switching providers -- becomes necessary when pricing changes make your current provider uneconomical, when a competitor offers meaningfully better quality for your tasks, when regulatory or data residency requirements change, when your provider deprecates the model you depend on, or when reliability problems erode trust. The deprecation scenario is not hypothetical: OpenAI has deprecated multiple model versions with relatively short notice windows, and any provider can do the same.

Multi-provider architectures are worth considering when different tasks in your system have different quality-cost-latency profiles, when you need redundancy for high-availability requirements, or when contractual terms limit your usage of a single provider.

## How It Works

### Abstraction Layers And Provider-Agnostic Interfaces

The first line of defense against vendor lock-in is an abstraction layer that isolates your application logic from provider-specific API details. The goal is to ensure that switching providers requires changing configuration, not rewriting application code.

**The OpenAI-compatible API convention.** The most practical abstraction is not a library but a convention: many providers expose APIs that follow the OpenAI chat completions format. Anthropic, Google, Mistral, and most open-model hosting platforms (Together AI, Fireworks, Groq) either natively support or offer OpenAI-compatible endpoints. If you write your application against this interface, you can switch providers by changing a base URL and API key. This convention has become a de facto standard, and it is the lowest-effort portability strategy available.

**[LiteLLM](https://github.com/BerriAI/litellm)** provides a unified Python SDK and optional proxy server that normalizes the API differences across 100+ LLM providers into a single interface. It handles the translation between provider-specific message formats, streaming protocols, tool calling schemas, and error codes. LiteLLM also provides built-in cost tracking, rate limiting, and fallback routing, which makes it useful beyond pure abstraction.

```python
from litellm import completion

# Switch provider by changing the model string — application code stays the same.
response = completion(
    model="anthropic/claude-sonnet-4-5-20250514",  # or "gpt-4o", "gemini/gemini-2.5-pro", etc.
    messages=[{"role": "user", "content": "Summarize this document."}],
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

**Custom abstraction patterns.** For teams that want tighter control, a thin internal wrapper is straightforward to build. Define an interface with the methods your application actually uses (complete, stream, embed, etc.), implement it for each provider, and inject the implementation at startup. Keep the interface minimal -- do not try to abstract every provider feature. The moment your abstraction tries to normalize prompt caching semantics or provider-specific tool calling formats, it becomes a maintenance burden that outweighs its benefits. Abstract the common path and use provider-specific code paths for features that genuinely differ.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class ChatResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        ...

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def complete(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        usage = response.usage
        return ChatResponse(
            content=response.choices[0].message.content,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            model=response.model,
        )
```

The key discipline is consistency: all LLM calls in your application go through the abstraction. A single direct API call buried in a utility function becomes a migration blocker when you discover it six months later.

### Model Comparison Frameworks And Methodology

Switching providers is an engineering decision, not a vibes-based preference. The methodology for comparing models across providers follows the same principles as any model evaluation, but with additional dimensions that matter for migration decisions.

**Start with your existing eval suite.** If you have been running [evals](05-evals.md) (and you should have been), your golden set is the foundation for model comparison. Run every candidate model against the same inputs and measure the same metrics you use in production. This gives you an apples-to-apples quality comparison on your actual workload, not on generic benchmarks that may not reflect your use case.

**Measure what matters for your application.** Generic leaderboard rankings (MMLU, HumanEval, Chatbot Arena) are useful for initial shortlisting but unreliable for predicting performance on your specific tasks. A model that ranks higher on general benchmarks can easily perform worse on your particular classification task, output format, or domain vocabulary. The only evaluation that counts is one run against your data.

**Control for prompt differences.** The same prompt often performs differently across models. A prompt optimized for GPT-4o may not be optimal for Claude Sonnet, and vice versa. For a fair comparison, spend some time adapting the prompt to each model's strengths before making your quality assessment. This does not mean extensive optimization -- a few iterations to ensure the prompt works idiomatically with each model is sufficient. If you skip this step, you are comparing "a model with an optimized prompt" against "a model with someone else's prompt," which tells you less than you think.

**Measure latency and throughput.** Quality is not the only dimension. Measure time-to-first-token (TTFT), tokens-per-second for streaming, and end-to-end latency at the percentiles that matter for your user experience (p50, p95, p99). Latency varies significantly across providers and models, and it varies by time of day and load. Measure over multiple days, not a single session.

**Calculate total cost of ownership.** Raw per-token pricing is the starting point but not the whole picture. Different tokenizers produce different token counts for the same input, which means a model that appears cheaper per token may actually cost the same or more per request. See the cost comparison section below.

### Migration Planning

A well-executed migration is invisible to users. Achieving that requires parallel running, gradual rollover, and rigorous comparison at each stage.

**Shadow mode (parallel running).** The safest way to validate a new provider is to run it in shadow alongside your production system. Every production request is sent to both the current provider and the candidate. The current provider's response is served to the user; the candidate's response is logged for offline evaluation. This gives you a realistic comparison on production traffic without any user impact.

Shadow mode has a cost -- you are paying for two providers simultaneously -- but it eliminates the risk of discovering quality problems only after you have migrated. Run shadow mode for long enough to capture the full diversity of your production traffic, including edge cases, peak loads, and any weekly or seasonal patterns. Two to four weeks is typical for most workloads.

```python
import asyncio

async def shadow_request(
    primary: LLMProvider,
    shadow: LLMProvider,
    messages: list[ChatMessage],
    **kwargs,
) -> ChatResponse:
    """Send request to both providers; return primary, log shadow for comparison."""
    primary_task = asyncio.create_task(
        asyncio.to_thread(primary.complete, messages, **kwargs)
    )
    shadow_task = asyncio.create_task(
        asyncio.to_thread(shadow.complete, messages, **kwargs)
    )

    primary_response = await primary_task

    # Don't let shadow failures affect the user.
    try:
        shadow_response = await shadow_task
        log_comparison(messages, primary_response, shadow_response)
    except Exception as e:
        log_shadow_error(e)

    return primary_response
```

**Gradual rollover.** After shadow mode confirms that the candidate meets your quality bar, migrate traffic incrementally. Start with 1-5% of production traffic, monitor quality metrics and error rates, and increase the percentage in stages (10%, 25%, 50%, 100%). At each stage, compare metrics against the baseline and have a clear rollback trigger: if error rates exceed a threshold or quality metrics drop below a floor, revert immediately. This is the same canary deployment pattern described in [Ops](08-ops.md), applied to a provider migration.

**Feature parity checklist.** Before starting migration, enumerate every provider-specific feature you use: streaming, tool calling, structured outputs, prompt caching, batch API, image inputs, system message handling, stop sequences, logprobs, and any beta features. Verify that each feature either has an equivalent on the target provider or that you have a workaround. Missing features that you discover mid-migration are the most common cause of delays.

### Prompt Portability

Prompts are the most deceptively portable artifact in an LLM system. The same prompt text can produce dramatically different results across models because models have different instruction-following strengths, different sensitivities to prompt structure, and different default behaviors.

**The portability-performance tension.** You can write prompts that work acceptably across multiple models, or you can write prompts that are optimized for a specific model. You generally cannot do both. A prompt optimized for Claude might rely on XML-tag delimiters and detailed role descriptions, while a prompt optimized for GPT-4o might use concise instructions and rely on the model's strong JSON mode. A "portable" prompt that avoids model-specific patterns will likely underperform a tuned prompt on every model.

The practical resolution is to maintain a single **canonical prompt** that expresses the intent and requirements, and then allow per-model **prompt adaptations** that adjust structure, delimiters, and phrasing for each target model. Store both the canonical prompt and the adaptations in version control. When you migrate, you write a new adaptation for the target model rather than trying to force the old prompt to work.

**What is portable.** The task definition, the input/output schema, the examples in your golden set, and the evaluation criteria are all portable. These represent what you want the model to do, and they transfer across providers unchanged.

**What is not portable.** Specific prompt engineering techniques (XML tags vs. markdown headers for structure, "think step by step" vs. more elaborate chain-of-thought scaffolding), system message conventions, temperature and sampling parameter choices, and the ordering of instructions relative to data -- these all require per-model tuning. Accept this cost as part of migration.

### Embedding Model Migration

Switching embedding models is the most painful migration in an LLM system because embeddings from different models are not compatible. You cannot query a vector index built with one model using embeddings from another -- the vectors live in different geometric spaces. This means that changing your embedding model requires re-embedding your entire corpus and rebuilding your index, which for large corpora can cost thousands of dollars in API calls and take days or weeks of pipeline time.

**The reindexing problem.** For a corpus of 10 million chunks at 500 tokens each, re-embedding with OpenAI's `text-embedding-3-small` costs approximately $100 (as of 2026-02-10 at $0.02 per million tokens). That is manageable. But for 100 million chunks, or with a more expensive embedding model, or when you factor in the engineering time to run the pipeline, validate the new index, and coordinate the cutover, the cost becomes significant. And during the reindexing window, you either serve stale results or run a complex dual-index system.

**Dual-index strategy.** Run two indexes in parallel: the old index (with the old embedding model) and the new index (with the new embedding model). New documents are embedded with both models and inserted into both indexes. Queries are run against both indexes during a transition period, with results merged or with the new index gradually taking over. Once the new index contains all documents and has been validated, decommission the old index. This approach avoids any gap in coverage but doubles your storage and compute costs during the transition.

**Lazy migration.** Instead of re-embedding the entire corpus upfront, embed documents on access. When a document is retrieved from the old index, re-embed it with the new model and insert it into the new index. Over time, the new index accumulates coverage of the most-accessed documents. This works well when access patterns follow a power law (a small fraction of documents account for most retrievals), but it means rarely accessed documents may never migrate, so you need to complement it with a background batch job that catches the long tail.

**Prevention.** The best strategy for embedding migration is to minimize the frequency of migrations. Choose an embedding model with a strong track record and a provider likely to maintain backward compatibility. Version your embedding model explicitly and include the model version in your index metadata so you can detect mismatches. And factor re-embedding cost into your embedding model selection -- a slightly less capable model that you can afford to re-embed is more valuable than a perfect model that locks you in permanently. See [Embeddings And Vector Search](12-embeddings-and-vector-search.md) for more on index management.

### Vendor Lock-In Vectors

Lock-in accumulates gradually through a series of individually reasonable decisions. Understanding where lock-in concentrates helps you make conscious trade-offs rather than discovering dependencies at migration time.

**Proprietary fine-tuned models.** A model fine-tuned through OpenAI's API is hosted on OpenAI's infrastructure and cannot be exported. Your training data is portable, but the resulting model weights are not. If you fine-tune, keep your training data, evaluation suite, and training configuration versioned and reproducible so that you can re-fine-tune on a different provider's base model if needed. See [Fine-Tuning And Model Customization](17-fine-tuning.md).

**Provider-specific features.** Prompt caching, batch APIs, structured output modes, and extended thinking features all have provider-specific implementations. Some of these features deliver significant cost or quality advantages that justify the lock-in. The discipline is to know which features you depend on, quantify the benefit they provide, and have a plan for how you would operate without them.

**Custom model IDs and versioning.** Pinning to a specific model snapshot (e.g., `gpt-4o-2024-08-06`) is good practice for reproducibility, but it creates a dependency on that provider's versioning scheme. Abstract the model identifier behind a configuration value so that it can be changed without code changes.

**Provider-specific tool calling formats.** OpenAI, Anthropic, and Google each have slightly different schemas for defining tools and parsing tool call results. If your tool definitions are hard-coded against one provider's format, migrating requires rewriting them. LiteLLM normalizes these differences, or you can maintain tool definitions in a provider-agnostic format and translate at the edge.

**Data residency and processing agreements.** If you have negotiated specific data processing terms with your current provider (data residency guarantees, zero-retention agreements, HIPAA Business Associate Agreements), you will need equivalent agreements with any new provider before you can migrate. These negotiations take weeks to months, so start them well before you need them.

### Cost Comparison Methodology

Comparing costs across providers is harder than it looks because providers differ in ways that make naive per-token comparisons misleading.

**Different tokenizers produce different token counts.** The same input text produces different token counts depending on the tokenizer. OpenAI's `cl100k_base` and `o200k_base`, Anthropic's tokenizer, and Google's tokenizer all segment text differently. A prompt that is 1,000 tokens on OpenAI might be 1,100 tokens on Anthropic or 950 tokens on Google. To do an accurate comparison, tokenize your actual production prompts with each provider's tokenizer and compare the actual cost per request, not the cost per abstract token.

**Different context window sizes affect architecture.** A model with a 200K-token context window and a model with a 32K-token context window may have similar per-token prices, but if your use case requires processing long documents, the cheaper model might force you to add a summarization or chunking layer that changes the total cost picture.

**Caching economics differ.** As of 2026-02-10, Anthropic charges 1.25x for cache writes and 0.1x for cache reads. OpenAI provides automatic caching at a 50% discount. Google offers configurable TTL caching at a 75% discount on reads. The total cost depends on your cache hit rate, which depends on your traffic patterns. Model the cost using your actual request distribution, not just the listed prices.

**Batch pricing varies.** Both OpenAI and Anthropic offer 50% batch discounts, but the batch size limits, completion time guarantees, and eligible models differ. If a significant fraction of your workload can be batched, these differences matter.

**The practical approach.** Sample 1,000 representative production requests. Run them through each candidate provider. Calculate the total cost including caching, batching, and any volume discounts. This gives you a real number instead of a spreadsheet estimate that falls apart on contact with reality. See [Cost Engineering And Optimization](21-cost-engineering.md) for a detailed framework.

### Multi-Provider Architectures

Running multiple providers simultaneously is not just a migration strategy -- it can be a steady-state architecture that optimizes cost, quality, and reliability across different tasks.

**Task-based routing.** Different tasks have different optimal providers. A summarization task might perform best on Claude Sonnet, while a code generation task might perform best on GPT-4o, and a simple classification task might be cheapest on Gemini Flash. Route each task to the provider that offers the best trade-off for that specific use case. This requires maintaining prompts and eval suites for each provider-task combination, which increases operational complexity. The trade-off is worth it only when the quality or cost differences are substantial.

**Fallback chains.** Configure a primary provider and one or more fallback providers for each task. If the primary provider returns an error, times out, or hits a rate limit, automatically retry with the fallback. This improves availability without requiring the fallback to match the primary's quality on every request -- the fallback is a degraded-but-functional path, not a full replacement.

```python
from litellm import completion

# LiteLLM supports fallback configuration natively.
response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Analyze this data."}],
    fallbacks=["anthropic/claude-sonnet-4-5-20250514", "gemini/gemini-2.5-flash"],
    max_tokens=1024,
)
```

**Load balancing for rate limits.** If you are hitting rate limits with a single provider, distributing requests across multiple providers increases your effective throughput ceiling. This is particularly relevant during traffic spikes or when running large batch jobs that compete with real-time traffic for rate limit capacity.

**Consistency considerations.** Multi-provider architectures introduce a new class of issues: different models produce different outputs for the same input, which can create inconsistent user experiences. If a user interacts with your system multiple times and gets responses from different providers, the tone, format, and level of detail may shift noticeably. Mitigate this with strong output schemas, post-processing normalization, and per-user or per-session provider stickiness.

### Contract And SLA Considerations

Provider relationships have a commercial dimension that engineering teams sometimes overlook. Understanding what you are contractually entitled to helps you plan around provider limitations and negotiate better terms.

**Uptime and availability.** As of 2026-02-10, most LLM API providers offer uptime targets in the range of 99.5-99.9%, but the specifics vary. Check whether the SLA covers the specific model you use (some SLAs apply only to certain model tiers), whether it includes degraded performance (slow responses may not count as downtime), and what remedies are available when the SLA is breached (typically service credits, not refunds).

**Rate limits.** Rate limits are a practical constraint that can bottleneck your application. Understand the limits for your tier (requests per minute, tokens per minute, concurrent connections), whether they can be increased through a paid tier or negotiation, and how the provider communicates changes to rate limits. A rate limit reduction with insufficient notice can break a production system.

**Data processing agreements.** Know how the provider handles your data. Key questions: Is your data used for model training? (Most providers offer opt-out; some require opt-in.) How long is data retained? Where is data processed and stored? Does the provider offer a [Data Processing Agreement](https://gdpr.eu/what-is-data-processing-agreement/) (DPA) that meets your regulatory requirements? For healthcare, does the provider sign a HIPAA Business Associate Agreement? Get these answers in writing before sending production data.

**Model deprecation policy.** Understand the provider's policy for deprecating models. How much notice do they give? Do they offer migration paths? Are there extended support tiers for enterprise customers? A provider that deprecates models with 90 days' notice gives you time to migrate; a provider that gives 30 days creates a fire drill.

### Exit Planning

Exit planning is the practice of preparing for provider transitions before you need them. Like disaster recovery planning, exit planning is an investment that pays off precisely when you can least afford to be unprepared.

**Maintain provider-agnostic artifacts.** Your eval suite, golden set, training data, tool definitions, output schemas, and documentation should all be provider-agnostic. These are the assets you need to set up with a new provider, and they should be ready to use without translation.

**Keep prompts in version control with provider metadata.** Store prompts alongside the model identifier they were tested with. When you migrate, you will need to know which prompts were optimized for which model so you can plan prompt adaptation work.

**Run periodic cross-provider evals.** Even when you are not actively migrating, run your eval suite against alternative providers quarterly. This gives you a current view of the competitive landscape, surfaces quality improvements from providers you are not using, and keeps your comparison methodology sharp. The incremental cost is small and the information value is high.

**Pre-negotiate agreements.** If you are in a regulated industry or have specific data handling requirements, initiate conversations with backup providers before you need them. Legal and procurement processes take time, and you do not want to be waiting on a DPA when your primary provider is having an outage.

**Document your lock-in inventory.** Maintain a list of every provider-specific dependency in your system: fine-tuned models, cached embeddings, provider-specific features in use, contractual commitments, and data residency assumptions. Review this list quarterly. For each dependency, note the migration cost (in time and money) and the trigger conditions that would make migration necessary. This document transforms exit planning from an abstract concern into a concrete, actionable assessment.

## Design Notes

The strongest teams treat vendor relationships as engineering decisions with ongoing trade-offs, not one-time procurement events. They invest in abstraction where it is cheap and provides clear returns (unified API interfaces, provider-agnostic eval suites), they accept lock-in where it delivers substantial benefits (prompt caching, provider-specific optimizations), and they maintain the information and tooling needed to change course if the trade-offs shift.

Resist the temptation to build a maximally portable architecture upfront. The cost of abstraction is real -- it adds complexity, it limits access to provider-specific features that might improve quality or reduce costs, and it requires ongoing maintenance as providers evolve their APIs. Build the minimum abstraction that meets your current needs and extend it when concrete migration requirements emerge.

## Pitfalls

**Treating abstraction as a substitute for testing.** A unified API interface lets you switch the provider string, but it does not guarantee that your system works the same way with the new provider. Every provider switch requires re-running your full eval suite and validating quality, latency, and cost on production-representative traffic. The abstraction eliminates code changes; it does not eliminate validation work.

**Optimizing for portability at the expense of quality.** Teams sometimes avoid provider-specific features (prompt caching, structured output modes, extended thinking) because they reduce portability. This trades a concrete quality or cost improvement today for optionality you may never exercise. Use provider-specific features when they deliver measurable benefits, and document the dependency so you can plan around it if you migrate.

**Underestimating embedding migration cost.** Teams that casually switch embedding models without accounting for the full re-embedding pipeline -- API costs, pipeline engineering time, dual-index operation, validation, and cutover coordination -- routinely underestimate the effort by 3-5x. Budget for it explicitly and treat embedding model selection as a higher-commitment decision than LLM selection.

**Comparing models without controlling for prompt optimization.** Running your GPT-4o-optimized prompt on Claude Sonnet and concluding that "Claude is worse" is not a valid comparison. Spend a few hours adapting the prompt to each model before drawing conclusions. The difference between a poorly adapted prompt and a well-adapted one is often larger than the difference between models.

**Ignoring the contractual dimension.** Engineering teams sometimes evaluate providers purely on technical merit and discover too late that the procurement process, data processing agreement, or regulatory compliance check takes months. Start the commercial conversation in parallel with the technical evaluation, not after it.

**Assuming all OpenAI-compatible APIs behave identically.** The OpenAI-compatible API convention covers the request/response format, but it does not guarantee identical behavior for edge cases: streaming error handling, tool calling with complex schemas, token counting discrepancies, and timeout behavior all vary. Test the specific flows your application uses, not just the basic chat completion path.

## Checklist
- Do all LLM calls go through an abstraction layer or unified interface?
- Can you switch providers by changing configuration, not code?
- Do you have a provider-agnostic eval suite that you run against candidate models?
- Have you measured the prompt adaptation cost for at least one alternative provider?
- Do you know the migration cost for your embedding indexes?
- Do you maintain a lock-in inventory documenting every provider-specific dependency?
- Do you have a data processing agreement with your current provider?
- Do you understand your provider's model deprecation policy and timeline?
- Have you pre-negotiated or at least initiated contact with at least one backup provider?
- Can you run shadow mode against a new provider without affecting production?

## References
- LiteLLM (unified LLM API proxy). https://github.com/BerriAI/litellm
- OpenAI API reference (de facto standard interface). https://platform.openai.com/docs/api-reference
- OpenAI model deprecation policy. https://platform.openai.com/docs/deprecations
- Anthropic API reference. https://docs.anthropic.com/en/api/getting-started
- Google Gemini API reference. https://ai.google.dev/gemini-api/docs
- OpenAI Prompt Caching. https://platform.openai.com/docs/guides/prompt-caching
- Anthropic Prompt Caching. https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

*Last audited: 2026-02-10 · [Audit methodology](23-audit-methodology.md)*

---
[Contents](README.md) | [Prev](28-multi-tenancy.md) | [Next](30-legal-and-ip.md)
