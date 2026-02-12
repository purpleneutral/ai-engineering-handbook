# Cost Engineering And Optimization

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](20-multimodal.md) | [Next](22-guardrails-and-moderation.md)

## Summary

LLM API calls are expensive relative to traditional compute, and costs scale with usage in ways that are easy to underestimate. A single poorly designed prompt loop can generate a four-figure bill overnight. Cost engineering is not about being cheap --- it is about spending deliberately, measuring what you spend, and ensuring that every token serves a purpose. The techniques in this chapter (model tiering, prompt caching, batching, structured outputs, and cost monitoring) routinely achieve 60--90% cost reductions compared to naive implementations.

## See Also
- [LLM Fundamentals](01-llm-fundamentals.md) --- Token economics and context window mechanics.
- [Prompting](02-prompting.md) --- Shorter, better prompts reduce both cost and latency.
- [Ops: Shipping And Running LLM Systems](08-ops.md) --- Operational cost control and budgeting.
- [Fine-Tuning And Model Customization](17-fine-tuning.md) --- Distillation as a cost optimization strategy.

## Token Pricing Models

Understanding how providers charge is the foundation of cost engineering. All major providers charge per token, with output tokens costing 2--5x more than input tokens. This asymmetry is important: techniques that reduce output length have outsized cost impact.

### Current Pricing (As of 2026-02-10)

Prices are per million tokens (input / output).

**OpenAI:**

| Model | Input | Output | Cached Input | Batch Input | Batch Output |
|-------|-------|--------|-------------|-------------|--------------|
| [GPT-4.1](https://platform.openai.com/docs/pricing) | $2.00 | $8.00 | $0.50 | $1.00 | $4.00 |
| [GPT-4.1-mini](https://platform.openai.com/docs/pricing) | $0.40 | $1.60 | $0.10 | $0.20 | $0.80 |
| [GPT-4o](https://platform.openai.com/docs/pricing) | $2.50 | $10.00 | $1.25 | $1.25 | $5.00 |
| [GPT-4o-mini](https://platform.openai.com/docs/pricing) | $0.15 | $0.60 | $0.075 | $0.075 | $0.30 |
| [o3](https://platform.openai.com/docs/pricing) | $2.00 | $8.00 | $0.50 | $1.00 | $4.00 |
| [o3-mini](https://platform.openai.com/docs/pricing) | $1.10 | $4.40 | $0.55 | $0.55 | $2.20 |

**Anthropic:**

| Model | Input | Output | Cache Read | Cache Write (5m) | Batch (50% off) |
|-------|-------|--------|-----------|-----------------|-----------------|
| [Claude Opus 4](https://docs.anthropic.com/en/docs/about-claude/models) | $15.00 | $75.00 | $1.50 | $18.75 | Yes |
| [Claude Sonnet 4](https://docs.anthropic.com/en/docs/about-claude/models) | $3.00 | $15.00 | $0.30 | $3.75 | Yes |
| [Claude Haiku 3.5](https://docs.anthropic.com/en/docs/about-claude/models) | $0.80 | $4.00 | $0.08 | $1.00 | Yes |

Anthropic charges a long-context surcharge on requests exceeding 200K input tokens.

**Google:**

| Model | Input | Output | Cache Read | Batch (50% off) |
|-------|-------|--------|-----------|-----------------|
| [Gemini 2.5 Pro](https://ai.google.dev/gemini-api/docs/pricing) | $1.25 | $10.00 | $0.31 | Yes |
| [Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/pricing) | $0.15 | $0.60 | $0.04 | Yes |
| [Gemini 2.0 Flash](https://ai.google.dev/gemini-api/docs/pricing) | $0.10 | $0.40 | $0.025 | Yes |

Google also offers a free tier with rate limits on most Gemini models.

**Important:** These prices change frequently. Always verify against the official pricing pages before making financial decisions. Date-stamp any prices you record.

### The Output Token Premium

Output tokens cost 2--5x more than input tokens across all providers. This is the single most actionable insight in LLM cost engineering: **reducing output length has outsized impact on cost.**

A classification task that returns a free-text paragraph ("The sentiment of this text is positive because the author expresses satisfaction with...") might generate 40 output tokens. The same task returning structured output (`{"sentiment": "positive", "confidence": 0.92}`) generates 12 tokens --- a 70% reduction in the most expensive token type.

## Prompt Caching

When multiple requests share the same prefix (system prompt, few-shot examples, reference documents), prompt caching avoids reprocessing those tokens on every call. The savings are substantial: 50--90% on cached input tokens.

### How Each Provider Implements Caching

**Anthropic: Explicit Cache Control.** You place `cache_control` markers on content blocks. The API caches everything up to each marked block. Cache reads cost 0.1x the base input price (90% discount). Two TTL options: 5 minutes (1.25x write cost) or 1 hour (2x write cost). The TTL refreshes on each cache hit.

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": "You are an expert analyst. Here is the reference manual: ...",
        "cache_control": {"type": "ephemeral", "ttl": "5m"},
    }],
    messages=[{"role": "user", "content": "Summarize chapter 3."}],
)
# response.usage includes: cache_creation_input_tokens, cache_read_input_tokens
```

**OpenAI: Automatic Caching.** No code changes required. The system automatically detects and caches prompt prefixes longer than 1,024 tokens. Cached tokens receive a 50% discount. Cache lifetime is 5--10 minutes of inactivity.

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "... long static instructions ..."},  # cached
        {"role": "user", "content": "Variable user query here"},            # not cached
    ],
)
# response.usage.prompt_tokens_details.cached_tokens shows cache hits
```

**Google: Context Caching with Configurable TTL.** You create a named cache object with a configurable TTL (default 1 hour) and reference it in subsequent requests. Cached reads cost 75% less than base input price. Storage is billed per million cached tokens per hour.

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-2.0-flash",
    config=types.CreateCachedContentConfig(
        system_instruction="You are a legal document analyst.",
        contents=[{"role": "user", "parts": [{"text": large_document}]}],
        ttl="7200s",
    ),
)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Summarize section 4.",
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

### Designing for Cacheability

The key principle is the same across all providers: **place stable content at the beginning of the prompt and variable content at the end.** System instructions, few-shot examples, and reference documents should come first. The user's query should come last. This maximizes the shared prefix and therefore the cache hit rate.

**Sources:**
- [Anthropic Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)
- [Google Context Caching](https://ai.google.dev/gemini-api/docs/caching)

## Batching

Both OpenAI and Anthropic offer batch APIs that process requests asynchronously at a 50% discount. Google offers equivalent batch pricing on Vertex AI.

### When to Use Batching

Batching is appropriate for any workload that does not need real-time responses: eval runs, dataset labeling, bulk classification, content generation pipelines, nightly summarization jobs, and offline content moderation. The tradeoff is latency: results are guaranteed within 24 hours, though they often complete much sooner.

### OpenAI Batch API

Create a JSONL file of requests, upload it, submit a batch job, and download results when complete.

```python
from openai import OpenAI
client = OpenAI()

batch_file = client.files.create(
    file=open("batch_requests.jsonl", "rb"),
    purpose="batch",
)

batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)
# Poll batch.status: "validating" -> "in_progress" -> "completed"
```

### Anthropic Message Batches API

Submit up to 10,000 requests per batch (100,000 on higher tiers). Can be combined with prompt caching for up to 95% total discount.

```python
import anthropic
client = anthropic.Anthropic()

batch = client.batches.create(
    requests=[{
        "custom_id": "request-1",
        "params": {
            "model": "claude-sonnet-4-5-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Classify this text: ..."}],
        },
    }],
)
```

**Sources:**
- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch)
- [Anthropic Batch Processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing)

## Model Tiering and Routing

The single largest cost lever is choosing the right model for each request. Many tasks that teams reflexively send to GPT-4o or Claude Sonnet work equally well on GPT-4o-mini or Haiku at 5--20x lower cost.

### The Tiering Pattern

Route queries to the cheapest model capable of handling them. A classifier or heuristic determines query difficulty, and only "hard" queries go to expensive models.

| Tier | Models | Cost Range (Input) | Best For |
|------|--------|-------------------|----------|
| Cheap | GPT-4o-mini, Haiku 3.5, Gemini 2.0 Flash | $0.08--$0.80/M | Classification, extraction, formatting, simple Q&A |
| Mid | GPT-4o, Sonnet 4, Gemini 2.5 Flash | $0.15--$3.00/M | Summarization, translation, moderate reasoning |
| Expensive | o3, Opus 4, Gemini 2.5 Pro | $2.00--$15.00/M | Complex reasoning, multi-step coding, analysis |

### Routing Approaches

**Rule-based routing.** Route by task type: classification always goes to the cheap model, complex reasoning always goes to the expensive model. Simple, predictable, no overhead.

**LLM-as-classifier.** Use a cheap model to score query difficulty (1--5), then route to the appropriate tier based on the score. Adds one cheap API call per request.

**Trained router.** Use a lightweight ML classifier trained on preference data to predict which model tier a query needs. [RouteLLM](https://github.com/lm-sys/RouteLLM) (UC Berkeley / LMSYS) provides four router architectures trained on Chatbot Arena data, achieving 85% cost reduction while maintaining 95% of GPT-4 quality. It exposes an OpenAI-compatible API for drop-in use.

```python
from routellm.controller import Controller

client = Controller(
    routers=["mf"],  # matrix factorization router
    strong_model="gpt-4o",
    weak_model="gpt-4o-mini",
)
response = client.chat.completions.create(
    model="router-mf-0.5",  # threshold controls strong/weak split
    messages=[{"role": "user", "content": "..."}],
)
```

**Sources:**
- [RouteLLM (LMSYS)](https://github.com/lm-sys/RouteLLM)
- [LLMRouter (UIUC)](https://github.com/ulab-uiuc/LLMRouter)

## Self-Host vs. API Economics

### When APIs Win

APIs win for most teams in most situations. The provider amortizes GPU infrastructure across millions of users, handles scaling, reliability, and model updates, and charges only for what you use. APIs are the right choice when:

- Volume is low to moderate (under ~2 million tokens per day)
- Workloads are variable or unpredictable
- You need frontier model capabilities (GPT-4o, Claude Sonnet, Gemini Pro)
- You use multiple model families

### When Self-Hosting Wins

Self-hosting becomes economical when:

- **High, sustained volume.** Above ~2 million tokens per day with consistent load, the economics shift.
- **Regulatory requirements.** HIPAA, PCI, or data sovereignty mandates that prohibit sending data to third-party APIs.
- **Small to medium open-weight models.** Models under 30B parameters are dramatically cheaper to self-host. As of 2026-02-10, Llama 3.1 8B on an H100 costs approximately $0.03--$0.05 per million input tokens via self-hosting --- compared to $2.50 for GPT-4o. That is a 50--80x difference for tasks where the smaller model is sufficient.
- **Predictable batch workloads.** If you process a consistent volume of documents every night, a dedicated GPU is more economical than per-token API pricing.

### The Hidden Costs of Self-Hosting

The per-token cost comparison understates the real cost of self-hosting. Budget for:

- **GPU infrastructure.** As of 2026-02-10, NVIDIA H100 80GB instances cost $2.85--$3.50/hour on-demand (Lambda Labs, RunPod, CoreWeave). A100 80GB instances cost $1.29--$2.29/hour.
- **Serving stack.** [vLLM](https://docs.vllm.ai/), [SGLang](https://docs.sglang.io/), or TensorRT-LLM for efficient inference. These are complex to configure and maintain.
- **Ops overhead.** GPU drivers, CUDA, health monitoring, scaling, load balancing, model updates, security patches. This is real engineering effort that does not exist with an API.
- **Utilization.** You pay for the GPU whether it is processing tokens or idle. If utilization drops below 50%, the economics often favor APIs.

### Decision Framework

The practical decision is not "API vs. self-host" but "which workloads justify self-hosting?" A common hybrid approach: use APIs for frontier-model tasks and variable workloads, self-host a small open-weight model for high-volume commodity tasks (classification, extraction, formatting).

## Practical Optimization Techniques

### Use Structured Outputs

Output tokens cost 2--5x more than input tokens. Structured outputs (JSON, enums, booleans) produce dramatically fewer output tokens than free-form prose.

```python
# Free-form: ~40 output tokens
# "The sentiment of this text is positive because the author expresses..."

# Structured: ~12 output tokens
# {"sentiment": "positive", "confidence": 0.92}
```

Use `response_format: {"type": "json_schema", ...}` (OpenAI) or tool use with a strict schema (Anthropic) to constrain output shape. See [Structured Outputs and Tool Calling](11-structured-outputs-and-tool-calling.md).

### Shorten Prompts

Every token in the prompt costs money. Remove redundant instructions. Replace verbose few-shot examples with concise ones. Use reference identifiers instead of repeating large context blocks. A prompt that is 30% shorter costs 30% less on input tokens.

### Limit Output Length

Set `max_tokens` to the minimum needed for the task. A classification task needs 5--10 tokens, not 1,000. This prevents verbose responses and caps worst-case cost per request.

### Cache at Multiple Levels

Layer caching for maximum impact:

1. **Application-level response cache.** Cache deterministic outputs (extraction from the same document, repeated identical queries) in your application database.
2. **Semantic cache.** Cache responses for semantically similar queries using embedding similarity. If a query is close enough to a previously answered one, return the cached response.
3. **Provider prompt cache.** Structure prompts with a static prefix for automatic or explicit caching (all three major providers support this).

### Batch Non-Urgent Work

Any workload that does not need real-time responses should use the Batch API for a 50% discount. Common candidates: nightly summarization, dataset labeling, eval runs, content moderation sweeps, report generation.

### Composite Strategy

A well-optimized pipeline combines multiple techniques:

1. **Route** simple queries to a cheap model ($0.15/M) and complex ones to an expensive model ($2.50/M)
2. **Cache** the system prompt prefix (50--90% input discount)
3. **Batch** all offline tasks (50% discount)
4. **Structured outputs** to reduce output tokens by 50--70%
5. **Response caching** for repeated queries

Combined impact: 60--90% cost reduction compared to a naive implementation that sends everything to the most expensive model with no caching.

## Cost Monitoring

Without visibility into what you are spending and where, cost optimization is guesswork. Set up monitoring before you need it.

### Provider Dashboards

All major providers offer built-in usage dashboards with per-model cost breakdowns:
- **OpenAI:** Usage page at [platform.openai.com/usage](https://platform.openai.com/usage)
- **Anthropic:** Console usage dashboard with per-model and per-key tracking
- **Google Cloud:** Vertex AI billing with per-model cost attribution

These dashboards show what you spent but not *why*. For attribution (which feature, which team, which user), you need application-level tracking.

### LiteLLM Proxy

[LiteLLM](https://github.com/BerriAI/litellm) is an open-source proxy that provides a unified OpenAI-compatible API across 100+ LLMs with built-in cost tracking. It supports per-key and per-team budgets with enforcement, virtual API keys with configurable spend limits, and export to Prometheus, Langfuse, OpenTelemetry, and Datadog.

### OpenTelemetry

[OpenLLMetry](https://github.com/traceloop/openllmetry) (Traceloop) provides OpenTelemetry-native instrumentation for LLM applications, including automatic token counting and cost attribution. It extends the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) to capture prompts, completions, token counts, and costs, and exports to any OTel-compatible backend.

### What to Monitor

- **Cost per request** as a first-class metric. Sudden spikes indicate prompt loops, unexpectedly verbose output, or traffic surges.
- **Cache hit rate.** Low hit rates mean your caching strategy is not working.
- **Cost per feature.** Tag requests with project or feature identifiers so you can attribute spend.
- **Weekly cost reports** to catch drift early. Review the top 10 most expensive requests to find optimization opportunities.

## Pitfalls

**Optimizing before measuring.** Do not guess where your costs come from. Instrument first, measure, then optimize the most expensive components. The bottleneck is often not where you expect.

**Defaulting to the most expensive model.** Teams often start with GPT-4o or Claude Sonnet for everything and never re-evaluate. Run evals across model tiers --- many tasks work equally well on models that cost 5--20x less.

**Ignoring output token costs.** Input optimization is important, but output tokens cost 2--5x more. A verbose system prompt instruction ("Please provide a detailed, comprehensive response...") can double your output costs. Be specific about the desired output length and format.

**Forgetting about reasoning model token costs.** Reasoning models (o1, o3, DeepSeek-R1) generate thousands of internal thinking tokens that you pay for. Use them only for tasks that genuinely require multi-step reasoning, not for simple Q&A or classification.

**Not batching offline work.** Every eval run, labeling job, and content generation pipeline that uses the real-time API instead of the Batch API is paying a 50% premium for latency it does not need.

**Running self-hosted GPUs at low utilization.** A GPU that sits idle 80% of the time costs the same as one running at full capacity. If your workload is bursty, APIs are almost certainly cheaper.

## Checklist
- Do you track cost per request and cost per feature as dashboard metrics?
- Have you evaluated whether cheaper models handle your simpler tasks adequately?
- Are system prompts and few-shot examples structured for maximum cache hit rates?
- Are all non-latency-sensitive workloads using the Batch API?
- Do you use structured outputs to minimize output token count?
- Is `max_tokens` set appropriately for each task, not left at the default?
- Do you have budget alerts set at 80% of expected spend?
- Have you compared self-hosting economics for your highest-volume commodity tasks?
- Do you review weekly cost reports and audit expensive requests?

## References
- OpenAI API Pricing. https://openai.com/api/pricing/
- OpenAI Prompt Caching. https://platform.openai.com/docs/guides/prompt-caching
- OpenAI Batch API. https://platform.openai.com/docs/guides/batch
- Anthropic Pricing. https://docs.anthropic.com/en/docs/about-claude/models
- Anthropic Prompt Caching. https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Anthropic Batch Processing. https://platform.claude.com/docs/en/build-with-claude/batch-processing
- Google Gemini Pricing. https://ai.google.dev/gemini-api/docs/pricing
- Google Context Caching. https://ai.google.dev/gemini-api/docs/caching
- RouteLLM (model routing). https://github.com/lm-sys/RouteLLM
- LiteLLM (proxy + cost tracking). https://github.com/BerriAI/litellm
- OpenLLMetry (OTel for LLMs). https://github.com/traceloop/openllmetry

---
[Contents](README.md) | [Prev](20-multimodal.md) | [Next](22-guardrails-and-moderation.md)
