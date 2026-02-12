# Caching And Latency Optimization

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](25-data-pipelines.md) | [Next](27-ux-design-for-ai.md)

## Summary

LLM API calls are slow relative to traditional backend operations -- a typical chat completion takes 500ms to 5s, compared to single-digit milliseconds for a database query. Latency is the primary driver of user experience in AI-powered applications, and caching is the primary tool for managing it. This chapter covers the full latency optimization stack: provider-level prompt caching (which is covered from a cost angle in [Cost Engineering](21-cost-engineering.md) -- here we focus on the latency mechanics and implementation patterns), semantic and response caching, streaming, model selection for speed-sensitive paths, parallel execution strategies, and the instrumentation needed to measure and improve end-to-end performance.

## See Also
- [Cost Engineering And Optimization](21-cost-engineering.md) -- Prompt caching pricing, batching economics, and model tiering from a cost perspective.
- [Ops: Shipping And Running LLM Systems](08-ops.md) -- Observability, tracing, and operational dashboards.
- [Architecture Recipes](07-architecture-recipes.md) -- Common system patterns where latency optimization applies.
- [Prompting](02-prompting.md) -- Shorter prompts reduce both cost and latency.
- [LLM Fundamentals](01-llm-fundamentals.md) -- Token mechanics, context windows, and the quadratic attention cost that drives latency.

## When To Use

Every production LLM application benefits from latency optimization, but the investment profile varies. Interactive applications (chatbots, copilots, real-time assistants) demand aggressive optimization because users perceive delays as low quality -- research consistently shows that response times above 1-2 seconds degrade user satisfaction. Backend pipelines (extraction, classification, summarization) are less latency-sensitive but still benefit from caching to reduce redundant API calls and improve throughput.

Prioritize latency work when: your P95 response time exceeds your user experience budget, your API costs are dominated by repeated or similar queries, you are building a user-facing conversational interface, or you are orchestrating multi-step agent workflows where latency compounds across steps.

## How It Works

### Prompt Caching: The Latency Angle

[Cost Engineering](21-cost-engineering.md) covers prompt caching mechanics and pricing in detail. Here we focus on the latency implications, which are significant and often overlooked.

When a provider caches a prompt prefix, it is not just saving you money -- it is skipping the computation of processing those cached tokens through the model's attention layers. This translates directly to reduced **time-to-first-token (TTFT)**, which is the delay between sending a request and receiving the first token of the response. TTFT is the metric users feel most acutely because it determines how quickly the response appears to begin.

**Anthropic** reports that prompt caching reduces TTFT by up to 85% for long prompts. A 10,000-token system prompt that normally takes ~800ms to process might take ~120ms on a cache hit. The effect is most dramatic with long prefixes: the longer the cached portion, the greater the TTFT reduction. Anthropic's explicit `cache_control` markers give you precise control over what gets cached, making it possible to optimize the cache boundary for your specific prompt structure.

**OpenAI** automatically caches prompt prefixes longer than 1,024 tokens. As of 2026-02-10, OpenAI's documentation states that cached requests have "latency that is on the order of the non-cached portion of the prompt," meaning the processing time scales only with the new, uncached tokens. There is no code change required to benefit, but you must structure prompts with the stable prefix first and variable content last.

**Google** context caching works differently: you create a named cache object via the API and reference it in subsequent requests. The latency benefit comes from the model not having to reprocess the cached content. Google's approach is best suited for scenarios where you have a large, stable context (a long document, a detailed knowledge base) that many requests share.

The practical implication across all providers is the same: **prompt structure determines cache effectiveness.** Place your system prompt, few-shot examples, and reference documents at the beginning. Place the user's variable query at the end. Even small reorderings can break prefix matching and eliminate cache hits entirely.

```python
# Pattern: maximize cacheable prefix
# Everything above the variable query is cache-eligible

import anthropic

client = anthropic.Anthropic()

SYSTEM_BLOCKS = [
    {
        "type": "text",
        "text": "You are an expert financial analyst. Here are the reference documents:\n\n"
                + large_reference_corpus,  # 8,000+ tokens of stable content
        "cache_control": {"type": "ephemeral"},
    }
]

def analyze(user_query: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1024,
        system=SYSTEM_BLOCKS,
        messages=[{"role": "user", "content": user_query}],
    )
    # Check cache performance
    usage = response.usage
    cache_hit = usage.cache_read_input_tokens > 0
    # cache_hit == True means reduced TTFT for this request
    return response.content[0].text
```

### Semantic Caching

**Semantic caching** returns a previously generated response when a new query is semantically similar to a cached one, even if the wording differs. "What's the capital of France?" and "Which city is the capital of France?" are different strings but have identical intent, and a semantic cache can serve the stored answer for the second query without making an API call at all.

The mechanism works by embedding each incoming query, comparing it against embeddings of cached queries using cosine similarity (or another distance metric), and returning the cached response if the similarity exceeds a configured threshold. This is conceptually simple but requires careful tuning.

**How to implement it.** The core loop is: embed the query, search the cache index for similar queries, and either return a cached response or forward to the LLM and store the result.

```python
import json
import numpy as np
from openai import OpenAI

client = OpenAI()

class SemanticCache:
    """Minimal semantic cache using embeddings for similarity lookup."""

    def __init__(self, similarity_threshold: float = 0.95):
        self.threshold = similarity_threshold
        self.cache: list[dict] = []  # production: use a vector DB

    def _embed(self, text: str) -> list[float]:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get(self, query: str) -> str | None:
        query_embedding = self._embed(query)
        best_score = 0.0
        best_response = None
        for entry in self.cache:
            score = self._cosine_similarity(query_embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_response = entry["response"]
        if best_score >= self.threshold:
            return best_response
        return None

    def put(self, query: str, response: str) -> None:
        self.cache.append({
            "query": query,
            "embedding": self._embed(query),
            "response": response,
        })
```

**Threshold tuning is critical.** Set the similarity threshold too low and you return irrelevant cached responses. Set it too high and you get almost no cache hits. There is no universal correct value -- it depends on your domain and query distribution. Start at 0.95, evaluate on a sample of real queries, and adjust. Always log cache hits with the similarity score so you can audit false matches.

[GPTCache](https://github.com/zilliztech/GPTCache) is an open-source library that implements semantic caching with pluggable embedding models, similarity functions, and storage backends. It integrates with OpenAI and LangChain and handles the embedding, similarity search, and cache management plumbing. It is a reasonable starting point, though production deployments often outgrow it and implement custom caching with a vector database like [Qdrant](https://qdrant.tech/documentation/) or [Weaviate](https://weaviate.io/developers/weaviate).

**When semantic caching breaks down.** Semantic caching is poorly suited for queries where small wording differences change the correct answer ("revenue in Q1 2025" vs. "revenue in Q2 2025"), for time-sensitive queries where the answer changes over time, for personalized queries where different users should get different responses, and for any query where the system prompt or context differs between requests. In these cases, exact-match caching (keyed on the full prompt hash) or no caching is safer.

### Response Caching (Exact-Match And Key-Based)

The simplest and most reliable caching strategy is **exact-match response caching**: hash the complete request (model, messages, temperature, tools, and all other parameters), store the response keyed by that hash, and return it on subsequent identical requests. This has zero risk of returning a wrong answer because the cache key includes every input that could affect the output.

```python
import hashlib
import json
import redis

r = redis.Redis()

def cache_key(model: str, messages: list[dict], **kwargs) -> str:
    """Deterministic cache key from the full request."""
    payload = json.dumps(
        {"model": model, "messages": messages, **kwargs},
        sort_keys=True,
    )
    return f"llm:v1:{hashlib.sha256(payload.encode()).hexdigest()}"

def cached_completion(model: str, messages: list[dict], ttl: int = 3600, **kwargs):
    key = cache_key(model, messages, **kwargs)
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    response = client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    r.setex(key, ttl, json.dumps(response.model_dump()))
    return response
```

Exact-match caching works best for deterministic workloads: extraction pipelines processing the same documents, classification of known inputs, and any scenario where temperature is 0 and the prompt is identical. The hit rate depends entirely on how often you see the same input twice.

**Key-based caching** is a middle ground: instead of hashing the entire request, you define a cache key from a subset of the inputs that you know determines the output. For example, in a document summarization pipeline, the cache key might be the document's content hash plus the prompt version, ignoring the timestamp or request metadata. This is more flexible but requires you to reason carefully about what inputs actually affect the output.

### Streaming Responses

**Streaming** is the single most impactful latency perception technique for interactive applications. Instead of waiting for the model to generate the complete response before sending anything to the client, streaming delivers tokens as they are generated. The user sees text appearing in real time, which transforms a 3-second wait into an experience that feels responsive from the first 200ms.

All major providers support streaming via **Server-Sent Events (SSE)**, a lightweight protocol built on HTTP where the server sends a series of `data:` events over a long-lived connection. The client reads each event as it arrives and renders the partial response incrementally.

```python
from openai import OpenAI

client = OpenAI()

# Streaming with the OpenAI SDK
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain gradient descent."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

```python
import anthropic

client = anthropic.Anthropic()

# Streaming with the Anthropic SDK
with client.messages.stream(
    model="claude-sonnet-4-5-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain gradient descent."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

**Proxying streams to web clients.** When your backend sits between the LLM API and the browser, you need to forward the SSE stream without buffering. In Python with FastAPI:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI

app = FastAPI()
client = OpenAI()

@app.get("/chat")
async def chat(query: str):
    def generate():
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield f"data: {delta.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

**Common streaming pitfalls.** Reverse proxies (nginx, Cloudflare) can buffer SSE responses if not configured correctly. In nginx, set `proxy_buffering off;` and `X-Accel-Buffering: no`. Cloudflare requires disabling response buffering in the dashboard or via page rules. If your users report that responses appear all at once after a delay, the problem is almost always a buffering proxy, not the LLM API.

### Client-Side Latency Perception

Even with streaming, there are moments where the user waits: the initial connection, the TTFT gap, tool call execution pauses. Client-side techniques can make these waits feel shorter.

**Progressive rendering** means displaying partial results as they become available. In a RAG system, show the retrieved source documents immediately while the synthesis is still generating. In an agent workflow, show each tool call and its result as it happens rather than waiting for the final answer. This gives users continuous feedback and makes the system feel faster even when total latency is unchanged.

**Skeleton screens and typing indicators** bridge the gap before the first token arrives. A pulsing "thinking" indicator or a skeleton layout communicates that the system is working. The key is to display something within 100-200ms of the user's action -- if the UI goes blank for a full second before showing a loading state, users perceive it as broken.

**Optimistic UI updates** can work for predictable operations. If the user asks the system to add an item to a list, update the UI immediately and reconcile with the model's response when it arrives. This pattern is borrowed from traditional web development and works well for structured actions where the outcome is predictable.

### Model Selection For Latency

Model size is a primary determinant of latency. Larger models have more parameters, which means more computation per token and higher TTFT. The relationship is roughly linear: a model with 4x the parameters takes roughly 4x as long per token (though providers optimize this with hardware and batching).

For latency-sensitive paths, use the smallest model that meets your quality requirements. As of 2026-02-10, typical TTFT ranges across providers are:

| Model Tier | Examples | Typical TTFT | Tokens/sec |
|------------|----------|-------------|------------|
| Small/Fast | GPT-4o-mini, Claude Haiku 3.5, Gemini 2.0 Flash | 100-300ms | 80-150 |
| Mid | GPT-4o, Claude Sonnet 4, Gemini 2.5 Flash | 200-600ms | 50-100 |
| Large/Reasoning | o3, Claude Opus 4, Gemini 2.5 Pro | 500ms-5s+ | 20-60 |

These numbers vary with prompt length, server load, and geographic distance to the provider's data center. Measure in your own environment -- do not rely on published benchmarks.

A common production pattern is **model tiering by latency budget**: use a fast model for the initial response in a conversation (where TTFT matters most) and a more capable model for complex follow-up questions where the user has already committed to waiting. This can be implemented with the same routing patterns described in [Cost Engineering](21-cost-engineering.md), but with latency thresholds instead of cost thresholds.

### Parallel Tool Calls

When an LLM invokes multiple tools in a single turn, executing them sequentially adds their latencies together. If each tool call takes 200ms and the model requests three, that is 600ms of sequential tool execution before the model can generate its response.

**Parallel tool execution** runs independent tool calls concurrently, reducing the wall-clock time to the duration of the slowest call. OpenAI supports this natively: when `parallel_tool_calls` is enabled (which is the default), the model can return multiple tool calls in a single response, and your code should execute them concurrently.

```python
import asyncio
import json
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def execute_tool(call) -> dict:
    """Dispatch a single tool call to the appropriate handler."""
    name = call.function.name
    args = json.loads(call.function.arguments)
    # Route to the appropriate tool implementation
    result = await tool_registry[name](**args)
    return {
        "tool_call_id": call.id,
        "role": "tool",
        "content": json.dumps(result),
    }

async def agent_step(messages: list[dict]) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tool_definitions,
    )
    message = response.choices[0].message

    if message.tool_calls:
        # Execute all tool calls concurrently
        tool_results = await asyncio.gather(
            *[execute_tool(call) for call in message.tool_calls]
        )
        messages.append(message.model_dump())
        messages.extend(tool_results)
        # Continue the conversation with tool results
        return await agent_step(messages)

    return message.content
```

For Anthropic, the model may return multiple tool use blocks in a single response. The same pattern applies: gather all tool use blocks, execute concurrently, and return all results in a single follow-up message.

Not all tool calls can run in parallel. If the result of one tool call determines the arguments for another, they must be sequential. The model generally handles this correctly -- it will issue dependent calls in separate turns. But if you have custom orchestration logic, you need to identify dependencies and parallelize only the independent calls.

### Speculative Execution

**Speculative execution** is a technique borrowed from CPU architecture: start work before you know it will be needed, and discard it if the speculation was wrong. In LLM systems, this takes several forms.

**Speculative decoding** is a provider-side optimization where a small "draft" model generates candidate tokens quickly, and the large target model verifies them in a single forward pass. If the draft tokens are correct (which they often are for predictable sequences), multiple tokens are confirmed in one step, significantly increasing throughput. As of 2026-02-10, this is an internal optimization at most providers -- you do not control it directly, but it is one reason why newer model versions often have lower latency than their predecessors.

**Application-level speculation** is something you control. In a multi-step agent workflow, you can start the next likely step before the current step completes. For example, if step 1 is "retrieve documents" and step 2 is almost always "summarize the retrieved documents," you can begin generating the summarization prompt template while the retrieval is still running. Similarly, if your system typically follows a user's question with a knowledge base lookup, you can start the embedding and retrieval in parallel with the LLM's processing of the user's message.

```python
import asyncio

async def speculative_rag(query: str) -> str:
    """Start retrieval and initial LLM processing concurrently."""
    # Speculate: the model will probably need these documents
    retrieval_task = asyncio.create_task(retrieve_documents(query))
    # Also start a fast classifier to determine query intent
    intent_task = asyncio.create_task(classify_intent(query))

    documents, intent = await asyncio.gather(retrieval_task, intent_task)

    # Now we have both results ready for the main LLM call
    # No sequential wait: retrieval and classification ran in parallel
    response = await generate_response(query, documents, intent)
    return response
```

The risk of speculation is wasted compute: if the speculation is wrong, you pay for work that gets discarded. This is acceptable when the speculated work is cheap (an embedding call, a cache lookup) and the latency savings are large (avoiding a sequential round-trip).

### Request Batching Vs. Real-Time

Not every request needs real-time processing. Separating your workload into real-time and batch paths is one of the most effective architectural decisions for latency optimization, because it lets you optimize each path independently.

**Real-time path:** User-facing requests that need sub-second TTFT. Use streaming, prompt caching, fast models, and aggressive response caching. Keep the pipeline short -- fewer sequential steps means lower latency.

**Batch path:** Background processing, analytics, content generation, eval runs. Use the batch API (50% discount from OpenAI and Anthropic), larger models (since latency does not matter), and longer prompts (since TTFT is irrelevant). Batch processing can also achieve higher throughput by packing requests more efficiently.

The [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) and [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) are covered in detail in [Cost Engineering](21-cost-engineering.md). From a latency perspective, the key insight is that separating batch work from real-time work reduces contention for your API rate limits, which in turn reduces queuing delays for your latency-sensitive requests.

### Edge Deployment Considerations

Geographic distance to the LLM provider's data center adds latency. A round-trip from Singapore to a US-East data center adds ~250ms of network latency alone, before any processing begins. For global applications, this is a significant portion of the total response time.

**Provider region selection.** Most providers offer multiple regions. Deploy your application backend in the same region as (or as close as possible to) the provider's inference endpoints. For OpenAI, requests are processed in US data centers. For Anthropic, US and EU regions are available. For Google, Vertex AI offers multiple regions globally.

**Edge caching.** Deploy your response cache at the edge (close to users) so that cache hits have minimal network latency. A CDN like Cloudflare or AWS CloudFront can cache static AI-generated content (pre-generated FAQ answers, product descriptions, recommendation explanations) at edge locations worldwide. This only works for content that does not change per-request, but for the content it covers, latency drops to single-digit milliseconds.

**Edge inference.** For the smallest models, running inference at the edge (on the user's device or at edge compute locations) eliminates network latency entirely. WebLLM and similar projects enable in-browser inference for models up to ~3B parameters. This is currently practical only for simple tasks (classification, short-text generation) but the feasible model size is increasing.

### Connection Pooling And Keep-Alive

LLM API calls over HTTPS require a TLS handshake that adds 50-150ms per new connection. If your application opens a new connection for each request, this overhead adds up quickly under load.

**Connection pooling** reuses established connections across requests, amortizing the handshake cost. Both the OpenAI and Anthropic Python SDKs use connection pooling by default via `httpx`, so if you are using the official SDKs, you are already benefiting from this. However, if you are making raw HTTP calls or using a custom client, ensure you are reusing connections.

The key configuration to check: keep the client object alive across requests rather than instantiating a new client per request. This is a common mistake in serverless environments where each function invocation creates a new client.

```python
# WRONG: new client per request, no connection reuse
def handle_request(query: str):
    client = OpenAI()  # new connection each time
    return client.chat.completions.create(...)

# RIGHT: shared client, connection pooling
client = OpenAI()  # created once, reused

def handle_request(query: str):
    return client.chat.completions.create(...)
```

In serverless environments (AWS Lambda, Google Cloud Functions), connections are reused within the same warm instance but not across cold starts. Minimize cold start impact by keeping the deployment package small and using provisioned concurrency for latency-sensitive functions.

## Design Notes

**Layer your caching strategy.** The most effective systems use multiple cache layers, each catching different types of redundancy. From innermost to outermost: provider-level prompt caching (reduces TTFT for repeated prefixes), application-level exact-match response caching (eliminates API calls for identical requests), semantic caching (eliminates API calls for similar requests), and edge/CDN caching (eliminates network round-trips for static content). Each layer has different hit rates, different latency characteristics, and different failure modes. Monitor all of them.

**Measure TTFT and total latency separately.** These are different metrics that matter to different stakeholders. TTFT determines how quickly the user perceives a response starting -- it drives user satisfaction in interactive applications. Total latency (time to complete response) determines throughput and affects downstream pipelines that consume the full response. A system with excellent TTFT but high total latency (common with large models using streaming) feels responsive but still takes a long time to complete. Optimize both, but prioritize TTFT for user-facing interactions.

**Cache invalidation is the hard part.** The classic computer science problem applies here in full force. Stale cached responses are worse than no cache because users trust the system's answers. Implement TTLs appropriate to your data freshness requirements, and build explicit invalidation paths for cases where you know the answer has changed (document updates, policy changes, data refreshes). Log cache hit ages so you can audit staleness.

**Warm caches proactively.** If you know which queries are common (from logs or analytics), pre-populate your semantic and response caches before users hit them. Run a nightly job that processes the top 1,000 queries from yesterday against the current system, storing the results in the cache. This converts cold-start latency into background compute.

## Measuring And Monitoring Latency

You cannot optimize what you do not measure. Latency instrumentation should be built into your LLM pipeline from the start, not added after users complain.

### Key Metrics

**Time-to-first-token (TTFT):** The elapsed time from sending the API request to receiving the first response token. This is the metric users feel. Measure it at the client side (including network latency), not just at the server side.

**Total response time:** The elapsed time from sending the request to receiving the last token. For streaming responses, this is TTFT plus the generation time. For non-streaming responses, this equals TTFT (since the first and last token arrive together).

**Tokens per second (TPS):** The rate at which the model generates output tokens after the first token. This determines how fast text appears during streaming and how long total generation takes.

**Cache hit rate:** The percentage of requests served from cache at each layer (prompt cache, response cache, semantic cache). Low hit rates indicate that your caching strategy is not matching your query distribution.

**Pipeline stage latency:** Break down total latency into its components: embedding generation, retrieval, re-ranking, prompt assembly, model inference, output validation, and response delivery. This identifies bottlenecks. Use [OpenTelemetry](https://opentelemetry.io/) spans to capture each stage.

### Percentile Monitoring

Always monitor latency as a distribution, not an average. Report P50 (median), P95, and P99. Averages hide bimodal distributions and tail latency problems. A service with 200ms average latency might have P99 at 5 seconds due to cache misses, cold starts, or provider-side queuing.

Set alerting thresholds on P95 and P99, not on averages. Alert when P95 exceeds your user experience budget (typically 2-3 seconds for interactive applications) or when any percentile increases by more than 50% compared to the trailing 24-hour baseline.

```python
import time
from dataclasses import dataclass, field

@dataclass
class LatencyTracker:
    """Track latency metrics for LLM API calls."""
    ttft_samples: list[float] = field(default_factory=list)
    total_samples: list[float] = field(default_factory=list)

    def record_streaming_call(self, stream) -> str:
        start = time.perf_counter()
        first_token_time = None
        chunks = []

        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
                self.ttft_samples.append(first_token_time - start)
            delta = chunk.choices[0].delta
            if delta.content:
                chunks.append(delta.content)

        self.total_samples.append(time.perf_counter() - start)
        return "".join(chunks)

    def percentile(self, samples: list[float], p: float) -> float:
        if not samples:
            return 0.0
        sorted_samples = sorted(samples)
        idx = int(len(sorted_samples) * p / 100)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def report(self) -> dict:
        return {
            "ttft_p50": self.percentile(self.ttft_samples, 50),
            "ttft_p95": self.percentile(self.ttft_samples, 95),
            "ttft_p99": self.percentile(self.ttft_samples, 99),
            "total_p50": self.percentile(self.total_samples, 50),
            "total_p95": self.percentile(self.total_samples, 95),
            "total_p99": self.percentile(self.total_samples, 99),
        }
```

## Pitfalls

**Semantic cache poisoning.** If a bad response gets cached, it will be served to every subsequent query that is semantically similar. This can happen when the model hallucinates, when context changes make a previously correct answer wrong, or when an adversarial query produces a harmful response that then gets served to legitimate queries. Mitigate by validating responses before caching, setting reasonable TTLs, and monitoring cache hit quality through sampling.

**Proxy buffering breaking streams.** The most common cause of "streaming not working" is an intermediate proxy buffering the SSE response. Nginx, Apache, Cloudflare, AWS ALB, and many other infrastructure components buffer responses by default. Each one needs explicit configuration to pass through SSE streams unbuffered. Test streaming end-to-end through your full infrastructure stack, not just locally.

**Over-caching non-deterministic responses.** If you cache responses from calls with temperature > 0, you are converting a non-deterministic system into a deterministic one for cached queries. This might be fine (and is often desirable for consistency), but be aware that you are changing the system's behavior. Users who expect varied responses to the same question will get the same answer every time until the cache expires.

**Ignoring tail latency.** P50 latency might be 400ms, but if P99 is 8 seconds, 1 in 100 users has a terrible experience. Tail latency in LLM systems is often caused by provider-side queuing, long output generation, or cold cache misses on semantic lookup. Monitor P95 and P99 alongside P50, set alerts on tail latency, and investigate outliers. A common fix is to set aggressive `max_tokens` limits and implement client-side timeouts with retry logic.

**Caching without cache keys that capture context.** A response to "What's the weather?" depends on the user's location, the current date, and the data source. If your cache key only includes the query string, you will serve a stale or wrong answer. Ensure your cache key includes every variable that affects the correct response: the query, the system prompt version, relevant user context, and any retrieved documents or tool results that influenced the response.

**Premature edge deployment.** Running models at the edge sounds appealing but adds significant operational complexity: model versioning across edge locations, cold starts, limited compute resources, and debugging difficulty. Start with centralized deployment and optimize network latency through region selection and caching. Move to edge inference only when you have measured the latency budget and confirmed that network latency is the binding constraint.

## Checklist
- Are you streaming responses for all user-facing interactions?
- Is your prompt structured with static content first and variable content last to maximize cache hits?
- Do you measure TTFT and total latency separately, with percentile distributions (P50/P95/P99)?
- Have you tested streaming end-to-end through your full infrastructure, including reverse proxies and load balancers?
- Are tool calls executed in parallel when they are independent?
- Do you have a response caching layer (exact-match or semantic) with appropriate TTLs?
- Is your LLM client instantiated once and reused (connection pooling), not recreated per request?
- Do you have latency budgets defined for each user-facing interaction, with alerts on P95 violations?
- Have you evaluated whether a smaller, faster model meets quality requirements for latency-sensitive paths?
- Are batch and real-time workloads separated to avoid contention for rate limits?
- Do you monitor cache hit rates and cache staleness at each caching layer?

## References
- OpenAI Prompt Caching. https://platform.openai.com/docs/guides/prompt-caching
- Anthropic Prompt Caching. https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Google Context Caching. https://ai.google.dev/gemini-api/docs/caching
- OpenAI Streaming Guide. https://platform.openai.com/docs/api-reference/streaming
- Anthropic Streaming Messages. https://docs.anthropic.com/en/api/streaming
- GPTCache (semantic caching). https://github.com/zilliztech/GPTCache
- OpenTelemetry (distributed tracing). https://opentelemetry.io/
- Server-Sent Events specification. https://html.spec.whatwg.org/multipage/server-sent-events.html
- OpenAI Function Calling (parallel tool calls). https://platform.openai.com/docs/guides/function-calling

*Last audited: 2026-02-10 · [Audit methodology](23-audit-methodology.md)*

---
[Contents](README.md) | [Prev](25-data-pipelines.md) | [Next](27-ux-design-for-ai.md)
