# Glossary

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](22-guardrails-and-moderation.md) | [Next](10-reading-list.md)

This glossary defines key terms used throughout this book. Terms are listed alphabetically. Where a term is covered in depth in a specific chapter, the chapter is linked.

## A

**Agent** — An LLM-driven control loop that can plan, call tools, observe results, and iterate toward a goal. Unlike a single prompt-response interaction, an agent maintains state across multiple steps and makes decisions about what to do next. See [Agents](04-agents.md).

**Alignment** — The process of training a model to follow instructions, refuse harmful requests, and behave in ways consistent with human intent. Techniques include reinforcement learning from human feedback ([RLHF](https://arxiv.org/abs/2203.02155)) and constitutional AI. See [LLM Fundamentals](01-llm-fundamentals.md).

**ANN (Approximate Nearest Neighbor)** — A family of algorithms ([HNSW](https://arxiv.org/abs/1603.09320), IVF, etc.) that trade a small amount of recall for large gains in search speed and memory efficiency. Used in vector databases to make similarity search practical at scale. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

**[Anthropic](https://docs.anthropic.com/)** — An AI safety and research company founded in 2021 by former OpenAI researchers, including Dario and Daniela Amodei. Anthropic develops the Claude family of large language models and is known for pioneering Constitutional AI (CAI) as an alignment technique. The company's research focus is on building reliable, interpretable, and steerable AI systems. See also Constitutional AI (CAI).

**Attention** — The mechanism at the heart of the [transformer](https://arxiv.org/abs/1706.03762) architecture that allows the model to weigh the relevance of each token in the input when producing each token in the output. Self-attention operates within a single sequence; cross-attention operates between two sequences. See [LLM Fundamentals](01-llm-fundamentals.md).

## B

**BM25** — A classic keyword-based ranking algorithm used in traditional information retrieval. Often combined with vector search in hybrid retrieval strategies to handle exact-match queries that semantic search misses. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

## C

**[Chain-of-Thought](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) (CoT)** — A prompting technique that encourages the model to show intermediate reasoning steps before arriving at a final answer. Improves performance on tasks requiring multi-step logic. See [Prompting](02-prompting.md).

**Chunking** — The process of splitting documents into smaller units for embedding and retrieval. Chunk quality directly affects retrieval quality: chunks should be semantically coherent, self-contained, and annotated with metadata. See [Retrieval-Augmented Generation](03-rag.md).

**[Constitutional AI](https://arxiv.org/abs/2212.08073) (CAI)** — An alignment technique developed by Anthropic in which the model is trained to critique and revise its own outputs according to a set of written principles (the "constitution"), rather than relying solely on human feedback for every judgment. The process has two phases: a supervised phase where the model generates self-critiques and revisions, and a reinforcement learning phase where a preference model trained on AI-generated feedback replaces much of the human labeling. CAI reduces the need for human evaluators while making alignment criteria explicit and auditable. See [LLM Fundamentals](01-llm-fundamentals.md).

**Context Caching** — A provider-level feature that stores the computed key-value (KV) tensors from the attention layers for a prompt prefix, so that subsequent requests sharing the same prefix can skip redundant computation. As of 2026-02-11, Anthropic, OpenAI, and Google all offer variants: Anthropic provides explicit cache controls with up to 90% cost reduction on cache reads; OpenAI applies automatic caching with a 50% discount; Google offers "context caching" with configurable TTLs up to 60 minutes. Effective use requires placing static content (system prompts, few-shot examples, reference documents) at the beginning of the request. See also Prompt Caching.

**Context Window** — The maximum number of tokens (input plus output) that a model can process in a single request. Larger context windows enable longer conversations and more reference material, but increase cost and latency. Not equivalent to "memory" — models do not retain information between requests unless explicitly re-supplied. See [LLM Fundamentals](01-llm-fundamentals.md).

**Cosine Similarity** — A distance metric that measures the angle between two vectors, commonly used to compare embeddings. Values range from -1 (opposite) to 1 (identical direction). See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

## D

**Data Exfiltration** — An attack in which the model is manipulated into outputting sensitive information from its context (such as system prompts, API keys, or user data) to an unauthorized party. See [Safety, Privacy, and Security](06-safety-privacy-security.md).

**Deterministic Output** — A configuration (typically temperature=0) intended to produce the same output for the same input. In practice, full determinism is not guaranteed across all providers and hardware configurations. See [LLM Fundamentals](01-llm-fundamentals.md).

**Distillation** — A training technique in which a smaller "student" model is trained to reproduce the outputs (or intermediate representations) of a larger "teacher" model. Distillation transfers much of the teacher's capability into a model that is cheaper to serve and faster at inference, at the cost of some quality loss. Widely used to create compact deployment models and to transfer reasoning capabilities from frontier models into smaller ones (as demonstrated by [DeepSeek-R1](https://arxiv.org/abs/2501.12948)'s distilled variants). See [Fine-Tuning and Model Customization](17-fine-tuning.md).

## E

**Embedding** — A fixed-length vector representation of content (text, image, audio) produced by an embedding model. Embeddings capture semantic meaning, allowing similarity to be computed mathematically. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

**Eval (Evaluation)** — A systematic test of model or system behavior, typically using a dataset of inputs with expected outputs. Evals can be automated (schema checks, exact match, model-graded) or human-reviewed. See [Evals and Testing](05-evals.md).

**Excessive Agency** — A risk category from the [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/), in which an agent or tool-using model is given more permissions or capabilities than necessary, creating potential for unintended or harmful actions. See [Agents](04-agents.md).

## F

**Few-Shot Prompting** — Including a small number of example input-output pairs in the prompt to demonstrate the desired behavior. Effective for formatting, edge cases, and tasks where instructions alone are ambiguous. See [Prompting](02-prompting.md).

**Fine-Tuning** — Additional training of a pre-trained model on a task-specific or domain-specific dataset. Changes model weights (unlike prompting, which only changes input). Primarily adjusts behavior, style, and format rather than adding factual knowledge. Useful when prompting alone cannot achieve the required behavior. See [Fine-Tuning and Model Customization](17-fine-tuning.md).

**Format Drift** — A failure mode in which the model's output gradually deviates from a required format (JSON schema, specific template) over the course of a conversation or across requests. See [LLM Fundamentals](01-llm-fundamentals.md).

**Function Calling** — See Tool Calling.

## G

**Golden Set** — A curated dataset of representative inputs with verified expected outputs, used for regression testing. The gold standard for detecting quality regressions after prompt, model, or index changes. See [Evals and Testing](05-evals.md).

**Guardrails** — Input validation and output validation layers wrapped around an LLM call to enforce safety, format, and policy constraints. Input guardrails filter or reject dangerous, off-topic, or malformed requests before they reach the model. Output guardrails check the model's response against schemas, allowlists, toxicity classifiers, or business rules before it reaches the user. Guardrails are a core component of defense-in-depth for production LLM systems. See [Safety, Privacy, and Security](06-safety-privacy-security.md).

**Grounding** — Tying model outputs to verifiable external sources, typically through retrieval. A grounded answer is one that can be traced back to specific source material. See [Retrieval-Augmented Generation](03-rag.md).

## H

**Hallucination** — Output that is fluent and confident but factually incorrect or unsupported by any provided source material. A fundamental failure mode of generative models that stems from their training objective of producing plausible text. See [LLM Fundamentals](01-llm-fundamentals.md).

**[HNSW](https://arxiv.org/abs/1603.09320) (Hierarchical Navigable Small World)** — A graph-based approximate nearest neighbor algorithm widely used in vector databases. Offers strong recall-speed trade-offs and is the default index type in many vector stores. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

**Hybrid Search** — A retrieval strategy that combines keyword-based search (BM25) with semantic vector search and merges the results. Particularly effective for queries containing proper nouns, codes, or exact terms that semantic search alone might miss. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

## I

**Idempotent** — A property of an operation meaning it can be safely repeated without changing the result beyond the first execution. Critical for tool design in agent systems, where retries are common. See [Agents](04-agents.md).

**Inference** — The process of running a trained model to produce outputs from inputs. In the context of LLMs, inference means generating text (or structured output) in response to a prompt. Inference cost, latency, and throughput are primary operational concerns for production systems. See [LLM Fundamentals](01-llm-fundamentals.md).

## J

**[JSON Schema](https://json-schema.org/)** — A vocabulary for annotating and validating JSON documents. Used to define the expected structure of model outputs in structured output and tool calling scenarios. See [Structured Outputs and Tool Calling](11-structured-outputs-and-tool-calling.md).

## L

**Latency** — The time between sending a request and receiving the complete response. In LLM systems, latency has two components: time-to-first-token (TTFT), which determines how quickly the user sees the response begin, and total generation time, which depends on output length. Latency is affected by model size, input length, output length, provider load, and whether streaming is used. See [Ops: Shipping and Running LLM Systems](08-ops.md).

**[LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation)** — A parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects small, trainable low-rank decomposition matrices into each transformer layer. LoRA can reduce the number of trainable parameters by 10,000x compared to full fine-tuning while achieving comparable quality, making it practical to fine-tune large models on limited hardware. Introduced by Hu et al. (2021). See [Fine-Tuning and Model Customization](17-fine-tuning.md).

**[Lost-in-the-Middle](https://doi.org/10.1162/tacl_a_00638)** — An observed behavior where models pay less attention to information in the middle of long contexts compared to information at the beginning or end. Affects retrieval order and prompt design. See [LLM Fundamentals](01-llm-fundamentals.md).

## M

**[MCP](https://modelcontextprotocol.io/specification/2025-11-25) (Model Context Protocol)** — An open protocol, originally introduced by Anthropic in November 2024, that standardizes how LLM applications connect to external tools and data sources. Inspired by the [Language Server Protocol](https://microsoft.github.io/language-server-protocol/), MCP defines a client-server architecture in which MCP servers expose tools, resources, and context to MCP clients (AI applications) through a common interface. As of 2026-02-11, MCP has been adopted by OpenAI, Google, and Microsoft, and was donated to the Agentic AI Foundation under the Linux Foundation in December 2025. The specification covers tool discovery, invocation, context management, and (since the November 2025 revision) asynchronous operations and server identity. See [Agents](04-agents.md) and [Structured Outputs and Tool Calling](11-structured-outputs-and-tool-calling.md).

**[Mixture of Experts](https://arxiv.org/abs/1701.06538) (MoE)** — A model architecture that replaces the single feed-forward network in each transformer layer with multiple specialized sub-networks ("experts") and a learned routing function ("gating network") that selects a sparse subset of experts for each input token. MoE allows dramatically scaling total model parameters without a proportional increase in compute cost, because only a fraction of the experts are active per token. Notable MoE-based models include Mixtral 8x7B and DeepSeek-V3. See [LLM Fundamentals](01-llm-fundamentals.md).

**[Model-Graded Evaluation](https://arxiv.org/abs/2306.05685)** — Using an LLM to score or judge the outputs of another LLM. Scales better than human review but requires calibration against human judgments and periodic spot checks. See [Evals and Testing](05-evals.md).

**Multimodal** — Describes models or systems that can process and generate multiple types of input and output beyond text, such as images, audio, video, and code. Modern frontier models (GPT-4, [Claude](https://docs.anthropic.com/), [Gemini](https://ai.google.dev/docs)) accept images alongside text; some also handle audio and video natively. Multimodal capabilities expand the range of tasks an LLM system can address but introduce additional considerations for tokenization, cost, and evaluation. See [Multimodal AI](20-multimodal.md).

## O

**[OpenAI](https://platform.openai.com/docs/)** — An AI research and deployment company founded in 2015 by Sam Altman, Greg Brockman, Ilya Sutskever, and others. OpenAI develops the GPT family of large language models, the ChatGPT conversational interface, the DALL-E image generation models, and the o1/o3 series of reasoning models. Its release of ChatGPT in November 2022 catalyzed widespread adoption of generative AI. OpenAI also provides one of the most widely used LLM APIs, whose documentation is referenced throughout this book.

## P

**PII (Personally Identifiable Information)** — Any data that can be used to identify an individual (names, email addresses, phone numbers, etc.). Must be handled carefully in prompts, logs, and model outputs. See [Safety, Privacy, and Security](06-safety-privacy-security.md).

**Pre-training** — The initial, large-scale training phase of an LLM in which the model learns language by predicting the next token on a massive corpus (typically trillions of tokens from the web, books, and code). Pre-training is enormously expensive and produces a base model with broad knowledge but no instruction-following behavior. Subsequent stages (supervised fine-tuning, RLHF) shape the base model into a usable assistant. See [LLM Fundamentals](01-llm-fundamentals.md).

**Prompt Caching** — The practice of caching repeated prompt prefixes so that subsequent API calls sharing the same prefix avoid redundant computation, reducing both cost and latency. Most major LLM providers now offer prompt caching in some form. Effective prompt design for caching places stable content (system prompts, instructions, reference material) at the beginning and variable content (user messages) at the end. See also Context Caching.

**Prompt Injection** — An attack in which untrusted text (user input, retrieved documents, web content) contains instructions that override or subvert the system's intended behavior. The most distinctive security risk of LLM systems. See [Safety, Privacy, and Security](06-safety-privacy-security.md).

## Q

**[QLoRA](https://arxiv.org/abs/2305.14314) (Quantized LoRA)** — An extension of LoRA that applies low-rank adaptation to a 4-bit quantized base model, dramatically reducing the memory required for fine-tuning. QLoRA introduced 4-bit NormalFloat (NF4) quantization and double quantization, enabling fine-tuning of a 65B-parameter model on a single 48GB GPU while preserving full 16-bit fine-tuning quality. Introduced by Dettmers et al. (2023). See [Fine-Tuning and Model Customization](17-fine-tuning.md).

**Quantization** — The process of reducing the numerical precision of model weights (e.g., from 16-bit floating point to 8-bit or 4-bit integers) to decrease model size and memory requirements, enabling deployment on smaller hardware. Quantization introduces a small quality trade-off but can reduce model size by 2--4x with minimal impact on output quality for many tasks. Common formats include GPTQ, AWQ, and GGUF. See [Fine-Tuning and Model Customization](17-fine-tuning.md) and [Installation and Local Setup](15-installation-and-local-setup.md).

## R

**[RAG](https://arxiv.org/abs/2005.11401) (Retrieval-Augmented Generation)** — An architecture that retrieves relevant documents from an external knowledge base and provides them as context to the model, grounding its response in specific source material. See [Retrieval-Augmented Generation](03-rag.md).

**[ReAct](https://arxiv.org/abs/2210.03629)** — A prompting and agent architecture that interleaves reasoning (chain-of-thought) with acting (tool calls), allowing the model to plan, execute, observe, and adjust. See [Agents](04-agents.md).

**Reasoning Models** — A category of LLMs trained to perform explicit, extended chain-of-thought reasoning at inference time before producing a final answer. Examples include OpenAI's o1 and o3 series and [DeepSeek-R1](https://arxiv.org/abs/2501.12948). These models generate internal "thinking tokens" that break down complex problems into steps, self-check for errors, and refine their approach, achieving substantially better performance on math, coding, and scientific reasoning tasks at the cost of higher token usage and latency. See [LLM Fundamentals](01-llm-fundamentals.md).

**Re-Ranking** — A second-stage retrieval step that uses a more expensive model to re-score and re-order the top-k results from an initial retrieval. Improves precision at the cost of additional latency. See [Retrieval-Augmented Generation](03-rag.md).

**Retrieval** — The process of finding and returning relevant documents or passages from a knowledge base in response to a query. In LLM systems, retrieval typically involves converting the query into an embedding, searching a vector index for similar content, and optionally re-ranking results. Retrieval quality is the single largest determinant of RAG system quality. See [Retrieval-Augmented Generation](03-rag.md) and [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

**[RLHF](https://arxiv.org/abs/2203.02155) (Reinforcement Learning from Human Feedback)** — A training technique that uses human preference judgments to fine-tune model behavior after pre-training. A key step in making models instruction-following and safe. See [LLM Fundamentals](01-llm-fundamentals.md).

## S

**Semantic Search** — Search based on the meaning of a query rather than exact keyword matching. In practice, this means converting both queries and documents into embeddings and finding the nearest vectors by cosine similarity or another distance metric. Semantic search handles synonyms, paraphrases, and conceptual similarity that keyword search misses, but can struggle with exact terms, proper nouns, and codes. Often combined with keyword search in a hybrid retrieval strategy. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

**Structured Output** — Model output that conforms to a predefined schema (typically JSON Schema), either enforced by the provider's API or validated after generation. See [Structured Outputs and Tool Calling](11-structured-outputs-and-tool-calling.md).

**Synthetic Data** — Data generated by a model rather than collected from real-world sources, used for training, fine-tuning, or evaluation. Common applications include generating training examples for fine-tuning (distillation), creating diverse eval datasets, augmenting scarce labeled data, and producing adversarial test cases. Synthetic data is powerful but carries risks: it can amplify biases present in the generating model, and models trained on their own outputs can degrade in quality (model collapse). See [Evals and Testing](05-evals.md) and [Fine-Tuning and Model Customization](17-fine-tuning.md).

**System Prompt** — The initial instruction set provided to the model that defines its role, rules, and constraints. Distinguished from user messages in that it sets persistent behavior for the entire conversation. See [Prompting](02-prompting.md).

## T

**Temperature** — A sampling parameter that controls the randomness of model outputs. Lower values (closer to 0) produce more deterministic outputs; higher values produce more varied outputs. See [LLM Fundamentals](01-llm-fundamentals.md).

**Token** — The basic unit of text processing for LLMs. Text is split into tokens by a tokenizer before being processed by the model. Tokens do not always correspond to words — they may be subwords, individual characters, or whitespace. Pricing and context limits are measured in tokens. See [LLM Fundamentals](01-llm-fundamentals.md).

**Tokenizer** — The component that splits raw text into tokens before it is processed by a language model. Most modern LLMs use a variant of [Byte Pair Encoding](https://arxiv.org/abs/1508.07909) (BPE) that starts with individual bytes or characters and iteratively merges frequent pairs until a target vocabulary size is reached (typically 32,000 to 100,000 tokens). The tokenizer determines how text maps to the model's vocabulary and directly affects cost, context utilization, and the model's ability to handle character-level tasks. Different model families use different tokenizers, so the same text may tokenize differently across providers. See [LLM Fundamentals](01-llm-fundamentals.md).

**Tool Calling (Function Calling)** — A mechanism by which the model outputs a structured request to invoke an external tool (API, function, database query) rather than generating a free-text answer. The calling code executes the tool and returns the result to the model. See [Structured Outputs and Tool Calling](11-structured-outputs-and-tool-calling.md).

**Top-p (Nucleus Sampling)** — A sampling method that considers only the smallest set of tokens whose cumulative probability exceeds a threshold p. An alternative to temperature for controlling output diversity. See [LLM Fundamentals](01-llm-fundamentals.md).

**Transformer** — The neural network architecture underlying modern LLMs, introduced in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017). Based on self-attention mechanisms rather than recurrence, enabling efficient parallel training and strong performance on sequence tasks. See [LLM Fundamentals](01-llm-fundamentals.md).

## V

**Vector Database (Vector Store)** — A database optimized for storing and querying high-dimensional vectors. Used to index embeddings and perform similarity search for RAG and other retrieval tasks. Examples include [pgvector](https://github.com/pgvector/pgvector), [Qdrant](https://qdrant.tech/), [Chroma](https://www.trychroma.com/), [Weaviate](https://weaviate.io/), and [Milvus](https://milvus.io/). See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

## Z

**Zero-Shot** — Prompting a model to perform a task without providing any examples, relying solely on instructions and the model's pre-trained knowledge. Contrast with few-shot prompting. See [Prompting](02-prompting.md).

---
[Contents](README.md) | [Prev](22-guardrails-and-moderation.md) | [Next](10-reading-list.md)
