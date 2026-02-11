# Glossary

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](13-staying-current.md) | [Next](10-reading-list.md)

This glossary defines key terms used throughout this book. Terms are listed alphabetically. Where a term is covered in depth in a specific chapter, the chapter is linked.

## A

**Agent** — An LLM-driven control loop that can plan, call tools, observe results, and iterate toward a goal. Unlike a single prompt-response interaction, an agent maintains state across multiple steps and makes decisions about what to do next. See [Agents](04-agents.md).

**Alignment** — The process of training a model to follow instructions, refuse harmful requests, and behave in ways consistent with human intent. Techniques include reinforcement learning from human feedback (RLHF) and constitutional AI. See [LLM Fundamentals](01-llm-fundamentals.md).

**ANN (Approximate Nearest Neighbor)** — A family of algorithms (HNSW, IVF, etc.) that trade a small amount of recall for large gains in search speed and memory efficiency. Used in vector databases to make similarity search practical at scale. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

**Attention** — The mechanism at the heart of the transformer architecture that allows the model to weigh the relevance of each token in the input when producing each token in the output. Self-attention operates within a single sequence; cross-attention operates between two sequences. See [LLM Fundamentals](01-llm-fundamentals.md).

## B

**BM25** — A classic keyword-based ranking algorithm used in traditional information retrieval. Often combined with vector search in hybrid retrieval strategies to handle exact-match queries that semantic search misses. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

## C

**Chain-of-Thought (CoT)** — A prompting technique that encourages the model to show intermediate reasoning steps before arriving at a final answer. Improves performance on tasks requiring multi-step logic. See [Prompting](02-prompting.md).

**Chunking** — The process of splitting documents into smaller units for embedding and retrieval. Chunk quality directly affects retrieval quality: chunks should be semantically coherent, self-contained, and annotated with metadata. See [Retrieval-Augmented Generation](03-rag.md).

**Context Window** — The maximum number of tokens (input plus output) that a model can process in a single request. Larger context windows enable longer conversations and more reference material, but increase cost and latency. Not equivalent to "memory" — models do not retain information between requests unless explicitly re-supplied. See [LLM Fundamentals](01-llm-fundamentals.md).

**Cosine Similarity** — A distance metric that measures the angle between two vectors, commonly used to compare embeddings. Values range from -1 (opposite) to 1 (identical direction). See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

## D

**Data Exfiltration** — An attack in which the model is manipulated into outputting sensitive information from its context (such as system prompts, API keys, or user data) to an unauthorized party. See [Safety, Privacy, and Security](06-safety-privacy-security.md).

**Deterministic Output** — A configuration (typically temperature=0) intended to produce the same output for the same input. In practice, full determinism is not guaranteed across all providers and hardware configurations. See [LLM Fundamentals](01-llm-fundamentals.md).

## E

**Embedding** — A fixed-length vector representation of content (text, image, audio) produced by an embedding model. Embeddings capture semantic meaning, allowing similarity to be computed mathematically. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

**Eval (Evaluation)** — A systematic test of model or system behavior, typically using a dataset of inputs with expected outputs. Evals can be automated (schema checks, exact match, model-graded) or human-reviewed. See [Evals and Testing](05-evals.md).

**Excessive Agency** — A risk category from the OWASP Top 10 for LLMs, in which an agent or tool-using model is given more permissions or capabilities than necessary, creating potential for unintended or harmful actions. See [Agents](04-agents.md).

## F

**Few-Shot Prompting** — Including a small number of example input-output pairs in the prompt to demonstrate the desired behavior. Effective for formatting, edge cases, and tasks where instructions alone are ambiguous. See [Prompting](02-prompting.md).

**Fine-Tuning** — Additional training of a pre-trained model on a task-specific or domain-specific dataset. Changes model weights (unlike prompting, which only changes input). Primarily adjusts behavior, style, and format rather than adding factual knowledge. Useful when prompting alone cannot achieve the required behavior. See [Fine-Tuning and Model Customization](17-fine-tuning.md).

**Format Drift** — A failure mode in which the model's output gradually deviates from a required format (JSON schema, specific template) over the course of a conversation or across requests. See [LLM Fundamentals](01-llm-fundamentals.md).

**Function Calling** — See Tool Calling.

## G

**Golden Set** — A curated dataset of representative inputs with verified expected outputs, used for regression testing. The gold standard for detecting quality regressions after prompt, model, or index changes. See [Evals and Testing](05-evals.md).

**Grounding** — Tying model outputs to verifiable external sources, typically through retrieval. A grounded answer is one that can be traced back to specific source material. See [Retrieval-Augmented Generation](03-rag.md).

## H

**Hallucination** — Output that is fluent and confident but factually incorrect or unsupported by any provided source material. A fundamental failure mode of generative models that stems from their training objective of producing plausible text. See [LLM Fundamentals](01-llm-fundamentals.md).

**HNSW (Hierarchical Navigable Small World)** — A graph-based approximate nearest neighbor algorithm widely used in vector databases. Offers strong recall-speed trade-offs and is the default index type in many vector stores. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

**Hybrid Search** — A retrieval strategy that combines keyword-based search (BM25) with semantic vector search and merges the results. Particularly effective for queries containing proper nouns, codes, or exact terms that semantic search alone might miss. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

## I

**Idempotent** — A property of an operation meaning it can be safely repeated without changing the result beyond the first execution. Critical for tool design in agent systems, where retries are common. See [Agents](04-agents.md).

## J

**JSON Schema** — A vocabulary for annotating and validating JSON documents. Used to define the expected structure of model outputs in structured output and tool calling scenarios. See [Structured Outputs and Tool Calling](11-structured-outputs-and-tool-calling.md).

## L

**Lost-in-the-Middle** — An observed behavior where models pay less attention to information in the middle of long contexts compared to information at the beginning or end. Affects retrieval order and prompt design. See [LLM Fundamentals](01-llm-fundamentals.md).

## M

**MCP (Model Context Protocol)** — An open standard for connecting LLM applications with external tools and data sources. Provides a standardized interface for tool discovery, invocation, and context management.

**Model-Graded Evaluation** — Using an LLM to score or judge the outputs of another LLM. Scales better than human review but requires calibration against human judgments and periodic spot checks. See [Evals and Testing](05-evals.md).

## P

**PII (Personally Identifiable Information)** — Any data that can be used to identify an individual (names, email addresses, phone numbers, etc.). Must be handled carefully in prompts, logs, and model outputs. See [Safety, Privacy, and Security](06-safety-privacy-security.md).

**Prompt Injection** — An attack in which untrusted text (user input, retrieved documents, web content) contains instructions that override or subvert the system's intended behavior. The most distinctive security risk of LLM systems. See [Safety, Privacy, and Security](06-safety-privacy-security.md).

## R

**RAG (Retrieval-Augmented Generation)** — An architecture that retrieves relevant documents from an external knowledge base and provides them as context to the model, grounding its response in specific source material. See [Retrieval-Augmented Generation](03-rag.md).

**ReAct** — A prompting and agent architecture that interleaves reasoning (chain-of-thought) with acting (tool calls), allowing the model to plan, execute, observe, and adjust. See [Agents](04-agents.md).

**Re-Ranking** — A second-stage retrieval step that uses a more expensive model to re-score and re-order the top-k results from an initial retrieval. Improves precision at the cost of additional latency. See [Retrieval-Augmented Generation](03-rag.md).

**RLHF (Reinforcement Learning from Human Feedback)** — A training technique that uses human preference judgments to fine-tune model behavior after pre-training. A key step in making models instruction-following and safe. See [LLM Fundamentals](01-llm-fundamentals.md).

## S

**Structured Output** — Model output that conforms to a predefined schema (typically JSON Schema), either enforced by the provider's API or validated after generation. See [Structured Outputs and Tool Calling](11-structured-outputs-and-tool-calling.md).

**System Prompt** — The initial instruction set provided to the model that defines its role, rules, and constraints. Distinguished from user messages in that it sets persistent behavior for the entire conversation. See [Prompting](02-prompting.md).

## T

**Temperature** — A sampling parameter that controls the randomness of model outputs. Lower values (closer to 0) produce more deterministic outputs; higher values produce more varied outputs. See [LLM Fundamentals](01-llm-fundamentals.md).

**Token** — The basic unit of text processing for LLMs. Text is split into tokens by a tokenizer before being processed by the model. Tokens do not always correspond to words — they may be subwords, individual characters, or whitespace. Pricing and context limits are measured in tokens. See [LLM Fundamentals](01-llm-fundamentals.md).

**Tool Calling (Function Calling)** — A mechanism by which the model outputs a structured request to invoke an external tool (API, function, database query) rather than generating a free-text answer. The calling code executes the tool and returns the result to the model. See [Structured Outputs and Tool Calling](11-structured-outputs-and-tool-calling.md).

**Top-p (Nucleus Sampling)** — A sampling method that considers only the smallest set of tokens whose cumulative probability exceeds a threshold p. An alternative to temperature for controlling output diversity. See [LLM Fundamentals](01-llm-fundamentals.md).

**Transformer** — The neural network architecture underlying modern LLMs, introduced in "Attention Is All You Need" (2017). Based on self-attention mechanisms rather than recurrence, enabling efficient parallel training and strong performance on sequence tasks. See [LLM Fundamentals](01-llm-fundamentals.md).

## V

**Vector Database (Vector Store)** — A database optimized for storing and querying high-dimensional vectors. Used to index embeddings and perform similarity search for RAG and other retrieval tasks. Examples include pgvector, Qdrant, Chroma, Weaviate, and Milvus. See [Embeddings and Vector Search](12-embeddings-and-vector-search.md).

## Z

**Zero-Shot** — Prompting a model to perform a task without providing any examples, relying solely on instructions and the model's pre-trained knowledge. Contrast with few-shot prompting. See [Prompting](02-prompting.md).

---
[Contents](README.md) | [Prev](13-staying-current.md) | [Next](10-reading-list.md)
