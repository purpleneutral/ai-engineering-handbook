# LLM Fundamentals

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](15-installation-and-local-setup.md) | [Next](02-prompting.md)

## Summary

Large language models are probabilistic text generators -- and increasingly multimodal systems that process images, audio, and code alongside natural language -- that can follow instructions with remarkable fluency. They are not, however, inherently truthful, consistent, or secure. Understanding why requires looking under the hood at how these models are built, what they optimize for, and where those design choices create predictable failure modes. This chapter covers the foundational concepts every engineer needs before building on top of LLMs: the transformer architecture, tokenization, the training pipeline, failure modes, and the reliability tactics that make production systems work.

## See Also
- [Prompting](02-prompting.md)
- [Structured Outputs And Tool Calling](11-structured-outputs-and-tool-calling.md)
- [Safety, Privacy, And Security](06-safety-privacy-security.md)

## The Transformer Architecture

Modern LLMs are built on the [transformer architecture](https://arxiv.org/abs/1706.03762), introduced in the 2017 paper ["Attention Is All You Need."](https://arxiv.org/abs/1706.03762) Before transformers, sequence models like RNNs and LSTMs processed text one token at a time, which made them slow to train and prone to forgetting information from earlier in a sequence. Transformers solved this with a mechanism called self-attention, which allows the model to look at every token in the input simultaneously and learn which tokens are relevant to each other.

At a high level, self-attention works like this: for each token in a sequence, the model computes three vectors -- a query, a key, and a value. The query represents "what am I looking for?", the key represents "what do I contain?", and the value represents "what information do I provide?" The model computes attention scores by comparing each query against all keys, then uses those scores to create a weighted combination of values. This lets the word "it" in a sentence attend strongly to the noun it refers to, even if that noun appeared many sentences earlier.

Transformers stack many layers of this attention mechanism (modern large models typically use 60-120+ layers, depending on architecture), with feed-forward networks between them. Each layer refines the model's internal representation. The result is a system that can capture long-range dependencies, handle complex syntax, and generalize across tasks -- but that fundamentally operates by predicting the next token given all previous tokens. This next-token prediction objective is the single most important thing to understand about LLMs: the model is always asking "what token is most likely to come next?" and that is not the same question as "what is true?"

Multi-head attention extends this further by running several attention operations in parallel, each with different learned projections. One head might learn to attend to syntactic relationships, another to semantic similarity, another to positional proximity. The outputs are concatenated and projected back, giving the model a rich, multi-faceted view of the input at each layer.

## Tokenization

LLMs do not operate on characters or words -- they operate on tokens, which are subword units produced by a tokenizer. Most modern models use a variant of [Byte Pair Encoding (BPE)](https://arxiv.org/abs/1508.07909), which works by starting with individual bytes or characters and iteratively merging the most frequent pairs until a target vocabulary size is reached (typically 32,000 to 100,000 tokens).

Tokenization matters for several practical reasons. First, cost: API pricing is per token, so understanding how your input maps to tokens directly affects your bill. A common English word like "running" might be a single token, while a rare technical term like "deserialization" might be split into three or four tokens. Code, non-English languages, and structured data like JSON tend to be less token-efficient than plain English prose.

Second, the context window is measured in tokens, not words. When a model advertises a 128K-token context window, the actual number of words that fits depends on the text. English prose averages roughly 1.3 tokens per word for modern tokenizers (older tokenizers can be higher), but code or structured data can be 2-3x more token-dense.

Third, tokenization affects model behavior at the edges. Models can struggle with character-level tasks (like counting letters in a word or reversing a string) because they literally do not see individual characters -- they see tokens. A word might be split differently depending on capitalization, surrounding punctuation, or even its position in the text. This is why asking a model "how many r's are in strawberry?" can produce wrong answers: the model may see the token `straw` + `berry` and never process individual letters.

## The Training Pipeline

Understanding how LLMs are trained explains both their capabilities and their limitations. The pipeline has three major stages, each adding a different kind of capability.

### Pretraining

In pretraining, the model learns language by predicting the next token on a massive corpus of text -- typically trillions of tokens scraped from the web, books, code repositories, and other sources. This stage is enormously expensive (millions of dollars in compute) and produces a base model that has broad knowledge of language, facts, reasoning patterns, and code, but that behaves like an autocomplete engine rather than an assistant. Empirical [scaling laws](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020) established that model performance improves predictably with model size, dataset size, and compute budget, directly informing the design of subsequent large models. The [GPT-3 paper](https://arxiv.org/abs/2005.14165) ("Language Models are Few-Shot Learners") demonstrated that scaling pretraining produces emergent few-shot capabilities. A pretrained model will happily continue any text you give it, including toxic, incorrect, or nonsensical text, because it learned to predict what comes next in its training data, not what is correct or helpful.

The pretraining corpus determines the model's knowledge cutoff. Facts that were not in the training data, or that were rare, will be unreliable. This is not a bug that can be fixed with more training -- it is a fundamental property of the approach.

### Supervised Fine-Tuning (SFT)

After pretraining, the model is fine-tuned on curated datasets of high-quality instruction-response pairs. This teaches the model to behave like an assistant: to answer questions, follow formatting instructions, decline harmful requests, and produce structured output. Fine-tuning is much cheaper than pretraining (thousands of examples rather than trillions of tokens) and can be done for specific domains or tasks.

Fine-tuning changes the model's default behavior but does not add new knowledge reliably. If you fine-tune a model on customer support transcripts, it will learn the style and patterns of customer support, but it will not reliably memorize specific product details from those transcripts. For factual grounding, [retrieval-augmented generation (RAG)](03-rag.md) is almost always more reliable than fine-tuning.

### RLHF And Alignment

[Reinforcement Learning from Human Feedback (RLHF)](https://arxiv.org/abs/2203.02155) and related techniques (DPO, RLAIF) further shape the model's behavior by training it to prefer outputs that human raters judge as helpful, harmless, and honest. This stage is what makes modern chat models feel "aligned" -- they refuse harmful requests, admit uncertainty (sometimes), and try to be genuinely helpful rather than just plausible.

However, alignment is shallow in an important sense: it adjusts the model's output distribution but does not give it genuine understanding of truth, ethics, or user intent. A sufficiently adversarial prompt can often bypass alignment guardrails (this is prompt injection). And the model may learn to sound confident and helpful even when it should express uncertainty, because human raters tended to prefer confident-sounding responses during training.

## Core Ideas

### Tokens And Context Window

Inputs and outputs are measured in tokens, and the context window is the maximum number of tokens the model can process in a single call. Longer context means higher cost (most APIs charge per token) and higher latency (attention is quadratic in sequence length, though various optimizations reduce this in practice). More importantly, "long context" is not the same as "good memory." Research on long-context behavior shows that models tend to pay the most attention to information at the beginning and end of the context, with material in the middle receiving less focus -- the so-called ["lost in the middle"](https://doi.org/10.1162/tacl_a_00638) phenomenon. For this reason, retrieval and summarization strategies remain essential even when the context window is technically large enough to hold all your data.

### Non-Determinism

LLMs generate text through sampling: at each step, the model produces a probability distribution over possible next tokens, and one is selected. The [temperature](https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature) parameter controls how "peaked" or "flat" this distribution is. At temperature 0, the model always picks the highest-probability token (greedy decoding), which is as deterministic as the system allows. At higher temperatures, lower-probability tokens have a better chance of being selected, producing more varied and creative output. The top-p (nucleus sampling) parameter offers a complementary control by limiting sampling to the smallest set of tokens whose cumulative probability exceeds a threshold.

Even with temperature set to 0, perfect reproducibility is not guaranteed across all providers. Implementation details like floating-point arithmetic order, batching strategies, and hardware differences can produce slightly different outputs for identical inputs. If your application requires bitwise-identical outputs, you need to test this explicitly with your specific provider and model version.

### Instruction Following Is Fragile

Despite the sophistication of the training pipeline, instruction following remains brittle. Conflicting instructions (e.g., "be concise" and "explain thoroughly") force the model to choose, and the choice may not be what you expect. Ambiguous format requirements ("return it as a table") leave room for interpretation that varies between calls. Long or noisy context can dilute the model's attention to your instructions, causing it to latch onto patterns in the data rather than your explicit rules. This fragility is not a temporary limitation that will be fixed in the next model version -- it is an inherent consequence of probabilistic text generation, and robust systems must account for it.

### Tool Use Changes The Game

One of the most important architectural shifts in LLM applications is the move from asking the model to do everything to giving it tools. Instead of asking the model to perform arithmetic (which it does unreliably), you give it a calculator tool. Instead of asking it to look up current data (which it cannot do from training alone), you give it a search tool. Instead of asking it to query a database (which would require hallucinating SQL), you give it a structured query tool with a defined interface.

Tool use transforms the model's role from "oracle that must know everything" to "reasoning engine that coordinates specialists." The model decides which tool to call and what arguments to pass; your code executes the tool and returns results. This separation is powerful because it plays to the model's strengths (language understanding, planning, intent recognition) while offloading its weaknesses (factual recall, computation, real-time data) to reliable code. Treat tool outputs as data flowing back into the conversation, not as "model creativity" -- the model did not generate those results, your tools did.

## Common Failure Modes

Understanding failure modes is essential because they are not random -- they are predictable consequences of how LLMs work. Each failure mode maps back to a specific aspect of the architecture or training.

### Hallucination

Hallucination occurs when the model generates confident, fluent statements that are factually wrong. This happens because the model's training objective is to produce plausible next tokens, not truthful ones. If the training data contained many plausible-sounding statements about a topic, the model learned to reproduce that pattern regardless of accuracy. Hallucination is most dangerous when the model's output is fluent and internally consistent -- it reads like a correct answer, making it hard for humans to catch without independent verification. Hallucination rates increase when the model is asked about rare topics, recent events, or specific numerical details, and decrease when the model is given relevant context to ground its responses.

### Format Drift

Format drift occurs when the model's output gradually deviates from a required schema or structure. You might ask for a JSON object with specific fields and get valid JSON back 95% of the time, but occasionally the model adds extra commentary outside the JSON block, renames a field, or wraps the output in markdown code fences you did not request. Format drift is particularly insidious in pipelines where the output feeds into downstream code, because it can cause silent failures that only surface later. It tends to worsen with longer outputs, more complex schemas, and when the model is simultaneously trying to follow many instructions.

### Context Distraction

Context distraction happens when irrelevant or misleading information in the context window dominates the model's output, overriding your instructions. The ["lost in the middle"](https://doi.org/10.1162/tacl_a_00638) variant is one form of this: when relevant information is buried in the middle of a long context, the model may ignore it in favor of information at the beginning or end. Another form occurs when the context contains text that strongly resembles instructions (even if it is data), and the model starts following those "instructions" instead of yours. This failure mode is closely related to prompt injection and is a key reason why separating instructions from data is a critical design pattern.

### Prompt Injection

[Prompt injection](06-safety-privacy-security.md) occurs when untrusted text in the model's input contains instructions that override or subvert your intended behavior. This can be malicious (an attacker embedding "ignore previous instructions and..." in a document the model processes) or accidental (a user's input happens to look like a system instruction). Prompt injection is fundamentally difficult to solve because the model processes instructions and data in the same "channel" -- it has no reliable built-in mechanism to distinguish between text you trust and text you do not. Mitigation requires defense in depth: input sanitization, delimiter-based separation, output validation, and limiting the model's ability to take dangerous actions.

## Reliability Tactics

Building reliable systems on top of LLMs requires acknowledging that the model is a probabilistic component, not a deterministic function. The following tactics, used in combination, bring the error rate down to manageable levels.

### Constrain Outputs

The most effective single tactic for reliability is constraining the model's output space. Instead of allowing free-form text, require the model to produce output conforming to a JSON Schema. Keep outputs small and focused -- a model asked to produce a 10-field JSON object is more reliable than one asked to produce a 100-field object. Use enums instead of free-text fields wherever possible. Some APIs support "strict mode" or "structured outputs" that constrain the model's token generation to only produce valid output for a given schema, which eliminates format drift entirely for supported schemas. Even when using strict mode, validate the output in your code -- defense in depth matters.

### Separate Instruction From Data

Treat the boundary between trusted instructions and untrusted data as a security boundary. Use clear delimiters (XML-style tags like `<reference>...</reference>` work well) to mark untrusted text, and include an explicit instruction telling the model that content within those delimiters is reference data only and must not be interpreted as instructions. Place your rules and instructions before the data in the prompt, where the model's attention is strongest. This does not make prompt injection impossible, but it significantly raises the bar for both accidental and intentional injection.

### Verify And Ground

Never trust model output without verification. For factual claims, use retrieval-augmented generation to provide the model with source documents and require citations. For calculations, use tool calls to a calculator or code interpreter rather than asking the model to compute. For structured output, validate against a schema before passing results downstream. Build post-processing checks into your pipeline: allowlists for sensitive fields, regex validation for formats like email addresses or URLs, and unit tests that run against model output in CI. The goal is to make the system correct by construction, not by hoping the model gets it right.

### Design For Retries

Because LLM output is probabilistic, validation failures are expected, not exceptional. Design your pipeline with a retry loop: when validation fails, send the model a new request that includes the original prompt, the invalid output, the specific validation error, and a concise instruction to fix just the error. Keep retries bounded (two or three attempts is typical) and have a fallback path for when retries are exhausted -- returning a safe default, escalating to a human, or returning an explicit error is always better than silently proceeding with invalid data.

## Reasoning Models

As of 2026-02-11, a distinct category of LLMs has emerged that are trained to perform explicit, extended chain-of-thought reasoning at inference time before producing a final answer. These are commonly called "reasoning models" and include [OpenAI's o1 and o3 series](https://platform.openai.com/docs/models), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), and Google's [Gemini 2.5](https://ai.google.dev/gemini-api/docs/models) "thinking" variants.

### How They Differ From Standard Models

Standard LLMs generate responses token by token in a single forward pass, with any reasoning being implicit in the generation process. Reasoning models add a separate "thinking" phase: before producing the user-visible answer, the model generates internal reasoning tokens (sometimes called "thinking tokens") that break the problem into steps, check intermediate results, backtrack from errors, and refine the approach. This extended thinking process is a form of inference-time compute scaling — the model spends more computation per request in exchange for better accuracy on difficult tasks.

The training methods also differ. While standard chat models are trained with supervised fine-tuning and RLHF on instruction-response pairs, reasoning models are additionally trained with reinforcement learning on tasks with verifiable correct answers (mathematics, code, logic puzzles). [DeepSeek-R1](https://arxiv.org/abs/2501.12948) demonstrated that reasoning behaviors like self-reflection, verification, and dynamic strategy switching can emerge from pure RL without any human-labeled reasoning trajectories. Some reasoning models (like OpenAI's o1/o3) hide their chain-of-thought from the user, while others (like DeepSeek-R1) expose it.

### When To Use Reasoning Models

Reasoning models excel at tasks that require multi-step logic, mathematical computation, complex code generation, scientific reasoning, and problems where getting the right answer matters more than getting a fast answer. They consistently outperform standard models on benchmarks like AIME (competition mathematics), Codeforces-style programming challenges, and PhD-level science questions.

However, reasoning models are not always the right choice. They use significantly more tokens per request (the thinking phase can generate thousands of tokens internally), which increases both cost and latency. For straightforward tasks — summarization, translation, simple Q&A, creative writing, format conversion — a standard model will typically produce equivalent or better results at a fraction of the cost and response time. The practical guidance is: use reasoning models when the task involves verifiable correctness and multi-step logic, and use standard models for everything else.

### Distillation And Accessibility

One of the most significant findings from [DeepSeek-R1](https://arxiv.org/abs/2501.12948) is that reasoning capabilities can be distilled from large reasoning models into much smaller ones. DeepSeek released distilled variants ranging from 1.5B to 70B parameters that retained much of the reasoning quality of the full 671B-parameter [Mixture-of-Experts (MoE)](https://arxiv.org/abs/1701.06538) model (37B parameters activated per forward pass). This pattern — training a large reasoning model and distilling it into smaller deployment models — is likely to become a standard part of the model development pipeline.

## Checklist
- Define "done": correct, safe, and parseable.
- Decide deterministic vs creative behavior (set temperature/top-p accordingly).
- Add validation and fallbacks for every model output.
- Log inputs/outputs safely (redact secrets/PII).
- Identify which failure modes are most dangerous for your use case and add specific mitigations.
- Test with adversarial inputs, not just happy-path examples.
- Version your prompts and model configurations alongside your code.

## References
- Transformers (foundation). https://arxiv.org/abs/1706.03762
- GPT-3 paper (few-shot learning). https://arxiv.org/abs/2005.14165
- InstructGPT / RLHF paper. https://arxiv.org/abs/2203.02155
- Chain-of-thought prompting (NeurIPS 2022). https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html
- Lost in the Middle (long context behavior). https://doi.org/10.1162/tacl_a_00638
- DeepSeek-R1 (reasoning via reinforcement learning). https://arxiv.org/abs/2501.12948
- Scaling Laws for Neural Language Models (Kaplan et al., 2020). https://arxiv.org/abs/2001.08361
- Sparsely-Gated Mixture-of-Experts (Shazeer et al., 2017). https://arxiv.org/abs/1701.06538

---
[Contents](README.md) | [Prev](15-installation-and-local-setup.md) | [Next](02-prompting.md)
