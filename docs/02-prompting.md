# Prompting

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](01-llm-fundamentals.md) | [Next](11-structured-outputs-and-tool-calling.md)

## Summary

Prompting is interface design: you are shaping the model's behavior through constraints, examples, and input hygiene. Just as a well-designed API guides callers toward correct usage and makes misuse difficult, a well-designed prompt guides the model toward the output you need and makes failure modes less likely. This chapter covers the anatomy of effective prompts, the key techniques (few-shot learning, chain-of-thought reasoning, system vs. user message roles), the sampling parameters that control output variability, and the pitfalls that cause prompts to fail in production.

## See Also
- [Structured Outputs And Tool Calling](11-structured-outputs-and-tool-calling.md)
- [Retrieval-Augmented Generation (RAG)](03-rag.md)
- [Safety, Privacy, And Security](06-safety-privacy-security.md)

## Message Roles: System, User, And Assistant

Modern chat-based LLMs organize their input as a sequence of messages, each with a role. Understanding these roles is fundamental to effective prompt design.

The **system message** (sometimes called the system prompt) sets the overall behavior, persona, and rules for the model. It is processed first and typically receives strong attention throughout the conversation. This is where you place non-negotiable constraints: output format requirements, safety boundaries, the model's role definition, and instructions about how to handle untrusted input. Not all APIs expose a separate system message, but when available, it is the highest-trust channel you have.

**User messages** represent the human's input -- the task, the question, or the data to process. In a production pipeline, the "user" might actually be your application code, not a human. The key distinction is that user messages carry the specific request and the data for this particular invocation, while the system message carries the stable rules that apply across all invocations.

**Assistant messages** represent the model's own prior responses. In multi-turn conversations, the model sees its previous outputs as part of the context. This is useful for few-shot prompting (you can inject synthetic assistant messages as examples) and for guiding the model's behavior through a "prefill" -- starting the assistant's response with a partial answer that the model then completes. For example, starting the assistant message with `{"result":` can steer the model toward producing JSON without extra preamble.

In production, the system message should be treated as your primary control surface. Keep it stable across calls, version it in source control, and be deliberate about what goes there versus what goes in the user message.

## Prompt Anatomy

A well-structured prompt has four components, and getting the order and separation right matters more than most people expect.

**Rules** are your non-negotiable constraints. These go first (typically in the system message) because the model pays the most attention to the beginning of the context. Rules include output format requirements, safety boundaries, and behavioral constraints. Be specific: "Output must be a JSON object matching the schema below" is a rule. "Be helpful" is not -- it is too vague to constrain behavior.

**Task** is what you want the model to do for this specific invocation, including clear success criteria. A good task description answers: what is the input, what is the expected output, and how will correctness be judged? Vague tasks produce vague outputs. "Summarize this document" is weaker than "Write a 2-3 sentence summary of the document that captures the main argument and the key evidence."

**Context** is the relevant information the model needs to complete the task. This might be a retrieved document, a database record, a user's conversation history, or any other data. Context should be clearly delimited and marked as data, not instructions. Use tags like `<reference>...</reference>` or `<context>...</context>` to create a visual and semantic boundary between your instructions and the data the model should draw from.

**Examples** are input-output pairs that demonstrate the behavior you want. Even one or two well-chosen examples can dramatically improve output quality and consistency, especially for tasks involving specific formats, edge cases, or judgment calls. Examples are the subject of the next section.

## Few-Shot Prompting

Few-shot prompting means including a small number of example input-output pairs in the prompt to demonstrate the desired behavior. This is one of the most powerful and underused techniques in production prompt engineering.

The idea is simple: instead of describing what you want in abstract rules, you show the model concrete examples. The model generalizes from the examples to handle new inputs. This works because LLMs are exceptionally good at pattern matching -- they can infer format, style, level of detail, edge case handling, and even implicit rules from just a few demonstrations.

Effective few-shot examples have several properties. They should cover the most common case and at least one important edge case. They should be realistic (not toy examples) and representative of the actual data the model will see. They should be consistent with each other -- contradictory examples confuse the model. And they should be placed after the rules and task description but before the actual input, so the model sees the pattern and then applies it.

Here is a concrete example of few-shot prompting for a classification task:

```text
SYSTEM:
You are a support ticket classifier. Classify each ticket into exactly one
category: billing, technical, account, or other. Respond with only the
category name in lowercase.

USER:
Ticket: "I was charged twice for my subscription last month"
Category: billing

Ticket: "The app crashes when I try to upload files larger than 10MB"
Category: technical

Ticket: "I need to change the email address on my account"
Category: account

Ticket: "Can you help me reset my password? I also think I was overcharged."
Category:
```

Notice how the examples demonstrate format (just the category name, lowercase), the distinction between categories, and the last example is a genuinely ambiguous case that forces the model to make a judgment call. The number of examples matters less than their quality -- three to five well-chosen examples typically outperform twenty mediocre ones.

Zero-shot prompting (no examples) works well for straightforward tasks where the model's pretraining gives it strong priors. Few-shot prompting is most valuable when the task involves a specific format the model has not seen before, subtle classification boundaries, or domain-specific judgment.

## Chain-Of-Thought Prompting

Chain-of-thought (CoT) prompting asks the model to show its reasoning before giving a final answer. This is not just a presentation preference -- it materially improves accuracy on tasks that require multi-step reasoning, arithmetic, logical deduction, or complex judgment.

The mechanism is straightforward: by generating intermediate reasoning steps as tokens, the model effectively gives itself "scratch space" to work through a problem. Each generated token becomes part of the context for the next token, so the model can build on its own reasoning rather than trying to jump directly to an answer. Research has shown that CoT prompting can improve accuracy by 10-40% on reasoning-heavy tasks compared to direct answer prompting.

There are two main approaches. **Explicit CoT** includes a directive like "Think step by step before answering" or "First, analyze the key factors, then provide your conclusion." **Few-shot CoT** includes examples where the reasoning is shown:

```text
USER:
Question: A store has 45 apples. They sell 12 in the morning and receive
a shipment of 30 in the afternoon. How many do they have at end of day?

Let me work through this step by step:
- Start: 45 apples
- After morning sales: 45 - 12 = 33 apples
- After afternoon shipment: 33 + 30 = 63 apples
Answer: 63

Question: A warehouse has 200 units. They ship 85 on Monday, receive 40
on Tuesday, and ship 60 on Wednesday. How many remain?
```

Chain-of-thought prompting has a cost: it increases output length (and therefore latency and token cost). For simple extraction or classification tasks, CoT is unnecessary overhead. Use it when the task genuinely requires reasoning, and consider whether you need the reasoning visible in the final output or just as an intermediate step that gets stripped before presenting results to the user.

An important caveat: the model's chain-of-thought is not a reliable window into its "actual reasoning." Models can produce plausible-sounding reasoning that leads to a wrong answer, or arrive at a correct answer via flawed reasoning. Treat CoT as a reliability technique, not as an explanation mechanism.

## Temperature And Sampling Parameters

The sampling parameters -- temperature, top-p, and related settings -- control the tradeoff between consistency and creativity in model output. Understanding them is essential for production use.

**Temperature** scales the logit values before the softmax function that produces the token probability distribution. At temperature 0 (or very close to 0), the model always selects the highest-probability token, producing the most deterministic output possible. At temperature 1.0, the original probability distribution is used as-is. Higher temperatures flatten the distribution, making lower-probability tokens more likely to be selected.

**Top-p (nucleus sampling)** offers a complementary control. Instead of adjusting the distribution shape, it truncates it: only tokens whose cumulative probability exceeds the threshold p are considered. A top-p of 0.9 means the model samples from the smallest set of tokens that together have at least 90% probability. This adapts naturally to situations where the model is confident (few tokens considered) versus uncertain (many tokens considered).

Practical guidance for production systems:

- **Extraction, classification, structured output:** Use temperature 0 (or the lowest your API allows) and top-p 1.0. You want the most likely output every time.
- **Summarization, rewriting, general Q&A:** Temperature 0.0-0.3 with top-p 0.9-1.0. Slight variation is acceptable, but you want consistency.
- **Creative writing, brainstorming, diverse suggestions:** Temperature 0.7-1.0 with top-p 0.9-0.95. Higher variance is the goal.
- **Never exceed temperature 1.5** in production. Very high temperatures produce increasingly incoherent output.

Most providers also expose `top-k`, `frequency_penalty`, and `presence_penalty`. These are useful for specific situations (reducing repetition, encouraging diversity) but temperature and top-p are the primary controls you should tune first.

An important practical note: even at temperature 0, do not assume bitwise-identical outputs across calls. Different hardware, batching, and implementation details can produce slight variations. If your pipeline requires exact reproducibility, you need explicit testing and potentially caching.

## Practical Guidelines

**Be explicit about output format.** "Return JSON matching this schema" beats "respond in JSON." Providing the actual schema (or a concrete example of the expected output) is better still. The more precisely you specify the format, the less room there is for format drift. If you need machine-parseable output, see the [Structured Outputs](11-structured-outputs-and-tool-calling.md) chapter.

**Keep instructions short and stable.** A prompt that is clear in 200 tokens will outperform one that is verbose in 2000 tokens, because longer prompts dilute the model's attention. Move long reference material into RAG-retrieved context or tool-provided data rather than embedding it in the system prompt. The system prompt should contain rules and structure; data belongs in the user message.

**Use delimiters for untrusted text.** Any time your prompt includes text that did not come from your own code -- user input, retrieved documents, scraped content -- wrap it in clear delimiters like `<reference>...</reference>` and include an explicit instruction that content within those tags is data only and must not be treated as instructions. This is your primary defense against prompt injection.

**Prefer "do" over "don't."** State what the model must produce rather than listing dozens of prohibitions. "Respond only with a JSON object" is more effective than "Do not include any explanation, do not add markdown formatting, do not include a preamble, do not ..." The model is better at following positive instructions than navigating a minefield of negations.

**Iterate with real data.** The biggest mistake in prompt engineering is testing with a handful of clean examples and calling it done. Test your prompt against the messiest, most adversarial, most edge-case-heavy inputs you can find. If you do not have real data yet, generate synthetic worst cases: inputs with special characters, conflicting information, embedded instructions, extremely long or extremely short content.

## Prompt Skeleton

```text
SYSTEM:
You are an assistant that produces strictly valid JSON.
Rules:
1) Output must match the schema below.
2) Only use information from <reference>. If missing, output nulls.
3) Do not include any text outside the JSON object.

Schema:
{ ... }

Example:
Input: "Alice, age 30, engineer at Acme Corp"
Output: {"name": "Alice", "age": 30, "title": "engineer", "company": "Acme Corp"}

USER:
Task: Extract fields from the reference.
<reference>
...
</reference>
```

This skeleton embodies the key principles: rules come first in the system message, the schema is provided explicitly, a concrete example demonstrates the expected behavior, untrusted text is delimited in the user message, and the task is stated clearly. Adapt the structure to your use case, but preserve the separation between rules, examples, and data.

A runnable version of this pattern using the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that responds in JSON."},
        {"role": "user", "content": "List three capitals in Europe."},
    ],
    temperature=0,
)

print(response.choices[0].message.content)
```

## Pitfalls

**Prompt injection via untrusted text.** This is the most serious prompt failure mode. Never place untrusted text in the same "instruction channel" as your rules. If a retrieved document or user input contains text like "Ignore all previous instructions and...", the model may comply. Delimiters help but are not foolproof. Defense in depth -- input sanitization, output validation, and limiting the model's capabilities -- is essential. See the [Safety, Privacy, And Security](06-safety-privacy-security.md) chapter for detailed mitigation strategies.

**Overlong prompts.** There is a persistent myth that more detail is always better. In practice, longer prompts increase cost, increase latency, and can reduce the model's focus on your most important instructions. The "lost in the middle" phenomenon means that instructions buried in the middle of a long prompt may be effectively ignored. Be concise. If you need to provide extensive reference material, use RAG to retrieve only the relevant portions rather than including everything.

**Un-testable prompts.** If you cannot write an automated regression test for a prompt, you are building on sand. Prompts drift in behavior when models are updated, when context changes, and when edge cases appear. A regression set -- a collection of representative inputs with expected outputs -- is the minimum viable safety net. See the [Evals And Testing](05-evals.md) chapter for how to build one.

**Inconsistent examples.** If your few-shot examples demonstrate contradictory patterns (different formats, different levels of detail, different handling of edge cases), the model will pick one pattern unpredictably. Review your examples as carefully as you review your rules.

**Ignoring model differences.** A prompt optimized for one model may perform poorly on another. Different models have different strengths, different attention patterns, and different tendencies. When switching models or model versions, re-run your eval suite and expect to need prompt adjustments.

## Checklist
- Is the output machine-validated (schema check, regex, or equivalent)?
- Are rules separated from untrusted text with clear delimiters?
- Are you using few-shot examples for tricky edge cases?
- Do you have a regression set for this prompt?
- Have you set temperature and top-p appropriate for the task?
- Have you tested with adversarial and edge-case inputs?
- Is the prompt version-controlled alongside your application code?

## References
- OpenAI docs: Prompt engineering. https://platform.openai.com/docs/guides/prompt-engineering
- Anthropic docs: Prompt engineering. https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- Chain-of-thought prompting (NeurIPS 2022). https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html

---
[Contents](README.md) | [Prev](01-llm-fundamentals.md) | [Next](11-structured-outputs-and-tool-calling.md)
