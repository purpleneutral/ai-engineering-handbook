# Fine-Tuning and Model Customization

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](12-embeddings-and-vector-search.md) | [Next](03-rag.md)

## Summary

Fine-tuning adapts a pre-trained model to your specific task or domain by training on your own data. It changes the model's weights, unlike prompting which only changes the input. The decision to fine-tune is primarily economic and architectural: fine-tuning makes sense when prompting alone cannot achieve the required behavior, or when the cost of long prompts exceeds the cost of training.

## See Also
- [LLM Fundamentals](01-llm-fundamentals.md)
- [Prompting](02-prompting.md)
- [Evals and Testing](05-evals.md)

## When to Fine-Tune (and When Not To)

The most important thing to understand about fine-tuning is that it should not be your first move. Start with prompting and few-shot examples. Measure how far that gets you. Fine-tune only when you have evidence that prompting alone cannot close the gap.

Fine-tuning is appropriate when you need **consistent style or format** that the model cannot reliably produce from instructions alone. If you need every output to follow a precise internal convention -- a specific tone, a particular abbreviation scheme, a report structure unique to your organization -- fine-tuning can bake that behavior into the model's defaults so you do not need to repeat lengthy instructions in every prompt.

Fine-tuning is appropriate when you need **domain-specific vocabulary or jargon**. A base model may not reliably use your company's internal terminology, product names, or industry-specific shorthand. Fine-tuning on examples that demonstrate correct usage teaches the model to speak your language naturally.

Fine-tuning is appropriate when you want to **reduce latency or cost at scale**. A well-fine-tuned model can produce the right output from a short prompt, eliminating the need for long system instructions, many few-shot examples, or chain-of-thought scaffolding. At high call volumes, the savings on input tokens can exceed the one-time cost of training.

Fine-tuning is appropriate when you need a **smaller model to match the behavior of a larger one**. Distillation -- using a large model to generate training data, then fine-tuning a smaller model on that data -- is a legitimate and cost-effective strategy for production workloads where the large model's quality is needed but its latency or price is not acceptable.

Fine-tuning is **not** appropriate when your data changes frequently. If the underlying information shifts weekly or daily, fine-tuning cannot keep up. Use retrieval-augmented generation (RAG) instead, which lets you update the knowledge base without retraining.

Fine-tuning is **not** appropriate when the goal is to give the model access to new factual knowledge. Fine-tuning adjusts behavior, style, and format. It does not reliably teach the model new facts. A fine-tuned model might learn to mention your product's name, but it will not reliably memorize the contents of your documentation. RAG is almost always the right tool for knowledge grounding.

Fine-tuning is **not** appropriate when you have fewer than a few hundred high-quality examples. With too little data, the model either does not learn the pattern or overfits to the specific examples and fails to generalize. If you cannot assemble enough quality data, invest your effort in better prompts instead.

The distinction that matters most: **fine-tuning is for behavior, RAG is for knowledge.** When you need the model to act differently (format, style, tone, vocabulary), fine-tune. When you need the model to know something (facts, documents, current data), retrieve.

## How Fine-Tuning Works

At a practitioner level, fine-tuning is supervised learning on your data, starting from a pre-trained model's weights rather than from scratch.

You prepare a dataset of (input, desired output) pairs that demonstrate the behavior you want. For chat models, these are formatted as conversation-style message arrays -- a sequence of system, user, and assistant messages where the assistant messages represent the ideal responses.

The training process takes the pre-trained model and performs additional gradient updates on your dataset. Each example slightly adjusts the model's weights to make it more likely to produce outputs like your examples when given similar inputs. The result is a new model checkpoint -- a copy of the base model with modified weights -- that behaves differently from the base model in the ways your training data demonstrates.

Critically, fine-tuning does **not** add new factual knowledge reliably. The model's weights encode statistical patterns, not a searchable database. Training on documents containing the fact "our headquarters is in Portland" might make the model more likely to mention Portland in some contexts, but it will not create a reliable lookup. The model may still hallucinate a different city if the prompt context pushes it in another direction. For factual grounding, retrieval is the reliable path.

The base model's general capabilities are preserved during fine-tuning, though they can degrade if the training data is too narrow or the training runs too long. This is the overfitting trade-off: the more you specialize the model, the more you risk degrading its performance on tasks outside your training distribution.

## Data Requirements

The quality of your fine-tuning data is the single most important factor in whether fine-tuning succeeds. This is not a platitude -- it is a practical reality that trips up most teams.

**Quality over quantity.** Fifty to one hundred excellent examples often produce better results than ten thousand noisy ones. Each example should be a clear, unambiguous demonstration of the behavior you want. If a human expert would disagree with the label or find the example confusing, the model will learn confusion. Audit your data manually before training. Read every example. This feels tedious and it is tedious, but it is the highest-leverage activity in the entire fine-tuning process.

**Data format.** For chat model fine-tuning, training data is typically a JSONL file where each line is a conversation represented as a messages array. Each conversation includes the system prompt (if any), one or more user messages, and the assistant responses you want the model to learn. The format varies slightly between providers, but the structure is consistent:

```json
{"messages": [{"role": "system", "content": "Classify the support ticket."}, {"role": "user", "content": "I can't log in to my account"}, {"role": "assistant", "content": "category: account_access"}]}
{"messages": [{"role": "system", "content": "Classify the support ticket."}, {"role": "user", "content": "My invoice amount is wrong"}, {"role": "assistant", "content": "category: billing"}]}
```

**Data diversity.** Your training set should cover the full range of inputs the model will encounter in production. If you fine-tune on only the easy cases, the model will learn the easy pattern and fail on edge cases. Include examples of ambiguous inputs, unusual formats, and boundary conditions. If there are inputs where the correct response is "I don't know" or "this input is out of scope," include those too.

**Data validation.** Before training, check your data for contradictions (two examples with similar inputs but different outputs), formatting errors (malformed JSON, inconsistent schemas), and PII or sensitive data that should not be embedded in model weights. Once you fine-tune on data containing PII, that information is baked into the model and is difficult to remove. Treat your training data with the same care you would treat a database of sensitive records.

**The labeling bottleneck.** Getting high-quality labels is usually the hardest and most expensive part of fine-tuning. Subject matter experts are busy. Labeling guidelines are ambiguous. Inter-annotator agreement is lower than you expect. Budget time and resources for labeling up front, and establish clear labeling criteria before you start. Pilot with a small batch, measure agreement, refine the guidelines, and only then label at scale.

## The Fine-Tuning Process

The practical workflow follows a predictable sequence. Resist the urge to skip steps -- each one exists because teams learned the hard way what happens when you omit it.

**Prepare and validate your dataset.** Clean the data, verify the format, audit for quality, remove PII, and check for contradictions. This step typically takes longer than the actual training.

**Split into training and validation sets.** Hold out 10-20% of your data as a validation set that the model never sees during training. The validation set is your early-warning system for overfitting.

**Upload and start the training job.** Most providers offer an API for this. Upload the training file, specify the base model and any hyperparameters (number of epochs, learning rate multiplier, batch size), and start the job.

A basic example using the OpenAI API:

```python
from openai import OpenAI

client = OpenAI()

# 1. Upload training data (JSONL format)
training_file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune",
)

# 2. Start fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-4o-mini-2024-07-18",
)

print(f"Job ID: {job.id}, Status: {job.status}")

# 3. Check status
job = client.fine_tuning.jobs.retrieve(job.id)
print(f"Status: {job.status}, Fine-tuned model: {job.fine_tuned_model}")
```

**Monitor training metrics.** Watch the training loss (how well the model fits the training data) and validation loss (how well it generalizes to unseen examples). Training loss should decrease steadily. Validation loss should decrease and then plateau. If validation loss starts increasing while training loss continues to decrease, the model is overfitting -- stop training or reduce the number of epochs.

**Evaluate the fine-tuned model against your eval suite.** Do not rely on training metrics alone. Run the fine-tuned model through the same evaluation pipeline you use for prompting -- your golden set, your edge cases, your regression tests. Measure the metrics that matter for your application: accuracy, format compliance, latency, cost.

**Compare against the base model with prompting.** This is the critical comparison. If the base model with a well-crafted prompt achieves 90% of the quality at zero training cost and faster iteration speed, fine-tuning may not be worth the operational overhead. Fine-tuning earns its keep only when it measurably outperforms the best prompt you can write.

**Deploy only if the fine-tuned model wins.** If the numbers do not clearly favor the fine-tuned model, stick with prompting. You can always fine-tune later when you have more data or a clearer signal.

## Evaluation

Evaluating a fine-tuned model follows the same principles as evaluating any LLM-powered system, but with a few additional concerns.

**Always compare against the base model with your best prompt.** This is your baseline. If you skip this comparison, you cannot know whether fine-tuning actually helped. It is surprisingly common for a well-engineered prompt to match or exceed a hastily fine-tuned model.

**Use the same eval suite you would use for prompting.** Your golden set of test inputs with expected outputs, your format compliance checks, your edge case battery -- all of these apply. Fine-tuning does not change what you need to evaluate; it changes what you expect to see in the results.

**Test for regressions on general capability.** Fine-tuning can degrade performance on tasks outside the training distribution. If you fine-tune a model for support ticket classification, test whether it can still summarize documents, answer general questions, and follow novel instructions. Narrowly fine-tuned models can lose breadth, and you need to know if that trade-off is acceptable for your use case.

**Monitor for overfitting.** A model that performs brilliantly on inputs that resemble the training data but poorly on novel inputs is overfitted. The validation set catches some of this during training, but production traffic is the ultimate test. Track performance on a rolling window of real inputs after deployment, and compare it against your held-out eval set.

## Operational Considerations

Fine-tuning introduces operational complexity that prompting does not. Plan for it before you start.

**Model lifecycle.** When the base model is updated by the provider, your fine-tuned model may become deprecated or behave differently. You need a plan for re-fine-tuning on the new base model, which means keeping your training data, evaluation suite, and training configuration versioned and reproducible. Treat a base model update as a trigger for a full re-training and re-evaluation cycle.

**Cost.** Fine-tuning has two cost components: the training cost (a one-time charge based on the number of tokens in your training data and the number of epochs) and the inference cost (fine-tuned models may have different per-token pricing than base models). Model the total cost of ownership, including the human time for data preparation, labeling, evaluation, and ongoing maintenance.

**Version management.** Track which base model was used, which training dataset (with version or hash), which hyperparameters, and which evaluation results. Without this, you cannot reproduce a model, diagnose a regression, or explain to a stakeholder why performance changed.

**Rollback.** Always maintain the ability to fall back to the base model with prompting. If your fine-tuned model degrades after a base model update, or if a bug is found in the training data, you need to be able to revert without downtime. This means keeping your prompt-based implementation alive as a fallback path, not deleting it after fine-tuning succeeds.

## Alternatives to Fine-Tuning

Before committing to fine-tuning, consider whether a simpler approach solves your problem.

**Few-shot prompting** is the cheapest and fastest option. Include a handful of examples directly in the prompt to demonstrate the desired behavior. Iteration is immediate -- you change the prompt and test again. For many tasks, a well-crafted prompt with three to five examples is competitive with fine-tuning. See [Prompting](02-prompting.md).

**Retrieval-augmented generation (RAG)** is the right tool when the model needs access to specific knowledge rather than a change in behavior. RAG lets you update the knowledge base without retraining and provides citation traceability that fine-tuning cannot. See [Retrieval-Augmented Generation](03-rag.md).

**Prompt caching** reduces the cost of repeated long prompts without any model customization. If your system prompt and few-shot examples are identical across many requests, caching avoids re-processing those tokens on every call. This addresses the cost argument for fine-tuning without any of the operational overhead.

**Distillation** uses a large, capable model to generate training data for a smaller, cheaper model. You run the large model on your task, collect high-quality outputs, and fine-tune the small model on those outputs. This is fine-tuning, but the labeling bottleneck is replaced by the large model's capabilities. Distillation is particularly effective when you need the quality of a frontier model at the cost of a smaller one.

## Pitfalls

**Fine-tuning on bad data amplifies the badness.** The model will learn whatever patterns your data contains, including errors, biases, and inconsistencies. If your training data contains incorrect labels, the model will confidently produce incorrect outputs. Data quality is not a nice-to-have; it is the entire game.

**Overfitting to training examples.** With too few examples or too many training epochs, the model memorizes the specific examples rather than learning the general pattern. The result is a model that performs perfectly on inputs that look exactly like the training data and poorly on everything else. Monitor validation loss and test on held-out data to catch this early.

**Losing general capability.** Fine-tuning narrows the model's behavior toward your training distribution, which can come at the expense of its ability to handle novel tasks. A model fine-tuned for classification might become worse at summarization. This trade-off is often acceptable, but you need to measure it rather than assume it is not happening.

**Underestimating the labeling effort.** Teams consistently underestimate how long it takes to produce high-quality training data. Labeling is not a weekend project -- it requires subject matter expertise, clear guidelines, quality checks, and iteration. Budget for it explicitly, and do not cut corners by using low-quality labels to hit a quantity target.

**Fine-tuning for knowledge instead of using RAG.** This is the most common strategic mistake. A team has a knowledge base and wants the model to "know" the information, so they fine-tune on the documents. The result is a model that sometimes mentions the right facts but cannot be trusted to do so reliably, and that cannot be updated without retraining. RAG is almost always the right answer for knowledge grounding.

## Checklist
- Have you tried prompting and few-shot first and measured the gap?
- Is your training data high-quality, diverse, and free of PII?
- Do you have a held-out eval set that matches production distribution?
- Have you compared the fine-tuned model against base model with best prompt?
- Do you have a plan for re-fine-tuning when the base model updates?
- Do you version your training data, base model, and fine-tuned checkpoints?

## References
- OpenAI docs: Fine-tuning. https://platform.openai.com/docs/guides/fine-tuning
- Anthropic docs: Fine-tuning. https://docs.anthropic.com/en/docs/build-with-claude/fine-tuning
- LoRA paper (efficient fine-tuning). https://arxiv.org/abs/2106.09685
- QLoRA paper (quantized fine-tuning). https://arxiv.org/abs/2305.14314

---
[Contents](README.md) | [Prev](12-embeddings-and-vector-search.md) | [Next](03-rag.md)
