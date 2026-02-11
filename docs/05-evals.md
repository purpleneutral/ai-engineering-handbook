# Evals And Testing

Last reviewed: 2026-02-10

[Contents](README.md) | [Prev](18-model-context-protocol.md) | [Next](06-safety-privacy-security.md)

## Summary
If you cannot measure quality, you cannot improve reliability. Treat prompts and model behavior as code: version, test, and monitor. Evals are the practice of systematically measuring how well your LLM system performs, and they serve the same role for AI systems that automated tests serve for traditional software. The difference is that LLM outputs are non-deterministic, subjective, and often lack a single correct answer, which makes evaluation harder but not less important.

## See Also
- [Ops: Shipping And Running LLM Systems](08-ops.md)
- [Structured Outputs And Tool Calling](11-structured-outputs-and-tool-calling.md)
- [Retrieval-Augmented Generation (RAG)](03-rag.md)

## Why Evals Are Non-Negotiable

Traditional software either works or it does not. A function returns the right value or it throws an error. LLM systems operate in a much grayer space: outputs can be partially correct, stylistically wrong, technically accurate but unhelpful, or subtly hallucinated in ways that are hard to detect without careful comparison to ground truth.

Without evals, you are flying blind. You cannot answer basic questions like "Did this prompt change make things better or worse?" or "Is the new model version safe to deploy?" or "Are we regressing on edge cases?" Every change to a prompt, model, retrieval index, or tool definition can alter system behavior in unexpected ways. Evals give you the feedback loop that makes iterative improvement possible.

The good news is that evals do not need to be perfect to be useful. A small eval set that runs on every change and catches obvious regressions is vastly better than no eval set at all. You can start simple and add sophistication over time.

## What To Test

A comprehensive eval strategy covers multiple dimensions of quality, each requiring different test approaches and metrics.

**Schema validity** is the most mechanical dimension: does the output parse as valid JSON, does it contain the required fields, are the types correct? For systems using structured outputs, schema validation should be a hard gate. Every response that fails schema validation is a bug, full stop. These tests are cheap to write and run, and they catch a surprising number of issues, especially after model or prompt changes.

**Task correctness** measures whether the system produces the right answer for the right reasons. For extraction tasks, this means comparing extracted fields against human-labeled ground truth. For classification, it means checking labels against a gold standard. For question answering, it means comparing answers to known-correct responses. Correctness metrics vary by task: exact match, F1 score, BLEU, ROUGE, or custom domain-specific measures.

**Robustness** tests whether the system maintains quality under stress. This includes edge cases (very long inputs, very short inputs, empty inputs), adversarial inputs (inputs designed to confuse or exploit the system), noisy inputs (typos, formatting errors, mixed languages), and boundary conditions (inputs at the limits of what the system was designed to handle). Robustness testing is where you discover the failures that will embarrass you in production.

**Safety** testing verifies that the system respects policy constraints. Does it refuse to generate harmful content? Does it handle PII appropriately? Does it comply with usage policies? Does it resist prompt injection? Safety evals are often the most politically sensitive because failures can have real-world consequences, but they are also among the most important.

**Consistency** measures whether the system produces similar outputs for similar inputs. If a user asks the same question twice, the answers should be substantively the same (even if not word-for-word identical). If two paraphrases of the same question produce contradictory answers, that is a problem. Consistency testing is especially important for systems where users interact repeatedly and would notice contradictions.

## Eval Methods

### Golden Set Regression

A golden set is a fixed dataset of representative inputs with expected outputs (or at least expected properties of outputs). Every time you change a prompt, model, tool, or retrieval index, you run the golden set and compare results against the baseline.

Building a good golden set requires thought. The inputs should represent the actual distribution of queries your system receives in production, including common cases, uncommon cases, and known failure modes. The expected outputs should be carefully reviewed by domain experts. A golden set that only covers easy cases provides false confidence.

Size the golden set for your iteration speed. If running the full set takes an hour, you will not run it on every change. Aim for a core set of 50 to 200 cases that runs in minutes for fast feedback, with a larger comprehensive set (500 to 2000 cases) that runs nightly or before releases.

Version the golden set alongside your prompts and code. When you update the golden set (adding cases, correcting labels), treat it as a meaningful change that should be reviewed and documented.

A minimal golden-set eval loop using the OpenAI SDK:

```python
import json

# A minimal golden-set eval loop
golden_set = [
    {"input": "Extract the date: 'Meeting on Jan 5th'", "expected": "2025-01-05"},
    {"input": "Extract the date: 'Due by March 20, 2026'", "expected": "2026-03-20"},
]

from openai import OpenAI
client = OpenAI()

results = []
for case in golden_set:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract the date as YYYY-MM-DD. Return only the date."},
            {"role": "user", "content": case["input"]},
        ],
        temperature=0,
    )
    output = response.choices[0].message.content.strip()
    results.append({"input": case["input"], "expected": case["expected"],
                     "actual": output, "pass": output == case["expected"]})

pass_rate = sum(r["pass"] for r in results) / len(results)
print(f"Pass rate: {pass_rate:.0%}")
```

### Human Review

Human evaluation is the gold standard for subjective quality dimensions: tone, helpfulness, clarity, cultural appropriateness, and overall user experience. No automated metric fully captures whether an answer "feels right" to a human reader.

Structure human review to be efficient and consistent. Use rating rubrics with clear criteria and examples for each score level. Have multiple reviewers evaluate the same outputs to measure inter-rater agreement. Disagreements between reviewers often reveal genuinely ambiguous cases that deserve discussion.

Human review does not scale to every change, so use it strategically: for major prompt rewrites, model migrations, and periodic quality audits. Between human reviews, rely on automated metrics to catch regressions, but remain aware that automated metrics can miss quality issues that humans would catch immediately.

### Model-Graded Evaluation

Using one LLM to evaluate another LLM's output --- sometimes called [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) --- is increasingly common and can provide a useful middle ground between fully automated metrics and expensive human review. A "judge" model receives the input, the system's output, and evaluation criteria, then produces a quality score or a pass/fail determination.

Model-graded evaluation has real strengths: it scales well, it can assess nuanced quality dimensions that simple string matching cannot, and it can provide explanations for its scores. However, it also has significant pitfalls that must be managed carefully.

**Calibration.** The judge model's scores must correlate with human judgment. Before relying on model-graded evals, run a calibration study: have both humans and the judge model evaluate the same set of outputs, and measure agreement. If the judge consistently disagrees with humans on certain types of outputs, its scores are noise rather than signal for those cases.

**Bias.** LLM judges have known biases. They tend to prefer longer answers, more formal language, and answers that agree with their own training data. They may rate their own model's outputs more favorably than outputs from other models. Be aware of these biases and design your evaluation prompts to mitigate them (for example, by evaluating answers without revealing which model produced them).

**Spot checks.** Even after calibration, continue performing random spot checks on model-graded evaluations. Models can develop blind spots that only become apparent when a human reviews the specific outputs being evaluated. A 5 to 10 percent human review rate on model-graded evals is a reasonable ongoing investment.

**Criteria specificity.** Vague evaluation criteria ("Is this a good answer?") produce unreliable scores. Specific criteria ("Does the answer address all three parts of the question? Does it cite sources? Is it free of factual errors?") produce more consistent and more useful evaluations.

## A/B Testing and Online Evaluation

Offline evals tell you how a system performs on a test set. Online evaluation tells you how it performs with real users on real queries, which is ultimately what matters.

**A/B testing** for LLM systems follows the same principles as A/B testing for any software: split traffic between variants, measure outcomes, and use statistical tests to determine whether differences are significant. The challenge for LLM systems is choosing the right outcome metrics. Click-through rate and task completion rate are useful proxies, but they do not capture everything. A user might complete a task with a technically correct but unhelpful answer by working around the system's deficiencies.

**Shadow evaluation** runs the new system variant on production traffic but does not show its outputs to users. Instead, you log the outputs and evaluate them offline (with automated metrics or human review) against the current production system's outputs. This lets you assess quality without any risk to users.

**Interleaving** is a technique borrowed from search engine evaluation: show results from both variants in a single response and measure which results users prefer or engage with. This is most applicable to systems that produce lists of results (like search or recommendation) rather than single answers.

Regardless of the method, ensure that your online evaluation captures enough data to detect regressions on subpopulations. An overall improvement can mask a regression for a specific user segment, query type, or language.

## Production Monitoring vs. Offline Evals

Offline evals and production monitoring serve different purposes and neither can replace the other.

**Offline evals** are controlled experiments. You know the inputs, you know the expected outputs, and you can run them repeatedly under identical conditions. They are essential for catching regressions before deployment and for systematic quality improvement. Their limitation is that they only test what you thought to include in the test set. The real world will always produce queries you did not anticipate.

**Production monitoring** catches what offline evals miss: novel query patterns, distribution shifts, infrastructure issues, and the long tail of real-world usage. Key production metrics include error rates (schema validation failures, empty responses, timeouts), latency distributions (not just averages but p95 and p99), user feedback signals (thumbs up/down, explicit complaints, task abandonment rates), and content safety flags.

Set up alerts for significant changes in these metrics. A sudden spike in schema validation failures might indicate a model API change. A gradual increase in user complaints might indicate distribution drift in incoming queries. A latency increase might indicate a retrieval index that has grown too large.

Connect production monitoring back to your eval pipeline. When production monitoring surfaces a failure, create a reproduction case, add it to your golden set, and verify that your system handles it correctly after the fix. This feedback loop is how your eval set evolves to match the reality of production traffic.

## Practical Tips

**Track changes rigorously.** Every eval result should be associated with a specific prompt version, model version, tool version, retrieval index version, and eval set version. Without this, you cannot attribute quality changes to their causes. A spreadsheet is a fine starting point; a proper experiment tracking system is better for teams.

**Prefer small evals that run often.** A 50-case eval that runs on every commit catches regressions within hours. A 2000-case eval that runs quarterly catches them months too late. Both have their place, but fast feedback is more valuable than comprehensive feedback for day-to-day development.

**Invest in eval infrastructure early.** The teams that move fastest on LLM quality are the ones that made it trivially easy to run evals. If running an eval requires manual setup, custom scripts, and an hour of waiting, developers will skip it. If it runs automatically in CI and posts results to a dashboard, it becomes part of the workflow.

**Measure what matters to users, not what is easy to measure.** Exact string match is easy to compute but rarely what users care about. If your users care about helpfulness, measure helpfulness (even if that requires human review or a calibrated model judge). The eval metrics you optimize for will shape the system you build.

## Checklist
- Do you have a golden set that matches the real production distribution?
- Are evals stable across time (or explicitly versioned when they change)?
- Can you bisect regressions to a specific prompt, model, or index change?
- Do you monitor for drift and rising error rates in production?
- Are model-graded evaluations calibrated against human judgment?
- Do you have both offline evals (pre-deployment) and online monitoring (post-deployment)?
- Is running an eval easy enough that developers actually do it?
- Do production failures feed back into the golden set?

## References
- MT-Bench / LLM-as-a-judge (Zheng et al.). https://arxiv.org/abs/2306.05685
- OpenAI Evals (framework). https://github.com/openai/evals
- Promptfoo (prompt testing). https://github.com/promptfoo/promptfoo

---
[Contents](README.md) | [Prev](18-model-context-protocol.md) | [Next](06-safety-privacy-security.md)
