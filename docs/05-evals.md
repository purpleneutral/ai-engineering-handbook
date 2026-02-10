# Evals And Testing

Last reviewed: 2026-02-10

## Summary
If you cannot measure quality, you cannot improve reliability. Treat prompts and model behavior as code: version, test, and monitor.

## What To Test
- Schema validity:
  - JSON parses, required fields present, types correct.
- Task correctness:
  - Extraction matches ground truth; classifications match labels.
- Robustness:
  - Edge cases, adversarial inputs, long inputs, noisy inputs.
- Safety:
  - Refusal behavior, PII handling, policy constraints.

## Eval Methods
- Golden set regression:
  - A fixed dataset of representative inputs with expected outputs.
- Human review:
  - Best for subjective outputs (tone, helpfulness) and nuanced errors.
- Model-graded evaluation:
  - Useful for scale, but needs calibration and spot checks.

## Practical Tips
- Track changes:
  - Prompt version, model version, tool versions, retrieval index version.
- Prefer small evals that run often:
  - Catch regressions early.
- Separate offline evals from production monitoring:
  - Both are required.

## Checklist
- Do you have a golden set that matches real production distribution?
- Are evals stable across time (or explicitly versioned)?
- Can you bisect regressions to a prompt/model/index change?
- Do you monitor for drift and rising error rates?

## References
- Add evaluation references here.

