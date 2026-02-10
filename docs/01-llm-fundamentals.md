# LLM Fundamentals

Last reviewed: 2026-02-10

## Summary
LLMs are probabilistic text generators (and increasingly multimodal) that can follow instructions, but are not inherently truthful, consistent, or secure.

## Core Ideas
### Tokens And Context Window
- Inputs and outputs are tokens; long context increases cost and latency.
- "Long context" is not the same as "good memory"; retrieval and summarization still matter.

### Non-Determinism
- Sampling (temperature/top-p) changes outputs.
- Even with deterministic settings, some systems are not perfectly repeatable.

### Instruction Following Is Fragile
- Conflicting instructions, ambiguous format requirements, or noisy context reduce reliability.

### Tool Use Changes The Game
- You can offload math, search, and structured data operations to tools.
- Treat tool outputs as data, not "model creativity".

## Common Failure Modes
- Hallucination: confident but incorrect statements.
- Format drift: output not matching a required schema.
- Context distraction: irrelevant context dominates (or "lost-in-the-middle" behavior).
- Prompt injection: malicious or accidental instructions inside untrusted text override your intent.

## Reliability Tactics
### Constrain Outputs
- Use a schema (JSON), strict formatting rules, and keep outputs small.

### Separate Instruction From Data
- Mark untrusted text as "reference" and explicitly forbid it from adding rules.

### Verify And Ground
- Use retrieval for facts; use tools for calculations and queries.
- Add post-checks (schema validation, allowlists, unit tests).

### Design For Retries
- If validation fails, retry with the error message and a minimal diff instruction.

## Checklist
- Define "done": correct, safe, and parseable.
- Decide deterministic vs creative behavior.
- Add validation and fallbacks.
- Log inputs/outputs safely (redact secrets/PII).

## References
- Add primary sources here (vendor docs, papers, specs).
