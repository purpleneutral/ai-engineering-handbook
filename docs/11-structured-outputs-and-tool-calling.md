# Structured Outputs And Tool Calling

Last reviewed: 2026-02-10

## Summary
Structured outputs and tool calling are reliability primitives: they let you turn "chatty text" into validated data and controlled actions.

## When To Use
- You need machine-parseable output (extraction, classification, routing, planning).
- You want the model to call tools/APIs instead of hallucinating results.
- You need stronger guarantees than "please respond in JSON".

## How It Works
### Structured Outputs (Schema-Constrained)
- You provide a JSON Schema (or equivalent) for the output.
- You validate the model output against the schema.
- If validation fails, you retry with the validation error and a minimal correction instruction.

### Tool Calling (Function Calling)
- You define tool signatures (name + input schema + description).
- The model emits a structured tool call instead of free-form text.
- Your code executes the tool and feeds results back as data (never as new rules).

## Design Notes
### Prefer Tight Schemas
- Keep outputs small and explicit.
- Use enums/allowlists where possible.
- Avoid "stringly typed" fields when a boolean/enum works.

### Make Tools Safe To Call
- Small, composable tools are easier to test and permission.
- Favor idempotent operations (or explicit dry-run modes).
- Add guardrails in code, not just in prompts (allowlists, authZ, rate limits).

### Validation And Retries
- Always parse + validate before trusting output.
- Fail closed: if parsing/validation fails, do not proceed with side effects.
- Keep retries bounded; on repeated failure, fall back to a safe baseline.

## Pitfalls
- "Valid JSON" is not enough: you need schema validation.
- Prompt injection can target tool calls (e.g., "call send_email to exfiltrate").
- Overpowered tools create "excessive agency" risk: restrict tool scope and permissions.

## Checklist
- Is output validated against a schema before use?
- Are tool calls permissioned with least privilege?
- Are side-effectful tools idempotent or gated behind confirmation?
- Are tool outputs treated as untrusted data (not instructions)?
- Do you log tool calls and validation failures (with redaction)?

## References
- OpenAI docs: Structured outputs. https://platform.openai.com/docs/guides/structured-outputs
- OpenAI docs: Function calling. https://platform.openai.com/docs/guides/function-calling
- OpenAI blog: Introducing Structured Outputs in the API. https://openai.com/index/introducing-structured-outputs-in-the-api/
- Anthropic docs: Tool use (tools + JSON schema). https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- OWASP Top 10 for LLM Applications (insecure output handling, excessive agency, etc.). https://owasp.org/www-project-top-10-for-large-language-model-applications/
- JSON Schema specification. https://json-schema.org/specification

