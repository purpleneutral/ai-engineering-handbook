# Safety, Privacy, And Security

Last reviewed: 2026-02-10

## Summary
LLM systems are susceptible to data leakage, prompt injection, and unintended actions. Treat them like any other input-driven system, but with a larger attack surface.

## Privacy Basics
- Classify data: public, internal, confidential, regulated (PII/PHI/etc).
- Minimize exposure: send the least data needed for the task.
- Redact or transform: strip secrets/PII before logging or sending to third parties.

## Security Risks
- Prompt injection: untrusted content can attempt to override rules.
- Data exfiltration: model may output sensitive content if it is in context or memory.
- Tool abuse: agents can perform unintended side effects if tools are overpowered.

## Mitigations
- Strong boundaries between instructions and data.
- Allowlist-driven tools and permissions.
- Sandboxed execution for code or file operations.
- Output filtering and schema validation.
- Auditing: tool calls, retrieval ids, decisions, and user actions.

## Checklist
- Are secrets excluded from prompts and logs?
- Are untrusted documents treated as untrusted input?
- Are tools permissioned and auditable?
- Do you have a documented incident response path?

## References
- Add security references here.
