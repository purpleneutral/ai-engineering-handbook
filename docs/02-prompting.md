# Prompting

Last reviewed: 2026-02-10

## Summary
Prompting is interface design: you are shaping behavior through constraints, examples, and input hygiene.

## Prompt Anatomy
- Rules: non-negotiable constraints (format, safety boundaries).
- Task: what to do, with success criteria.
- Context: relevant information, clearly marked as data.
- Examples: a small number of representative input-output pairs.

## Practical Guidelines
- Be explicit about output format. "Return JSON matching this schema" beats "respond in JSON".
- Keep instructions short and stable. Move long references into RAG or attachments, not the system prompt.
- Use delimiters for untrusted text. Example tags: `<reference> ... </reference>`.
- Prefer "do" over "don't". State what the model must produce rather than listing dozens of prohibitions.

## Prompt Skeleton
```text
SYSTEM:
You are an assistant that produces strictly valid JSON.
Rules:
1) Output must match the schema below.
2) Only use information from <reference>. If missing, output nulls.

Schema:
{ ... }

USER:
Task: Extract fields from the reference.
<reference>
...
</reference>
```

## Pitfalls
- Prompt injection via untrusted text. Never put untrusted text in the same "instruction channel" as rules.
- Overlong prompts. Longer is not always better; it increases cost and can reduce focus.
- Un-testable prompts. If you cannot write a regression test for a prompt, expect surprise breakage later.

## Checklist
- Is the output machine-validated?
- Are rules separated from untrusted text?
- Are you using few-shot examples for tricky edge cases?
- Do you have a regression set for this prompt?

## References
- Add prompt design references here.
