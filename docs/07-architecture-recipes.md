# Architecture Recipes

Last reviewed: 2026-02-10

## Summary
These are common AI feature shapes and the pieces that usually matter for correctness and reliability.

## Chat Assistant (General Q&A)
- Inputs: conversation + user profile + optional knowledge
- Core: prompting + tool calling + safety checks
- Risks: hallucinations, policy violations, inconsistent tone

### Checklist
- Clear refusal and escalation paths
- Memory policy (what persists, what does not)
- Logging/redaction strategy

## Document Q&A (RAG)
- Inputs: question + retrieved chunks (+ metadata)
- Core: retrieval + reranking + grounded synthesis
- Risks: wrong retrieval, prompt injection in docs, fake citations

### Checklist
- Access control at retrieval time
- Faithfulness evals
- "No answer" behavior for low-confidence retrieval

## Extraction / Structuring
- Inputs: unstructured text
- Core: schema + validation + retries
- Risks: partial extraction, format drift, silent mis-parses

### Checklist
- Strict JSON schema validation
- Test set of tricky formats
- Deterministic settings when possible

## Workflow Automation (Agent)
- Inputs: goal + environment state
- Core: tool loop + budgets + human-in-the-loop
- Risks: unintended actions, tool misuse, infinite loops

### Checklist
- Least-privilege tools
- Idempotency and rollbacks
- Stop conditions and loop detection

## References
- Add architecture references here.
