# Agents

Last reviewed: 2026-02-10

## Summary
An "agent" is an LLM-driven loop that can plan, call tools, observe results, and continue until a goal is met or a budget is exhausted.

## When To Use
- The task needs multiple steps with intermediate tool calls.
- The environment is dynamic (APIs, files, tickets, browsers).
- You need automation, not just answers.

## Key Design Choices
### Tools
- Small, composable, idempotent operations are easier to make reliable.

### State
- Decide what gets persisted (plan, memory, artifacts, logs).

### Budgets
- Token/cost limits, wall-clock time, and tool call limits.

### Autonomy
- Human-in-the-loop checkpoints for high-impact actions.

## Failure Modes
### Infinite Loops / Thrashing
- Agent keeps trying variations without new information.

### Tool Misuse
- Wrong parameters, wrong assumptions, wrong environment.

### Prompt Injection
- Web content and documents can manipulate behavior.

## Safety Boundaries
- Principle of least privilege for tools.
- Explicit allowlists for file paths, network hosts, and actions.
- Sandboxing for code execution.
- Audit logging for tool calls and outputs.

## Checklist
- Are tool effects reversible or safely idempotent?
- Do you have timeouts and retries per tool?
- Do you have loop detection (no progress) and a stop condition?
- Do you log tool calls in a privacy-safe way?

## References
- ReAct paper (reasoning + acting). https://arxiv.org/abs/2210.03629
- Toolformer paper (tool-use training). https://arxiv.org/abs/2302.04761
- OpenAI docs: Function calling. https://platform.openai.com/docs/guides/function-calling
- Anthropic docs: Tool use. https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- OWASP Top 10 for LLM Applications (excessive agency, etc.). https://owasp.org/www-project-top-10-for-large-language-model-applications/
