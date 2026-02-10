# Governance And Risk

Last reviewed: 2026-02-10

## Summary
Shipping AI safely is mostly about governance: knowing what system you built, what it can do, how it can fail, and how you control and audit it.

## Risk Areas (Common)
- Safety: harmful content, policy violations, and unsafe instructions.
- Privacy: PII leakage, training data concerns, retention, logging.
- Security: prompt injection, data exfiltration, tool abuse.
- Reliability: hallucinations, brittle formats, regressions after model/prompt updates.
- Compliance: sector-specific rules (varies by org and jurisdiction).

## Practical Governance Moves
### Document The System
- Model/provider, prompt versions, tool inventory, retrieval sources, logging policy.
- Data classification and redaction rules.
- Human-in-the-loop checkpoints for high-impact actions.

### Measure And Monitor
- Offline evals (golden sets) and online monitoring (drift, error rates).
- Track incidents and turn them into regression tests.

### Control Change
- Explicit release process for prompt/model/index updates.
- Rollback plan and a way to reproduce failures (sanitized artifacts).

## Checklist
- Do you have an inventory of tools and permissions?
- Do you have a documented data handling policy (inputs, logs, retention)?
- Do you run evals on every prompt/model/index change?
- Do you have a rollback and incident playbook?

## References
- NIST AI Risk Management Framework (AI RMF 1.0). https://www.nist.gov/itl/ai-risk-management-framework
- NIST Generative AI Profile (GenAI risk guidance). https://www.nist.gov/itl/ai-risk-management-framework/generative-ai-profile
- OWASP Top 10 for LLM Applications. https://owasp.org/www-project-top-10-for-large-language-model-applications/
- OpenAI usage policies. https://openai.com/policies/usage-policies/
- OpenAI "Your data" guide (data handling controls). https://platform.openai.com/docs/guides/your-data

